/* BW1000 single-file HIP program: global/memory
 * Compile: hipcc -O3 -std=c++17 --offload-arch=gfx936 memory.hip -o bench
 * Run: ./bench --mode latency_one --bytes 4K --width 1 --policy default --repeats 30
 * Minimal run (NOT a timing reproduction): ./bench --mode latency_one --bytes 4K --width 1 --repeats 1
 * No local headers, private libraries, scripts, data files or other source files required.
 * Only public HIP/DTK, its driver/runtime, and the C++ standard library are required.
 * Prints in-process summaries in timer tick and calibrated ns. BW1000_RAW=1 also prints samples.
 * Original measurement core: vgpr/src/global_mem_bench.cpp; SHA256=15b2eb8457522a5fa015780fdfef8fda07ca2ac15b860692a30adfd7aeae2385
 * Host-only additions: calibration, sample grouping, original normalization/statistics, defaults.
 * Historical published measurements remain in README/results.csv; new runs do not replace them.
 */
// Self-contained host-side CSV observation and statistics. No subprocesses or private includes.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <strings.h>
#include <tuple>
#include <vector>

namespace bw {
using Row=std::map<std::string,std::string>;
using Key=std::array<std::string,4>;
static std::vector<Row> rows;
static std::vector<std::string> header;
static std::string pending;
static double mhz=0;
static std::map<Key,std::vector<double>> measurements;
static double median(std::vector<double> v) {
  if(v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(),v.end()); size_t n=v.size();return n%2?v[n/2]:(v[n/2-1]+v[n/2])/2;
}
static std::vector<std::string> split(const std::string&s) {
  std::vector<std::string> out; std::string x; bool quote=false;
  for(size_t i=0;i<s.size();++i) {char c=s[i];if(c=='"'){if(quote&&i+1<s.size()&&s[i+1]=='"'){x+='"';++i;}else quote=!quote;}else if(c==','&&!quote){out.push_back(x);x.clear();}else x+=c;}out.push_back(x);return out;
}
static std::string str(const Row&r,const std::string&k,const std::string&fallback="") {auto i=r.find(k);return i==r.end()?fallback:i->second;}
static double num(const Row&r,const std::string&k,double fallback=std::numeric_limits<double>::quiet_NaN()) {auto i=r.find(k);if(i==r.end()||i->second.empty())return fallback;char*end=nullptr;double v=std::strtod(i->second.c_str(),&end);return end!=i->second.c_str()&&!*end?v:fallback;}
static std::string quoted(const std::string&s) {std::string out="\"";for(char c:s){out+=c;if(c=='"')out+='"';}return out+'"';}
static void observe(const std::string&line) {
  if(std::getenv("BW1000_RAW"))std::puts(line.c_str());
  auto v=split(line);if(v.empty())return;
  if(v[0]=="suite"||v[0]=="category"||v[0]=="config"||v[0]=="mode"||v[0]=="method"){header=v;return;}
  if(v.size()!=header.size())return;
  Row r;for(size_t i=0;i<v.size();++i)r[header[i]]=v[i];rows.push_back(std::move(r));
}
static int emit(const char*fmt,...) {
  va_list a,b;va_start(a,fmt);va_copy(b,a);int n=std::vsnprintf(nullptr,0,fmt,b);va_end(b);
  if(n<0){va_end(a);return n;}std::vector<char>s(n+1);std::vsnprintf(s.data(),s.size(),fmt,a);va_end(a);pending.append(s.data(),n);
  size_t p;while((p=pending.find('\n'))!=std::string::npos){observe(pending.substr(0,p));pending.erase(0,p+1);}return n;
}
static int line(const char*s){emit("%s\n",s);return 0;}
static std::string conditions(const Row&r,const std::set<std::string>&omit={}) {
  std::string s;for(const char*k:{"pattern","operation","selector","streams","waves","state","policy","address_mode","working_set_bytes","actual_bytes","stride_bytes","width","param_name","param","conflict","n","independent_requests","rounds","count","alignment","path","op","threshold","age","payload","skew","inner_reps","variant","l2_state","access_pattern","independent_instructions"})if(!omit.count(k)&&r.count(k)){if(!s.empty())s+=';';s+=std::string(k)+'='+r.at(k);}return s;
}
static void add(const std::string&t,const std::string&c,const std::string&m,const std::string&d,double v){if(std::isfinite(v))measurements[{t,c,m,d}].push_back(v);}
static std::string instruction(const Row&r) {
  if(r.count("instruction"))return r.at("instruction");
  std::string p=str(r,"path","global"),op=str(r,"op","load");int w=int(num(r,"width",4));
  std::string suffix=w==1?(op=="load"?"ubyte":"byte"):w==2?(op=="load"?"ushort":"short"):w==4?"dword":"dwordx"+std::to_string(w/4);
  return p+'_'+op+'_'+suffix;
}
static double slope(const std::vector<std::pair<double,double>>&p) {
  if(p.size()<2)return std::numeric_limits<double>::quiet_NaN();double x=0,y=0;for(auto a:p){x+=a.first;y+=a.second;}x/=p.size();y/=p.size();double d=0,n=0;for(auto a:p){d+=(a.first-x)*(a.first-x);n+=(a.first-x)*(a.second-y);}return d?n/d:std::numeric_limits<double>::quiet_NaN();
}
// Same clock64/HIP-event calibration method as the archived MMAC calibration.
__global__ void calibration(unsigned long long*out) {
  if(threadIdx.x)return;auto a=clock64(),b=a;do{b=clock64();}while(b-a<1500000000ULL);out[blockIdx.x]=b-a;
}
static void check(hipError_t e){if(e!=hipSuccess){std::fprintf(stderr,"HIP: %s\n",hipGetErrorString(e));std::exit(2);}}
static void start() {
  check(hipSetDevice(0));hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));
  if(p.warpSize!=64||std::strncmp(p.gcnArchName,"gfx936",6)){std::fprintf(stderr,"requires gfx936 Wave64, got %s Wave%d\n",p.gcnArchName,p.warpSize);std::exit(3);}
  unsigned long long*d=nullptr;check(hipMalloc(&d,p.multiProcessorCount*sizeof(*d)));hipEvent_t a,b;check(hipEventCreate(&a));check(hipEventCreate(&b));
  check(hipEventRecord(a));hipLaunchKernelGGL(calibration,dim3(p.multiProcessorCount),dim3(64),0,0,d);check(hipGetLastError());check(hipEventRecord(b));check(hipEventSynchronize(b));float ms=0;check(hipEventElapsedTime(&ms,a,b));
  std::vector<unsigned long long>h(p.multiProcessorCount);check(hipMemcpy(h.data(),d,h.size()*sizeof(*d),hipMemcpyDeviceToHost));std::sort(h.begin(),h.end());mhz=h[h.size()/2]/(ms*1000.0);
  check(hipEventDestroy(a));check(hipEventDestroy(b));check(hipFree(d));if(!(mhz>0)){std::fprintf(stderr,"invalid calibration\n");std::exit(4);}
  std::printf("# device=%s arch=%s Wave%d timer_tick_MHz=%.9f; ns=tick*1000/MHz; NOT proved core cycles\n",p.name,p.gcnArchName,p.warpSize,mhz);
}
static void report() {
  std::puts("kind,target,condition,metric,denominator,ticks,ns,samples,min_ticks,max_ticks,min_ns,max_ns");
  for(auto&x:measurements){auto&k=x.first;auto&v=x.second;double t=median(v);std::printf("SUMMARY,%s,%s,%s,%s,%.9f,%.9f,%zu,%.9f,%.9f,%.9f,%.9f\n",quoted(k[0]).c_str(),quoted(k[1]).c_str(),quoted(k[2]).c_str(),quoted(k[3]).c_str(),t,t*1000/mhz,v.size(),*std::min_element(v.begin(),v.end()),*std::max_element(v.begin(),v.end()),*std::min_element(v.begin(),v.end())*1000/mhz,*std::max_element(v.begin(),v.end())*1000/mhz);}
}
// Inline reducers retain the archived paired-vs-unpaired and normalization definitions.
static void reduce(const std::string&family) {
  std::map<std::tuple<std::string,std::string,std::string>,std::map<int,std::vector<double>>> batches;
  std::vector<double> baseline;
  for(auto&r:rows){if(str(r,"config")=="baseline")baseline.push_back(num(r,"raw_ticks"));if(str(r,"instruction")=="s_waitcnt_lgkmcnt0")baseline.push_back(num(r,"completion_ticks"));if(str(r,"suite")=="baseline"&&r.count("completion_ticks"))baseline.push_back(num(r,"completion_ticks"));}
  const double base=median(baseline);
  for(auto&r:rows){std::string c=conditions(r),t=instruction(r),s=str(r,"suite",str(r,"category",str(r,"mode")));auto put=[&](const std::string&m,const std::string&d,double v){add(t,c,m,d,v);};
    double gbps=num(r,"GBps",num(r,"GB_per_s",num(r,"bandwidth_GBps",0)));
    if(gbps>0&&num(r,"event_ms",0)>0){
      put("kernel_event_duration_tick_equivalent","whole kernel; tick equivalent from HIP event ns",num(r,"event_ms")*1e6*mhz/1000);
      std::printf("BANDWIDTH,%s,%.9f,GB/s\n",t.c_str(),gbps);continue;
    }
    if(family=="global_memory") {
      if(r.count("adjusted_cycles_per_load"))put("dependent_adjusted","load",num(r,"adjusted_cycles_per_load"));
      else if(s=="load_issue"||s=="store_issue") {double n=num(r,"independent_requests");put("issue_observed","request",num(r,"issue_cycles_per_instruction"));put("completion","request",num(r,"completion_cycles_per_instruction"));batches[{t,"op="+s,"issue_fit"}][int(n)].push_back(num(r,"issue_cycles_per_instruction")*n);}
      else if(r.count("GB_per_s"))std::printf("BANDWIDTH,%s,%.9f,GB/s\n",t.c_str(),num(r,"GB_per_s"));
      else if(s=="timer_calibration")put("calibration_interval","bracket",num(r,"ticks"));
      continue;
    }
  }
  for(auto&g:batches){auto[t,c,mode]=g.first;auto&vs=g.second;std::vector<std::pair<double,double>>p;for(int n:{4,8,16})if(vs.count(n))p.emplace_back(n,median(vs[n]));
    if(mode=="wave_fit") {p.clear();for(auto&z:vs)p.emplace_back(1.0/z.first,median(z.second));double sl=slope(p),xm=0,ym=0;for(auto z:p){xm+=z.first;ym+=z.second;}if(p.size()>=3)add(t,c,"issue_plateau_fit","native instruction",ym/p.size()-sl*xm/p.size());}
    else if(mode=="issue_fit") {add(t,c,"issue_slope_N4_8_16","request",slope(p));if(vs.count(1)&&vs.count(2))add(t,c,"issue_initial_difference","request",median(vs[2])-median(vs[1]));std::vector<std::pair<double,double>>low;for(int n:{1,2,4})if(vs.count(n))low.emplace_back(n,median(vs[n]));add(t,c,"issue_low_load_slope","request",slope(low));}
    else if(mode=="barrier_base") {auto j=batches.find({t,c,"barrier_control"});if(j!=batches.end())for(auto&z:vs)if(j->second.count(z.first))add(t,c+";inner_reps="+std::to_string(z.first),"matched_control_net","barrier",(median(z.second)-median(j->second[z.first]))/z.first);}
    else if(mode=="event_wait") {auto j=batches.find({t,c,"event_control"});if(j!=batches.end()){p.clear();for(auto&z:vs)if(j->second.count(z.first)){
          std::map<int,double> controls,targets;
          for(auto&r:rows)if("s_waitcnt "+str(r,"counter")==t&&"threshold="+str(r,"threshold")==c&&num(r,"rounds")==z.first){
            if(str(r,"method")=="event_control")controls[int(num(r,"repeat"))]=num(r,"event_ms")*1e6;
            if(str(r,"method")=="event_wait")targets[int(num(r,"repeat"))]=num(r,"event_ms")*1e6;
          }
          std::vector<double>deltas;for(auto&sample:targets)if(controls.count(sample.first))deltas.push_back(sample.second-controls[sample.first]);
          if(!deltas.empty())p.emplace_back(z.first*256.0,median(deltas));
        }double ns=slope(p);add(t,c,"event_only_paired_slope_tick_equivalent","wait (tick equivalent from event ns)",ns*mhz/1000);}}
    else if(mode=="clock_drained"&&vs.count(0)&&vs.count(1))add(t,c,"matched_single_increment","wait (timer-observed increment)",median(vs[1])-median(vs[0]));
  }
}
static int finish(const std::string&family) {reduce(family);report();if(measurements.empty()){if(!rows.empty()&&str(rows.front(),"category")=="correctness"){std::puts("FUNCTIONAL_ONLY: no latency claimed");return 0;}std::fprintf(stderr,"no timing summary generated\n");return 8;}return 0;}
} // namespace bw

// ---- Original measurement implementation (host additions noted inline) ----


using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));

#define HIP_CHECK(call) do { \
  hipError_t _e = (call); \
  if (_e != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s:%d: %s -> %s\n", __FILE__, __LINE__, #call, hipGetErrorString(_e)); \
    std::exit(2); \
  } \
} while (0)

static constexpr int kRepeats = 30;
static constexpr int kBlock = 256;

enum class Policy : int { Default = 0, Glc = 1, Slc = 2, GlcSlc = 3 };

static const char* policy_name(Policy p) {
  switch (p) {
    case Policy::Default: return "default";
    case Policy::Glc: return "glc";
    case Policy::Slc: return "slc";
    case Policy::GlcSlc: return "glc_slc";
  }
  return "unknown";
}

static uint64_t parse_bytes(const std::string& text) {
  if (text.empty()) return 0;
  char* end = nullptr;
  double v = std::strtod(text.c_str(), &end);
  if (end == text.c_str()) {
    std::fprintf(stderr, "invalid byte count: %s\n", text.c_str());
    std::exit(2);
  }
  uint64_t mul = 1;
  if (*end) {
    if (!strcasecmp(end, "k") || !strcasecmp(end, "kb") || !strcasecmp(end, "kib")) mul = 1024ull;
    else if (!strcasecmp(end, "m") || !strcasecmp(end, "mb") || !strcasecmp(end, "mib")) mul = 1024ull * 1024ull;
    else if (!strcasecmp(end, "g") || !strcasecmp(end, "gb") || !strcasecmp(end, "gib")) mul = 1024ull * 1024ull * 1024ull;
    else {
      std::fprintf(stderr, "invalid byte suffix: %s\n", end);
      std::exit(2);
    }
  }
  return static_cast<uint64_t>(v * static_cast<double>(mul));
}

static std::vector<uint64_t> default_scan_sizes() {
  const uint64_t K = 1024ull, M = K * K;
  return {
    256, 512, 1*K, 2*K, 4*K, 8*K, 12*K, 16*K, 24*K, 32*K, 48*K, 64*K,
    96*K, 128*K, 192*K, 256*K, 384*K, 512*K, 768*K, 1*M, 1536*K, 2*M,
    3*M, 4*M, 6*M, 8*M, 12*M, 16*M, 24*M, 32*M, 48*M, 64*M,
    96*M, 128*M, 192*M, 256*M, 384*M, 512*M
  };
}

static std::vector<uint64_t> default_width_sizes() {
  const uint64_t K = 1024ull, M = K * K;
  return {1*K, 4*K, 16*K, 64*K, 256*K, 1*M, 4*M, 16*M, 64*M, 256*M};
}

#define DEFINE_LOAD2_POLICY(fn, mods) \
__device__ __forceinline__ u32x2 fn(const u32x2* p) { \
  u32x2 v; \
  const uint64_t addr = reinterpret_cast<uint64_t>(p); \
  asm volatile("global_load_dwordx2 %0, %1, off " mods "\n\ts_waitcnt vmcnt(0)" \
               : "=v"(v) : "v"(addr) : "memory"); \
  return v; \
}
DEFINE_LOAD2_POLICY(global_load2_default, "")
DEFINE_LOAD2_POLICY(global_load2_glc, "glc")
DEFINE_LOAD2_POLICY(global_load2_slc, "slc")
DEFINE_LOAD2_POLICY(global_load2_glc_slc, "glc slc")
#undef DEFINE_LOAD2_POLICY

#define DEFINE_LOAD4_POLICY(fn, mods) \
__device__ __forceinline__ u32x4 fn(const u32x4* p) { \
  u32x4 v; \
  const uint64_t addr = reinterpret_cast<uint64_t>(p); \
  asm volatile("global_load_dwordx4 %0, %1, off " mods "\n\ts_waitcnt vmcnt(0)" \
               : "=v"(v) : "v"(addr) : "memory"); \
  return v; \
}
DEFINE_LOAD4_POLICY(global_load4_default, "")
DEFINE_LOAD4_POLICY(global_load4_glc, "glc")
DEFINE_LOAD4_POLICY(global_load4_slc, "slc")
DEFINE_LOAD4_POLICY(global_load4_glc_slc, "glc slc")
#undef DEFINE_LOAD4_POLICY

#define DEFINE_LOAD1_POLICY(fn, mods) \
__device__ __forceinline__ uint32_t fn(const uint32_t* p) { \
  uint32_t v; \
  const uint64_t addr = reinterpret_cast<uint64_t>(p); \
  asm volatile("global_load_dword %0, %1, off " mods "\n\ts_waitcnt vmcnt(0)" \
               : "=v"(v) : "v"(addr) : "memory"); \
  return v; \
}
DEFINE_LOAD1_POLICY(global_load1_default, "")
DEFINE_LOAD1_POLICY(global_load1_glc, "glc")
DEFINE_LOAD1_POLICY(global_load1_slc, "slc")
DEFINE_LOAD1_POLICY(global_load1_glc_slc, "glc slc")
#undef DEFINE_LOAD1_POLICY

__device__ __forceinline__ uint32_t chase_load1(const uint8_t* base, uint32_t byte_offset, Policy p) {
  const uint32_t* addr = reinterpret_cast<const uint32_t*>(base + byte_offset);
  switch (p) {
    case Policy::Default: return global_load1_default(addr);
    case Policy::Glc: return global_load1_glc(addr);
    case Policy::Slc: return global_load1_slc(addr);
    case Policy::GlcSlc: return global_load1_glc_slc(addr);
  }
  return 0;
}

struct TickResult {
  uint64_t ticks;
  uint64_t aux_ticks;
  uint32_t value;
  uint32_t pad;
};

#define DEFINE_POINTER_CHASE_DWORD(kernel_name, load_fn) \
extern "C" __global__ void kernel_name(const uint8_t* base, uint32_t start, \
                                         uint64_t warm_steps, uint64_t steps, TickResult* out) { \
  if (blockIdx.x != 0 || threadIdx.x != 0) return; \
  uint32_t p = start; \
  uint32_t sink = 0; \
  _Pragma("clang loop unroll(disable)") \
  for (uint64_t i = 0; i < warm_steps; ++i) { \
    p = load_fn(reinterpret_cast<const uint32_t*>(base + p)); sink ^= p; \
  } \
  asm volatile("" ::: "memory"); \
  const uint64_t begin = clock64(); \
  _Pragma("clang loop unroll(disable)") \
  for (uint64_t i = 0; i < steps; ++i) { \
    p = load_fn(reinterpret_cast<const uint32_t*>(base + p)); \
    sink ^= p + static_cast<uint32_t>(i); \
  } \
  const uint64_t end = clock64(); \
  out[0] = TickResult{end - begin, 0, p ^ sink, 0}; \
  return; \
}
DEFINE_POINTER_CHASE_DWORD(pointer_chase_dword_default, global_load1_default)
DEFINE_POINTER_CHASE_DWORD(pointer_chase_dword_glc, global_load1_glc)
DEFINE_POINTER_CHASE_DWORD(pointer_chase_dword_slc, global_load1_slc)
DEFINE_POINTER_CHASE_DWORD(pointer_chase_dword_glc_slc, global_load1_glc_slc)
#undef DEFINE_POINTER_CHASE_DWORD

#define DEFINE_POINTER_CHASE_DWORDX2(kernel_name, load_fn) \
extern "C" __global__ void kernel_name(const uint8_t* base, uint32_t start, uint64_t warm_steps, uint64_t steps, TickResult* out) { \
  if (blockIdx.x != 0 || threadIdx.x != 0) return; \
  uint32_t p = start, sink = 0; \
  _Pragma("clang loop unroll(disable)") \
  for (uint64_t i = 0; i < warm_steps; ++i) { \
    const u32x2 v = load_fn(reinterpret_cast<const u32x2*>(base + p)); p = v[0]; sink ^= v[1]; \
  } \
  const uint64_t begin = clock64(); \
  _Pragma("clang loop unroll(disable)") \
  for (uint64_t i = 0; i < steps; ++i) { \
    const u32x2 v = load_fn(reinterpret_cast<const u32x2*>(base + p)); \
    p = v[0]; sink ^= v[1] + static_cast<uint32_t>(i); \
  } \
  const uint64_t end = clock64(); out[0] = TickResult{end - begin, 0, p ^ sink, 0}; return; \
}
DEFINE_POINTER_CHASE_DWORDX2(pointer_chase_dwordx2_default, global_load2_default)
DEFINE_POINTER_CHASE_DWORDX2(pointer_chase_dwordx2_glc, global_load2_glc)
DEFINE_POINTER_CHASE_DWORDX2(pointer_chase_dwordx2_slc, global_load2_slc)
DEFINE_POINTER_CHASE_DWORDX2(pointer_chase_dwordx2_glc_slc, global_load2_glc_slc)
#undef DEFINE_POINTER_CHASE_DWORDX2

#define DEFINE_POINTER_CHASE_DWORDX4(kernel_name, load_fn) \
extern "C" __global__ void kernel_name(const uint8_t* base, uint32_t start, uint64_t warm_steps, uint64_t steps, TickResult* out) { \
  if (blockIdx.x != 0 || threadIdx.x != 0) return; \
  uint32_t p = start, sink = 0; \
  _Pragma("clang loop unroll(disable)") \
  for (uint64_t i = 0; i < warm_steps; ++i) { \
    const u32x4 v = load_fn(reinterpret_cast<const u32x4*>(base + p)); p = v[0]; sink ^= v[1] + v[2] + v[3]; \
  } \
  const uint64_t begin = clock64(); \
  _Pragma("clang loop unroll(disable)") \
  for (uint64_t i = 0; i < steps; ++i) { \
    const u32x4 v = load_fn(reinterpret_cast<const u32x4*>(base + p)); \
    p = v[0]; sink ^= v[1] + v[2] + v[3] + static_cast<uint32_t>(i); \
  } \
  const uint64_t end = clock64(); out[0] = TickResult{end - begin, 0, p ^ sink, 0}; return; \
}
DEFINE_POINTER_CHASE_DWORDX4(pointer_chase_dwordx4_default, global_load4_default)
DEFINE_POINTER_CHASE_DWORDX4(pointer_chase_dwordx4_glc, global_load4_glc)
DEFINE_POINTER_CHASE_DWORDX4(pointer_chase_dwordx4_slc, global_load4_slc)
DEFINE_POINTER_CHASE_DWORDX4(pointer_chase_dwordx4_glc_slc, global_load4_glc_slc)
#undef DEFINE_POINTER_CHASE_DWORDX4

extern "C" __global__ void pointer_baseline(uint32_t start, uint32_t stride,
                                              uint32_t span, uint64_t steps,
                                              TickResult* out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  uint32_t p = start;
  uint32_t sink = 0;
  const uint64_t begin = clock64();
#pragma clang loop unroll(disable)
  for (uint64_t i = 0; i < steps; ++i) {
    p += stride;
    if (p >= span) p -= span;
    sink ^= p + static_cast<uint32_t>(i);
  }
  const uint64_t end = clock64();
  out[0] = TickResult{end - begin, 0, p ^ sink, 0};
  return;
}

extern "C" __global__ void warmup_kernel(uint64_t iters, uint64_t* out) {
  uint64_t x = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
#pragma clang loop unroll(disable)
  for (uint64_t i = 0; i < iters; ++i) x = x * 1664525ull + 1013904223ull;
  if (threadIdx.x == 0) out[blockIdx.x] = x;
  return;
}

extern "C" __global__ void timer_calibration_kernel(uint64_t iters, TickResult* out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  uint32_t x = 1;
  const uint64_t begin = clock64();
#pragma clang loop unroll(disable)
  for (uint64_t i = 0; i < iters; ++i) x = x * 1664525u + 1013904223u;
  const uint64_t end = clock64();
  out[0] = TickResult{end - begin, 0, x, 0};
  return;
}

static void launch_chase(int width, const uint8_t* base, uint32_t start, uint64_t warm_steps,
                         uint64_t steps, Policy policy, TickResult* out, hipStream_t stream = nullptr) {
  if (width == 1) {
    if (policy == Policy::Default) hipLaunchKernelGGL(pointer_chase_dword_default, dim3(1), dim3(64), 0, stream, base, start, warm_steps, steps, out);
    else if (policy == Policy::Glc) hipLaunchKernelGGL(pointer_chase_dword_glc, dim3(1), dim3(64), 0, stream, base, start, warm_steps, steps, out);
    else if (policy == Policy::Slc) hipLaunchKernelGGL(pointer_chase_dword_slc, dim3(1), dim3(64), 0, stream, base, start, warm_steps, steps, out);
    else hipLaunchKernelGGL(pointer_chase_dword_glc_slc, dim3(1), dim3(64), 0, stream, base, start, warm_steps, steps, out);
  } else if (width == 2) {
    if (policy == Policy::Default) hipLaunchKernelGGL(pointer_chase_dwordx2_default, dim3(1), dim3(64), 0, stream, base, start, warm_steps, steps, out);
    else if (policy == Policy::Glc) hipLaunchKernelGGL(pointer_chase_dwordx2_glc, dim3(1), dim3(64), 0, stream, base, start, warm_steps, steps, out);
    else if (policy == Policy::Slc) hipLaunchKernelGGL(pointer_chase_dwordx2_slc, dim3(1), dim3(64), 0, stream, base, start, warm_steps, steps, out);
    else hipLaunchKernelGGL(pointer_chase_dwordx2_glc_slc, dim3(1), dim3(64), 0, stream, base, start, warm_steps, steps, out);
  } else if (width == 4) {
    if (policy == Policy::Default) hipLaunchKernelGGL(pointer_chase_dwordx4_default, dim3(1), dim3(64), 0, stream, base, start, warm_steps, steps, out);
    else if (policy == Policy::Glc) hipLaunchKernelGGL(pointer_chase_dwordx4_glc, dim3(1), dim3(64), 0, stream, base, start, warm_steps, steps, out);
    else if (policy == Policy::Slc) hipLaunchKernelGGL(pointer_chase_dwordx4_slc, dim3(1), dim3(64), 0, stream, base, start, warm_steps, steps, out);
    else hipLaunchKernelGGL(pointer_chase_dwordx4_glc_slc, dim3(1), dim3(64), 0, stream, base, start, warm_steps, steps, out);
  } else { std::fprintf(stderr, "unsupported width %d\n", width); std::exit(2); }
  HIP_CHECK(hipGetLastError());
}

static void build_pointer_cycle(std::vector<uint8_t>& host, uint64_t working_set,
                                uint32_t stride, int width, uint32_t seed,
                                uint32_t* start_out) {
  const uint64_t min_node = std::max<uint64_t>(stride, static_cast<uint64_t>(width * 4));
  const uint64_t nodes = std::max<uint64_t>(2, working_set / min_node);
  const uint64_t actual = nodes * min_node;
  if (actual > std::numeric_limits<uint32_t>::max()) {
    std::fprintf(stderr, "pointer offsets are 32-bit; working set too large: %llu\n",
                 static_cast<unsigned long long>(actual));
    std::exit(2);
  }
  host.assign(static_cast<size_t>(actual), 0);
  std::vector<uint32_t> perm(static_cast<size_t>(nodes));
  std::iota(perm.begin(), perm.end(), 0u);
  std::mt19937 rng(seed);
  std::shuffle(perm.begin(), perm.end(), rng);
  for (uint64_t i = 0; i < nodes; ++i) {
    const uint32_t cur = perm[static_cast<size_t>(i)] * static_cast<uint32_t>(min_node);
    const uint32_t next = perm[static_cast<size_t>((i + 1) % nodes)] * static_cast<uint32_t>(min_node);
    auto* p = reinterpret_cast<uint32_t*>(host.data() + cur);
    p[0] = next;
    if (width >= 2) p[1] = 0x9e3779b9u ^ cur;
    if (width >= 4) { p[2] = 0x7f4a7c15u + cur; p[3] = 0x94d049bbu ^ next; }
  }
  *start_out = perm[0] * static_cast<uint32_t>(min_node);
}

static TickResult copy_tick_result(TickResult* d) {
  TickResult h{};
  HIP_CHECK(hipMemcpy(&h, d, sizeof(h), hipMemcpyDeviceToHost));
  return h;
}

static void gpu_warmup(int cu_count) {
  uint64_t* d = nullptr;
  HIP_CHECK(hipMalloc(&d, static_cast<size_t>(cu_count) * sizeof(uint64_t)));
  for (int i = 0; i < 3; ++i) {
    hipLaunchKernelGGL(warmup_kernel, dim3(cu_count), dim3(256), 0, 0, 500000ull, d);
    HIP_CHECK(hipGetLastError());
  }
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipFree(d));
}

static void run_timer_calibration(int repeats) {
  TickResult* d = nullptr;
  HIP_CHECK(hipMalloc(&d, sizeof(TickResult)));
  hipEvent_t start{}, stop{};
  HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
  const uint64_t iters = 100000000ull;
  bw::emit("suite,iters,repeat,ticks,event_ms,tick_rate_MHz,result\n");
  for (int r = -2; r < repeats; ++r) {
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(timer_calibration_kernel, dim3(1), dim3(64), 0, 0, iters, d);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
    float ms=0; HIP_CHECK(hipEventElapsedTime(&ms,start,stop));
    TickResult got=copy_tick_result(d);
    if (r >= 0) bw::emit("timer_calibration,%llu,%d,%llu,%.9f,%.6f,%u\n",
      static_cast<unsigned long long>(iters),r,static_cast<unsigned long long>(got.ticks),ms,
      static_cast<double>(got.ticks)/(static_cast<double>(ms)*1000.0),got.value);
  }
  HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop)); HIP_CHECK(hipFree(d));
}

static uint64_t choose_steps(uint64_t working_set, uint32_t stride) {
  const uint64_t nodes = std::max<uint64_t>(2, working_set / stride);
  return std::max<uint64_t>(4096, nodes);
}

static void run_latency_sizes(const std::vector<uint64_t>& sizes, int width, Policy policy,
                              int repeats, uint32_t stride, const char* suite_name,
                              int timed_pass_multiplier = 1) {
  bw::emit("suite,instruction,policy,working_set_bytes,actual_bytes,stride_bytes,steps,repeat,raw_ticks,baseline_ticks,adjusted_ticks,raw_cycles_per_load,adjusted_cycles_per_load,result\n");
  std::fflush(stdout);
  uint8_t* d_data = nullptr;
  TickResult* d_result = nullptr;
  HIP_CHECK(hipMalloc(&d_result, sizeof(TickResult)));
  size_t allocated = 0;
  std::vector<uint8_t> host;
  for (uint64_t requested : sizes) {
    uint32_t start = 0;
    build_pointer_cycle(host, requested, stride, width,
                        0x93600000u ^ static_cast<uint32_t>(requested) ^ static_cast<uint32_t>(width), &start);
    if (host.size() > allocated) {
      if (d_data) HIP_CHECK(hipFree(d_data));
      HIP_CHECK(hipMalloc(&d_data, host.size()));
      allocated = host.size();
    }
    HIP_CHECK(hipMemcpy(d_data, host.data(), host.size(), hipMemcpyHostToDevice));
    const uint64_t warm_steps = std::max<uint64_t>(2, host.size() / stride);
    const uint64_t steps = choose_steps(host.size(), stride) * static_cast<uint64_t>(timed_pass_multiplier);
    for (int r = 0; r < repeats; ++r) {
      hipLaunchKernelGGL(pointer_baseline, dim3(1), dim3(64), 0, 0,
                         start, stride, static_cast<uint32_t>(host.size()), steps, d_result);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      const TickResult base = copy_tick_result(d_result);
      launch_chase(width, d_data, start, warm_steps, steps, policy, d_result);
      HIP_CHECK(hipDeviceSynchronize());
      const TickResult got = copy_tick_result(d_result);
      const uint64_t adjusted = got.ticks > base.ticks ? got.ticks - base.ticks : 0;
      const char* inst = width == 1 ? "global_load_dword" : (width == 2 ? "global_load_dwordx2" : "global_load_dwordx4");
      bw::emit("%s,%s,%s,%llu,%zu,%u,%llu,%d,%llu,%llu,%llu,%.9f,%.9f,%u\n",
                  suite_name, inst, policy_name(policy),
                  static_cast<unsigned long long>(requested), host.size(), stride,
                  static_cast<unsigned long long>(steps), r,
                  static_cast<unsigned long long>(got.ticks),
                  static_cast<unsigned long long>(base.ticks),
                  static_cast<unsigned long long>(adjusted),
                  static_cast<double>(got.ticks) / static_cast<double>(steps),
                  static_cast<double>(adjusted) / static_cast<double>(steps), got.value);
      std::fflush(stdout);
    }
  }
  if (d_data) HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipFree(d_result));
}

// Independent VMEM issue kernels. One wave participates; each instruction is coalesced over Wave64.
#define LOAD1_ASYNC(dst, ptr) do { \
  const uint64_t _a = reinterpret_cast<uint64_t>(ptr); \
  asm volatile("global_load_dword %0, %1, off" : "=v"(dst) : "v"(_a) : "memory"); \
} while (0)
#define LOAD2_ASYNC(dst, ptr) do { \
  const uint64_t _a = reinterpret_cast<uint64_t>(ptr); \
  asm volatile("global_load_dwordx2 %0, %1, off" : "=v"(dst) : "v"(_a) : "memory"); \
} while (0)
#define LOAD4_ASYNC(dst, ptr) do { \
  const uint64_t _a = reinterpret_cast<uint64_t>(ptr); \
  asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(dst) : "v"(_a) : "memory"); \
} while (0)

#define DO_N(N, STMT) do { \
  if ((N) > 0) { STMT(0); } if ((N) > 1) { STMT(1); } if ((N) > 2) { STMT(2); } if ((N) > 3) { STMT(3); } \
  if ((N) > 4) { STMT(4); } if ((N) > 5) { STMT(5); } if ((N) > 6) { STMT(6); } if ((N) > 7) { STMT(7); } \
  if ((N) > 8) { STMT(8); } if ((N) > 9) { STMT(9); } if ((N) > 10) { STMT(10); } if ((N) > 11) { STMT(11); } \
  if ((N) > 12) { STMT(12); } if ((N) > 13) { STMT(13); } if ((N) > 14) { STMT(14); } if ((N) > 15) { STMT(15); } \
} while (0)

extern "C" __global__ void issue_load_clock_baseline_kernel(int rounds, TickResult* out) {
  if (blockIdx.x != 0 || threadIdx.x >= 64) return;
  uint64_t issue_total = 0;
  uint64_t completion_total = 0;
#pragma clang loop unroll(disable)
  for (int it = 0; it < rounds; ++it) {
    const uint64_t begin = clock64();
    asm volatile("" ::: "memory");
    const uint64_t issued = clock64();
    asm volatile("" ::: "memory");
    const uint64_t complete = clock64();
    issue_total += issued - begin;
    completion_total += complete - begin;
  }
  if (threadIdx.x == 0) out[0] = TickResult{issue_total, completion_total, 0, 0};
  return;
}

template <int N> __global__ void issue_load_dword_kernel(const uint32_t* in, int rounds, TickResult* out) {
  if (blockIdx.x != 0 || threadIdx.x >= 64) return;
  const int lane = threadIdx.x;
  uint32_t r[16] = {};
  uint32_t sink = 0;
  // Warm the exact lines.
#define WARM1(j) sink ^= in[lane + (j) * 64]
  DO_N(N, WARM1);
#undef WARM1
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  uint64_t issue_total = 0, completion_total = 0;
#pragma clang loop unroll(disable)
  for (int it = 0; it < rounds; ++it) {
    const uint64_t begin = clock64();
#define ONE1(j) LOAD1_ASYNC(r[j], in + lane + (j) * 64)
    DO_N(N, ONE1);
#undef ONE1
    const uint64_t issued = clock64();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    const uint64_t complete = clock64();
    issue_total += issued - begin;
    completion_total += complete - begin;
#define RED1(j) sink ^= r[j] + static_cast<uint32_t>(it)
    DO_N(N, RED1);
#undef RED1
  }
  if (lane == 0) out[0] = TickResult{issue_total, completion_total, sink, 0};
  return;
}

template <int N> __global__ void issue_load_dwordx2_kernel(const u32x2* in, int rounds, TickResult* out) {
  if (blockIdx.x != 0 || threadIdx.x >= 64) return;
  const int lane = threadIdx.x;
  u32x2 r[16] = {};
  uint32_t sink = 0;
#define WARM2(j) do { u32x2 _v = in[lane + (j) * 64]; sink ^= _v[0] ^ _v[1]; } while (0)
  DO_N(N, WARM2);
#undef WARM2
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  uint64_t issue_total = 0, completion_total = 0;
#pragma clang loop unroll(disable)
  for (int it = 0; it < rounds; ++it) {
    const uint64_t begin = clock64();
#define ONE2(j) LOAD2_ASYNC(r[j], in + lane + (j) * 64)
    DO_N(N, ONE2);
#undef ONE2
    const uint64_t issued = clock64();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    const uint64_t complete = clock64();
    issue_total += issued - begin;
    completion_total += complete - begin;
#define RED2(j) sink ^= r[j][0] + r[j][1] + static_cast<uint32_t>(it)
    DO_N(N, RED2);
#undef RED2
  }
  if (lane == 0) out[0] = TickResult{issue_total, completion_total, sink, 0};
  return;
}

template <int N> __global__ void issue_load_dwordx4_kernel(const u32x4* in, int rounds, TickResult* out) {
  if (blockIdx.x != 0 || threadIdx.x >= 64) return;
  const int lane = threadIdx.x;
  u32x4 r[16] = {};
  uint32_t sink = 0;
#define WARM4(j) do { u32x4 _v = in[lane + (j) * 64]; sink ^= _v[0] ^ _v[1] ^ _v[2] ^ _v[3]; } while (0)
  DO_N(N, WARM4);
#undef WARM4
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  uint64_t issue_total = 0, completion_total = 0;
#pragma clang loop unroll(disable)
  for (int it = 0; it < rounds; ++it) {
    const uint64_t begin = clock64();
#define ONE4(j) LOAD4_ASYNC(r[j], in + lane + (j) * 64)
    DO_N(N, ONE4);
#undef ONE4
    const uint64_t issued = clock64();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    const uint64_t complete = clock64();
    issue_total += issued - begin;
    completion_total += complete - begin;
#define RED4(j) sink ^= r[j][0] + r[j][1] + r[j][2] + r[j][3] + static_cast<uint32_t>(it)
    DO_N(N, RED4);
#undef RED4
  }
  if (lane == 0) out[0] = TickResult{issue_total, completion_total, sink, 0};
  return;
}

static void run_load_issue(int repeats) {
  constexpr size_t bytes = 64 * 16 * 16;
  void* d_data = nullptr;
  TickResult* d_result = nullptr;
  HIP_CHECK(hipMalloc(&d_data, bytes));
  HIP_CHECK(hipMemset(d_data, 0x5a, bytes));
  HIP_CHECK(hipMalloc(&d_result, sizeof(TickResult)));
  bw::emit("suite,instruction,independent_requests,rounds,repeat,issue_ticks,baseline_issue_ticks,adjusted_issue_ticks,completion_ticks,issue_cycles_per_instruction,adjusted_issue_cycles_per_instruction,completion_cycles_per_instruction,result\n");
  for (int width : {1, 2, 4}) {
    const char* inst = width == 1 ? "global_load_dword" : (width == 2 ? "global_load_dwordx2" : "global_load_dwordx4");
    for (int n : {1, 2, 4, 8, 16}) {
      const int rounds = 8192;
      // Two warmup launches.
      for (int w = 0; w < 2; ++w) {
        if (width == 1) {
#define LAUNCH_I1(NN) hipLaunchKernelGGL((issue_load_dword_kernel<NN>), dim3(1), dim3(64), 0, 0, static_cast<uint32_t*>(d_data), rounds, d_result)
          if (n == 1) LAUNCH_I1(1); else if (n == 2) LAUNCH_I1(2); else if (n == 4) LAUNCH_I1(4); else if (n == 8) LAUNCH_I1(8); else LAUNCH_I1(16);
#undef LAUNCH_I1
        } else if (width == 2) {
#define LAUNCH_I2(NN) hipLaunchKernelGGL((issue_load_dwordx2_kernel<NN>), dim3(1), dim3(64), 0, 0, static_cast<u32x2*>(d_data), rounds, d_result)
          if (n == 1) LAUNCH_I2(1); else if (n == 2) LAUNCH_I2(2); else if (n == 4) LAUNCH_I2(4); else if (n == 8) LAUNCH_I2(8); else LAUNCH_I2(16);
#undef LAUNCH_I2
        } else {
#define LAUNCH_I4(NN) hipLaunchKernelGGL((issue_load_dwordx4_kernel<NN>), dim3(1), dim3(64), 0, 0, static_cast<u32x4*>(d_data), rounds, d_result)
          if (n == 1) LAUNCH_I4(1); else if (n == 2) LAUNCH_I4(2); else if (n == 4) LAUNCH_I4(4); else if (n == 8) LAUNCH_I4(8); else LAUNCH_I4(16);
#undef LAUNCH_I4
        }
        HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
      }
      for (int r = 0; r < repeats; ++r) {
        hipLaunchKernelGGL(issue_load_clock_baseline_kernel, dim3(1), dim3(64), 0, 0, rounds, d_result);
        HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
        const TickResult baseline = copy_tick_result(d_result);
        if (width == 1) {
#define LAUNCH_I1(NN) hipLaunchKernelGGL((issue_load_dword_kernel<NN>), dim3(1), dim3(64), 0, 0, static_cast<uint32_t*>(d_data), rounds, d_result)
          if (n == 1) LAUNCH_I1(1); else if (n == 2) LAUNCH_I1(2); else if (n == 4) LAUNCH_I1(4); else if (n == 8) LAUNCH_I1(8); else LAUNCH_I1(16);
#undef LAUNCH_I1
        } else if (width == 2) {
#define LAUNCH_I2(NN) hipLaunchKernelGGL((issue_load_dwordx2_kernel<NN>), dim3(1), dim3(64), 0, 0, static_cast<u32x2*>(d_data), rounds, d_result)
          if (n == 1) LAUNCH_I2(1); else if (n == 2) LAUNCH_I2(2); else if (n == 4) LAUNCH_I2(4); else if (n == 8) LAUNCH_I2(8); else LAUNCH_I2(16);
#undef LAUNCH_I2
        } else {
#define LAUNCH_I4(NN) hipLaunchKernelGGL((issue_load_dwordx4_kernel<NN>), dim3(1), dim3(64), 0, 0, static_cast<u32x4*>(d_data), rounds, d_result)
          if (n == 1) LAUNCH_I4(1); else if (n == 2) LAUNCH_I4(2); else if (n == 4) LAUNCH_I4(4); else if (n == 8) LAUNCH_I4(8); else LAUNCH_I4(16);
#undef LAUNCH_I4
        }
        HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
        TickResult got = copy_tick_result(d_result);
        const uint64_t count = static_cast<uint64_t>(n) * rounds;
        const uint64_t adjusted = got.ticks > baseline.ticks ? got.ticks - baseline.ticks : 0;
        bw::emit("load_issue,%s,%d,%d,%d,%llu,%llu,%llu,%llu,%.9f,%.9f,%.9f,%u\n", inst, n, rounds, r,
                    static_cast<unsigned long long>(got.ticks), static_cast<unsigned long long>(baseline.ticks),
                    static_cast<unsigned long long>(adjusted), static_cast<unsigned long long>(got.aux_ticks),
                    static_cast<double>(got.ticks) / count, static_cast<double>(adjusted) / count,
                    static_cast<double>(got.aux_ticks) / count, got.value);
      }
    }
  }
  HIP_CHECK(hipFree(d_result));
  HIP_CHECK(hipFree(d_data));
}

#undef DO_N
#undef LOAD1_ASYNC
#undef LOAD2_ASYNC
#undef LOAD4_ASYNC

// Coalesced bandwidth kernels. Volatile vector types force the requested instruction width.
template <typename T>
__global__ void stream_load_kernel(const T* input, uint64_t elements, int passes, uint32_t* output) {
  const uint64_t gid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t threads = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const T* p = input;
  uint32_t sum = static_cast<uint32_t>(gid);
#pragma clang loop unroll(disable)
  for (int pass = 0; pass < passes; ++pass) {
#pragma clang loop unroll(disable)
    for (uint64_t i = gid; i < elements; i += threads) {
      T v = p[i];
      if constexpr (sizeof(T) == 4) sum ^= static_cast<uint32_t>(v);
      else if constexpr (sizeof(T) == 8) sum ^= v[0] + v[1];
      else sum ^= v[0] + v[1] + v[2] + v[3];
    }
  }
  output[gid] = sum;
  return;
}

template <typename T>
__global__ void hot_l1_load_kernel(const T* input, uint64_t elements_per_block,
                                   int iterations, uint32_t* output) {
  const uint64_t block_base = static_cast<uint64_t>(blockIdx.x) * elements_per_block;
  const T* p = input;
  uint32_t sum = threadIdx.x + blockIdx.x;
  uint64_t idx = threadIdx.x % elements_per_block;
#pragma clang loop unroll(disable)
  for (int it = 0; it < iterations; ++it) {
    T v = p[block_base + idx];
    if constexpr (sizeof(T) == 4) sum ^= static_cast<uint32_t>(v);
    else if constexpr (sizeof(T) == 8) sum ^= v[0] + v[1];
    else sum ^= v[0] + v[1] + v[2] + v[3];
    idx += blockDim.x;
    if (idx >= elements_per_block) idx -= elements_per_block;
  }
  const uint64_t gid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  output[gid] = sum;
  return;
}

template <typename T>
__global__ void hot_l2_glc_load_kernel(const T* input, uint64_t elements,
                                       int iterations, uint32_t* output) {
  const int lane_global = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t sum = static_cast<uint32_t>(lane_global);
  uint64_t idx = threadIdx.x % elements;
#pragma clang loop unroll(disable)
  for (int it = 0; it < iterations; ++it) {
    if constexpr (sizeof(T) == 4) {
      uint32_t v = global_load1_glc(reinterpret_cast<const uint32_t*>(input) + idx);
      sum ^= v;
    } else if constexpr (sizeof(T) == 8) {
      u32x2 v = global_load2_glc(reinterpret_cast<const u32x2*>(input) + idx);
      sum ^= v[0] + v[1];
    } else {
      u32x4 v = global_load4_glc(reinterpret_cast<const u32x4*>(input) + idx);
      sum ^= v[0] + v[1] + v[2] + v[3];
    }
    idx += blockDim.x;
    if (idx >= elements) idx -= elements;
  }
  output[lane_global] = sum;
  return;
}

template <typename T>
__global__ void stream_store_kernel(T* output, uint64_t elements, int passes, uint32_t seed) {
  const uint64_t gid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t threads = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  T* p = output;
#pragma clang loop unroll(disable)
  for (int pass = 0; pass < passes; ++pass) {
#pragma clang loop unroll(disable)
    for (uint64_t i = gid; i < elements; i += threads) {
      const uint32_t x = seed + static_cast<uint32_t>(i) + static_cast<uint32_t>(pass);
      if constexpr (sizeof(T) == 4) p[i] = x;
      else if constexpr (sizeof(T) == 8) { u32x2 v = {x, x ^ 0xa5a5a5a5u}; p[i] = v; }
      else { u32x4 v = {x, x + 1, x + 2, x + 3}; p[i] = v; }
    }
  }
  return;
}

struct EventPair {
  hipEvent_t start{}, stop{};
  EventPair() { HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop)); }
  ~EventPair() { hipEventDestroy(start); hipEventDestroy(stop); }
  template <typename F> float time(F&& launch) {
    HIP_CHECK(hipEventRecord(start));
    launch();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

template <typename T, typename Launch>
static int calibrate_repetition(int initial, double target_ms, Launch&& launch_with_repetition) {
  int reps = initial;
  EventPair ev;
  for (int k = 0; k < 6; ++k) {
    float ms = ev.time([&] { launch_with_repetition(reps); });
    HIP_CHECK(hipGetLastError());
    if (ms >= target_ms * 0.75 || reps > (1 << 27)) break;
    const double scale = std::min(16.0, std::max(2.0, target_ms / std::max<double>(ms, 0.001)));
    reps = static_cast<int>(std::min<double>(static_cast<double>(1 << 28), std::ceil(reps * scale)));
  }
  return reps;
}

template <typename T>
static void run_load_bandwidth_width(const char* level, uint64_t bytes, int cu_count,
                                     int repeats, int width) {
  const int grid = cu_count * 4;
  const uint64_t element_bytes = sizeof(T);
  uint64_t elements = std::max<uint64_t>(static_cast<uint64_t>(grid) * kBlock,
                                         bytes / element_bytes);
  // Make the stream divisible by the launch size so requested-byte accounting is exact.
  const uint64_t quantum = static_cast<uint64_t>(grid) * kBlock;
  elements = ((elements + quantum - 1) / quantum) * quantum;
  uint64_t actual_bytes = elements * element_bytes;
  T* d_input = nullptr;
  uint32_t* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_input, actual_bytes));
  HIP_CHECK(hipMemset(d_input, 0x3c, actual_bytes));
  HIP_CHECK(hipMalloc(&d_output, quantum * sizeof(uint32_t)));

  int iterations = 1;
  bool hot = !std::strcmp(level, "L1");
  uint64_t bytes_per_iteration = 0;
  uint64_t elements_per_block = 0;
  if (hot) {
    const uint64_t hot_bytes_per_block = 4096;
    elements_per_block = std::max<uint64_t>(kBlock, hot_bytes_per_block / element_bytes);
    const uint64_t needed = static_cast<uint64_t>(grid) * elements_per_block;
    if (needed > elements) {
      HIP_CHECK(hipFree(d_input));
      elements = needed;
      actual_bytes = elements * element_bytes;
      HIP_CHECK(hipMalloc(&d_input, actual_bytes));
      HIP_CHECK(hipMemset(d_input, 0x3c, elements * element_bytes));
    }
    bytes_per_iteration = static_cast<uint64_t>(grid) * kBlock * element_bytes;
    auto launch = [&](int it) {
      hipLaunchKernelGGL((hot_l1_load_kernel<T>), dim3(grid), dim3(kBlock), 0, 0,
                         d_input, elements_per_block, it, d_output);
    };
    launch(128); HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    iterations = calibrate_repetition<T>(128, 20.0, launch);
  } else {
    bytes_per_iteration = elements * element_bytes;
    auto launch = [&](int pass) {
      hipLaunchKernelGGL((stream_load_kernel<T>), dim3(grid), dim3(kBlock), 0, 0,
                         d_input, elements, pass, d_output);
    };
    launch(2); HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    iterations = calibrate_repetition<T>(2, 20.0, launch);
  }

  const char* inst = width == 1 ? "global_load_dword" : (width == 2 ? "global_load_dwordx2" : "global_load_dwordx4");
  EventPair ev;
  for (int r = 0; r < repeats; ++r) {
    float ms = 0.0f;
    if (hot) {
      ms = ev.time([&] {
        hipLaunchKernelGGL((hot_l1_load_kernel<T>), dim3(grid), dim3(kBlock), 0, 0,
                           d_input, elements_per_block, iterations, d_output);
      });
    } else {
      ms = ev.time([&] {
        hipLaunchKernelGGL((stream_load_kernel<T>), dim3(grid), dim3(kBlock), 0, 0,
                           d_input, elements, iterations, d_output);
      });
    }
    HIP_CHECK(hipGetLastError());
    const uint64_t moved = bytes_per_iteration * static_cast<uint64_t>(iterations);
    const double gbps = static_cast<double>(moved) / (static_cast<double>(ms) * 1.0e6);
    bw::emit("load_bandwidth,%s,%s,%llu,%llu,%llu,%d,%d,%d,%.9f,%llu,%.9f\n",
                level, inst, static_cast<unsigned long long>(bytes),
                static_cast<unsigned long long>(actual_bytes),
                static_cast<unsigned long long>(hot ? elements_per_block * element_bytes : 0),
                grid, iterations, r, ms, static_cast<unsigned long long>(moved), gbps);
    std::fflush(stdout);
  }
  HIP_CHECK(hipFree(d_output));
  HIP_CHECK(hipFree(d_input));
}

template <typename T>
static void run_l2_glc_bandwidth_width(uint64_t bytes, int cu_count, int repeats, int width) {
  const int grid = cu_count * 4;
  uint64_t elements = std::max<uint64_t>(kBlock, bytes / sizeof(T));
  const uint64_t actual_bytes = elements * sizeof(T);
  T* d_input = nullptr; uint32_t* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_input, actual_bytes)); HIP_CHECK(hipMemset(d_input, 0x3c, actual_bytes));
  HIP_CHECK(hipMalloc(&d_output, static_cast<size_t>(grid) * kBlock * sizeof(uint32_t)));
  auto launch = [&](int it) {
    hipLaunchKernelGGL((hot_l2_glc_load_kernel<T>), dim3(grid), dim3(kBlock), 0, 0,
                       d_input, elements, it, d_output);
  };
  launch(128); HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
  const int iterations = calibrate_repetition<T>(128, 20.0, launch);
  const char* inst = width == 1 ? "global_load_dword" : (width == 2 ? "global_load_dwordx2" : "global_load_dwordx4");
  EventPair ev;
  for (int r = 0; r < repeats; ++r) {
    const float ms = ev.time([&] { launch(iterations); }); HIP_CHECK(hipGetLastError());
    const uint64_t moved = static_cast<uint64_t>(grid) * kBlock * sizeof(T) * static_cast<uint64_t>(iterations);
    const double gbps = static_cast<double>(moved) / (static_cast<double>(ms) * 1.0e6);
    bw::emit("load_bandwidth,L2,%s,%llu,%llu,%llu,%d,%d,%d,%.9f,%llu,%.9f\n", inst,
                static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(actual_bytes),
                static_cast<unsigned long long>(actual_bytes), grid, iterations, r, ms,
                static_cast<unsigned long long>(moved), gbps);
  }
  HIP_CHECK(hipFree(d_output)); HIP_CHECK(hipFree(d_input));
}

static void run_load_bandwidth(const char* level, uint64_t bytes, int cu_count, int repeats) {
  bw::emit("suite,level,instruction,requested_working_set_bytes,actual_working_set_bytes,hot_bytes_per_block,grid,iterations,repeat,event_ms,requested_bytes,GB_per_s\n");
  if (!std::strcmp(level, "L2")) {
    run_l2_glc_bandwidth_width<uint32_t>(bytes, cu_count, repeats, 1);
    run_l2_glc_bandwidth_width<u32x2>(bytes, cu_count, repeats, 2);
    run_l2_glc_bandwidth_width<u32x4>(bytes, cu_count, repeats, 4);
  } else {
    run_load_bandwidth_width<uint32_t>(level, bytes, cu_count, repeats, 1);
    run_load_bandwidth_width<u32x2>(level, bytes, cu_count, repeats, 2);
    run_load_bandwidth_width<u32x4>(level, bytes, cu_count, repeats, 4);
  }
}

template <typename T>
static void run_store_bandwidth_width(uint64_t bytes, int cu_count, int repeats, int width) {
  const int grid = cu_count * 8;
  const uint64_t quantum = static_cast<uint64_t>(grid) * kBlock;
  uint64_t elements = std::max<uint64_t>(quantum, bytes / sizeof(T));
  elements = ((elements + quantum - 1) / quantum) * quantum;
  const uint64_t actual_bytes = elements * sizeof(T);
  T* d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, actual_bytes));
  HIP_CHECK(hipMemset(d_output, 0, actual_bytes));
  auto launch = [&](int pass) {
    hipLaunchKernelGGL((stream_store_kernel<T>), dim3(grid), dim3(kBlock), 0, 0,
                       d_output, elements, pass, 0x93600000u);
  };
  launch(2); HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
  const int passes = calibrate_repetition<T>(2, 20.0, launch);
  const char* inst = width == 1 ? "global_store_dword" : (width == 2 ? "global_store_dwordx2" : "global_store_dwordx4");
  EventPair ev;
  for (int r = 0; r < repeats; ++r) {
    const float ms = ev.time([&] { launch(passes); });
    HIP_CHECK(hipGetLastError());
    const uint64_t moved = actual_bytes * static_cast<uint64_t>(passes);
    const double gbps = static_cast<double>(moved) / (static_cast<double>(ms) * 1.0e6);
    uint32_t check = 0;
    HIP_CHECK(hipMemcpy(&check, d_output, sizeof(check), hipMemcpyDeviceToHost));
    bw::emit("store_bandwidth,%s,%llu,%d,%d,%d,%.9f,%llu,%.9f,%u\n",
                inst, static_cast<unsigned long long>(actual_bytes), grid, passes, r, ms,
                static_cast<unsigned long long>(moved), gbps, check);
    std::fflush(stdout);
  }
  HIP_CHECK(hipFree(d_output));
}

// Store issue/completion: take a timestamp before and after the final VMEM wait.
#define STORE1_ASYNC(ptr, value) do { const uint64_t _a = reinterpret_cast<uint64_t>(ptr); asm volatile("global_store_dword %0, %1, off" :: "v"(_a), "v"(value) : "memory"); } while (0)
#define STORE2_ASYNC(ptr, value) do { const uint64_t _a = reinterpret_cast<uint64_t>(ptr); asm volatile("global_store_dwordx2 %0, %1, off" :: "v"(_a), "v"(value) : "memory"); } while (0)
#define STORE4_ASYNC(ptr, value) do { const uint64_t _a = reinterpret_cast<uint64_t>(ptr); asm volatile("global_store_dwordx4 %0, %1, off" :: "v"(_a), "v"(value) : "memory"); } while (0)

template <int N> __global__ void store_issue_dword_kernel(uint32_t* outp, int rounds, TickResult* out) {
  if (blockIdx.x != 0 || threadIdx.x >= 64) return;
  const int lane = threadIdx.x;
  const uint32_t value = 0x12340000u + lane;
  const uint64_t begin = clock64();
#pragma clang loop unroll(disable)
  for (int it = 0; it < rounds; ++it) {
    if (N > 0) STORE1_ASYNC(outp + lane + 0 * 64, value + 0);
    if (N > 1) STORE1_ASYNC(outp + lane + 1 * 64, value + 1);
    if (N > 2) STORE1_ASYNC(outp + lane + 2 * 64, value + 2);
    if (N > 3) STORE1_ASYNC(outp + lane + 3 * 64, value + 3);
    if (N > 4) STORE1_ASYNC(outp + lane + 4 * 64, value + 4);
    if (N > 5) STORE1_ASYNC(outp + lane + 5 * 64, value + 5);
    if (N > 6) STORE1_ASYNC(outp + lane + 6 * 64, value + 6);
    if (N > 7) STORE1_ASYNC(outp + lane + 7 * 64, value + 7);
    if (N > 8) STORE1_ASYNC(outp + lane + 8 * 64, value + 8);
    if (N > 9) STORE1_ASYNC(outp + lane + 9 * 64, value + 9);
    if (N > 10) STORE1_ASYNC(outp + lane + 10 * 64, value + 10);
    if (N > 11) STORE1_ASYNC(outp + lane + 11 * 64, value + 11);
    if (N > 12) STORE1_ASYNC(outp + lane + 12 * 64, value + 12);
    if (N > 13) STORE1_ASYNC(outp + lane + 13 * 64, value + 13);
    if (N > 14) STORE1_ASYNC(outp + lane + 14 * 64, value + 14);
    if (N > 15) STORE1_ASYNC(outp + lane + 15 * 64, value + 15);
  }
  const uint64_t issued = clock64();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  const uint64_t complete = clock64();
  if (lane == 0) out[0] = TickResult{issued - begin, complete - begin, value, 0};
  return;
}

template <int N> __global__ void store_issue_dwordx2_kernel(u32x2* outp, int rounds, TickResult* out) {
  if (blockIdx.x != 0 || threadIdx.x >= 64) return;
  const int lane = threadIdx.x;
  const u32x2 value = {0x12340000u + static_cast<uint32_t>(lane), 0x56780000u + static_cast<uint32_t>(lane)};
  const uint64_t begin = clock64();
#pragma clang loop unroll(disable)
  for (int it = 0; it < rounds; ++it) {
#define S2(j) if (N > (j)) STORE2_ASYNC(outp + lane + (j) * 64, value)
    S2(0); S2(1); S2(2); S2(3); S2(4); S2(5); S2(6); S2(7);
    S2(8); S2(9); S2(10); S2(11); S2(12); S2(13); S2(14); S2(15);
#undef S2
  }
  const uint64_t issued = clock64();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  const uint64_t complete = clock64();
  if (lane == 0) out[0] = TickResult{issued - begin, complete - begin, value[0], 0};
  return;
}

template <int N> __global__ void store_issue_dwordx4_kernel(u32x4* outp, int rounds, TickResult* out) {
  if (blockIdx.x != 0 || threadIdx.x >= 64) return;
  const int lane = threadIdx.x;
  const u32x4 value = {0x12340000u + static_cast<uint32_t>(lane), 2u, 3u, 4u};
  const uint64_t begin = clock64();
#pragma clang loop unroll(disable)
  for (int it = 0; it < rounds; ++it) {
#define S4(j) if (N > (j)) STORE4_ASYNC(outp + lane + (j) * 64, value)
    S4(0); S4(1); S4(2); S4(3); S4(4); S4(5); S4(6); S4(7);
    S4(8); S4(9); S4(10); S4(11); S4(12); S4(13); S4(14); S4(15);
#undef S4
  }
  const uint64_t issued = clock64();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  const uint64_t complete = clock64();
  if (lane == 0) out[0] = TickResult{issued - begin, complete - begin, value[0], 0};
  return;
}

static void run_store_issue(int repeats) {
  constexpr size_t bytes = 64 * 16 * 16;
  void* d_data = nullptr;
  TickResult* d_result = nullptr;
  HIP_CHECK(hipMalloc(&d_data, bytes));
  HIP_CHECK(hipMalloc(&d_result, sizeof(TickResult)));
  bw::emit("suite,instruction,independent_requests,rounds,repeat,issue_ticks,completion_ticks,issue_cycles_per_instruction,completion_cycles_per_instruction,result\n");
  for (int width : {1, 2, 4}) {
    const char* inst = width == 1 ? "global_store_dword" : (width == 2 ? "global_store_dwordx2" : "global_store_dwordx4");
    for (int n : {1, 2, 4, 8, 16}) {
      const int rounds = 8192;
      for (int r = -2; r < repeats; ++r) {
        if (width == 1) {
#define LAUNCH_S1(NN) hipLaunchKernelGGL((store_issue_dword_kernel<NN>), dim3(1), dim3(64), 0, 0, static_cast<uint32_t*>(d_data), rounds, d_result)
          if (n == 1) LAUNCH_S1(1); else if (n == 2) LAUNCH_S1(2); else if (n == 4) LAUNCH_S1(4); else if (n == 8) LAUNCH_S1(8); else LAUNCH_S1(16);
#undef LAUNCH_S1
        } else if (width == 2) {
#define LAUNCH_S2(NN) hipLaunchKernelGGL((store_issue_dwordx2_kernel<NN>), dim3(1), dim3(64), 0, 0, static_cast<u32x2*>(d_data), rounds, d_result)
          if (n == 1) LAUNCH_S2(1); else if (n == 2) LAUNCH_S2(2); else if (n == 4) LAUNCH_S2(4); else if (n == 8) LAUNCH_S2(8); else LAUNCH_S2(16);
#undef LAUNCH_S2
        } else {
#define LAUNCH_S4(NN) hipLaunchKernelGGL((store_issue_dwordx4_kernel<NN>), dim3(1), dim3(64), 0, 0, static_cast<u32x4*>(d_data), rounds, d_result)
          if (n == 1) LAUNCH_S4(1); else if (n == 2) LAUNCH_S4(2); else if (n == 4) LAUNCH_S4(4); else if (n == 8) LAUNCH_S4(8); else LAUNCH_S4(16);
#undef LAUNCH_S4
        }
        HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
        if (r < 0) continue;
        TickResult got = copy_tick_result(d_result);
        const uint64_t count = static_cast<uint64_t>(n) * rounds;
        bw::emit("store_issue,%s,%d,%d,%d,%llu,%llu,%.9f,%.9f,%u\n", inst, n, rounds, r,
                    static_cast<unsigned long long>(got.ticks), static_cast<unsigned long long>(got.aux_ticks),
                    static_cast<double>(got.ticks) / count, static_cast<double>(got.aux_ticks) / count, got.value);
      }
    }
  }
  HIP_CHECK(hipFree(d_result)); HIP_CHECK(hipFree(d_data));
}

static void run_store_bandwidth(uint64_t bytes, int cu_count, int repeats) {
  bw::emit("suite,instruction,working_set_bytes,grid,passes,repeat,event_ms,requested_bytes,GB_per_s,readback\n");
  run_store_bandwidth_width<uint32_t>(bytes, cu_count, repeats, 1);
  run_store_bandwidth_width<u32x2>(bytes, cu_count, repeats, 2);
  run_store_bandwidth_width<u32x4>(bytes, cu_count, repeats, 4);
}

static std::string get_arg(int argc, char** argv, const char* key, const std::string& def = "") {
  for (int i = 1; i + 1 < argc; ++i) if (!std::strcmp(argv[i], key)) return argv[i + 1];
  return def;
}
static bool has_arg(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], key)) return true;
  return false;
}

static Policy parse_policy(const std::string& s) {
  if (s == "default") return Policy::Default;
  if (s == "glc") return Policy::Glc;
  if (s == "slc") return Policy::Slc;
  if (s == "glc_slc" || s == "glc+slc") return Policy::GlcSlc;
  std::fprintf(stderr, "unknown policy %s\n", s.c_str()); std::exit(2);
}

static void print_device() {
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  hipDeviceProp_t p{};
  HIP_CHECK(hipGetDeviceProperties(&p, dev));
  std::fprintf(stderr,
               "DEVICE device=%d name=%s arch=%s cu=%d wave=%d clockRate_kHz=%d memoryClockRate_kHz=%d busWidth=%d totalGlobalMem=%zu\n",
               dev, p.name, p.gcnArchName, p.multiProcessorCount, p.warpSize, p.clockRate,
               p.memoryClockRate, p.memoryBusWidth, static_cast<size_t>(p.totalGlobalMem));
}

int main(int argc, char** argv) {
  bw::start();

  const std::string mode = get_arg(argc, argv, "--mode", "latency_one");
  const int repeats = std::atoi(get_arg(argc, argv, "--repeats", "30").c_str());
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  print_device();
  gpu_warmup(prop.multiProcessorCount);

  if (mode == "timer_calibration") {
    run_timer_calibration(repeats);
  } else if (mode == "cache_scan") {
    const uint32_t stride = static_cast<uint32_t>(parse_bytes(get_arg(argc, argv, "--stride", "64")));
    const Policy scan_policy = parse_policy(get_arg(argc, argv, "--policy", "default"));
    run_latency_sizes(default_scan_sizes(), 1, scan_policy, repeats, stride, "cache_scan");
  } else if (mode == "width_latency_scan") {
    for (int width : {1, 2, 4}) run_latency_sizes(default_width_sizes(), width, Policy::Default, repeats, 64, "width_latency_scan");
  } else if (mode == "policy") {
    std::vector<uint64_t> sizes;
    const std::string b = get_arg(argc, argv, "--bytes", "4K");
    sizes.push_back(parse_bytes(b));
    for (Policy p : {Policy::Default, Policy::Glc, Policy::Slc, Policy::GlcSlc})
      run_latency_sizes(sizes, 1, p, repeats, 64, "policy");
  } else if (mode == "latency_one" || mode == "profile_chase") {
    std::vector<uint64_t> sizes{parse_bytes(get_arg(argc, argv, "--bytes", "4K"))};
    int width = std::atoi(get_arg(argc, argv, "--width", "1").c_str());
    Policy p = parse_policy(get_arg(argc, argv, "--policy", "default"));
    uint32_t stride = static_cast<uint32_t>(parse_bytes(get_arg(argc, argv, "--stride", "64")));
    run_latency_sizes(sizes, width, p, mode == "profile_chase" ? 1 : repeats, stride, mode.c_str(),
                      mode == "profile_chase" ? 4 : 1);
  } else if (mode == "load_issue") {
    run_load_issue(repeats);
  } else if (mode == "load_bandwidth") {
    const std::string level = get_arg(argc, argv, "--level", "HBM");
    const uint64_t bytes = parse_bytes(get_arg(argc, argv, "--bytes", level == "L1" ? "4K" : (level == "L2" ? "4M" : "1G")));
    run_load_bandwidth(level.c_str(), bytes, prop.multiProcessorCount, repeats);
  } else if (mode == "store_issue") {
    run_store_issue(repeats);
  } else if (mode == "store_bandwidth") {
    const uint64_t bytes = parse_bytes(get_arg(argc, argv, "--bytes", "1G"));
    run_store_bandwidth(bytes, prop.multiProcessorCount, repeats);
  } else {
    std::fprintf(stderr,
      "Usage: %s --mode {timer_calibration|cache_scan|width_latency_scan|policy|latency_one|profile_chase|load_issue|load_bandwidth|store_issue|store_bandwidth} [options]\n",
      argv[0]);
    return mode == "help" ? 0 : 2;
  }
  HIP_CHECK(hipDeviceSynchronize());
  return bw::finish("global_memory");
}
