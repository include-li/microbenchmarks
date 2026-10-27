/* BW1000 single-file HIP program: buffer/issue
 * Compile: hipcc -O2 -std=c++17 --offload-arch=gfx936 issue.hip -o bench
 * Run: ./bench --mode timing --path buffer --op load --width 4 --repeats 30
 * Minimal run (NOT a timing reproduction): ./bench --mode timing --path buffer --op load --width 4 --repeats 1
 * No local headers, private libraries, scripts, data files or other source files required.
 * Only public HIP/DTK, its driver/runtime, and the C++ standard library are required.
 * Prints in-process summaries in timer tick and calibrated ns. BW1000_RAW=1 also prints samples.
 * Original measurement core: buffer/src/width_issue_bench.cpp; SHA256=a1829d0cbe73bf0c41b27b48c28644011aca3d1bcac269af5c251222977f1a3e
 * Host-only additions: calibration, sample grouping, original normalization/statistics, defaults.
 * Historical published measurements remain in README/results.csv; new runs do not replace them.
 */
// Self-contained host-side CSV observation and statistics. No subprocesses or private includes.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
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
    if(family=="lds_core"||family=="lds_extended"||family=="global_width"||family=="buffer_issue") {
      if(s=="baseline"){put("baseline_completion","bracket",num(r,"completion_ticks"));continue;}
      const bool core=family=="lds_core";const std::string field=core?"issue_or_dep_ticks":"issue_ticks",comp=core?"completion_or_baseline_ticks":"completion_ticks";
      if(s=="dependent")put("dependent_adjusted","dependent instruction",(num(r,field)-num(r,comp))/num(r,"n"));
      else if(s=="completion") {put("completion_raw","request",num(r,comp));if(std::isfinite(base))put("completion_adjusted","request",num(r,comp)-base);}
      else if(s=="issue") {
        double rounds=num(r,"rounds",1),n=num(r,"n");put("issue_batch","N-request batch/round",num(r,field)/rounds);put("completion_batch","N-request batch/round",num(r,comp)/rounds);put("drain_batch","N-request batch/round",(num(r,comp)-num(r,field))/rounds);put("issue_per_request","request at the stated N",num(r,field)/(rounds*n));put("total_per_request","request at the stated N, includes drain",num(r,comp)/(rounds*n));put("drain_per_request","request at the stated N",(num(r,comp)-num(r,field))/(rounds*n));
        batches[{t,conditions(r,{"n","rounds"}),"issue_fit"}][int(n)].push_back(num(r,field)/rounds);
      }else {put("observed_issue_bracket","observed bracket, NOT native latency",num(r,"issue_ticks"));put("observed_completion_bracket","observed bracket, NOT native latency",num(r,"completion_ticks"));}
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
using u32x3 = uint32_t __attribute__((ext_vector_type(3)));
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));

#define HIP_CHECK(call) do { \
  hipError_t e_ = (call); \
  if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s:%d: %s -> %s\n", __FILE__, __LINE__, #call, hipGetErrorString(e_)); \
    std::exit(2); \
  } \
} while (0)

struct TickResult { uint64_t issue_ticks, completion_ticks; uint32_t value, pad; };

template<int B> struct GType;
template<> struct GType<1> { using type = uint32_t; };
template<> struct GType<2> { using type = uint32_t; };
template<> struct GType<4> { using type = uint32_t; };
template<> struct GType<8> { using type = u32x2; };
template<> struct GType<12> { using type = u32x3; };
template<> struct GType<16> { using type = u32x4; };
template<int B> using GV = typename GType<B>::type;

__device__ __forceinline__ u32x4 make_rsrc(const void* base, uint32_t stride = 0) {
  const uint64_t p = reinterpret_cast<uint64_t>(base);
  return u32x4{static_cast<uint32_t>(p),
               (static_cast<uint32_t>(p >> 32) & 0xffffu) | (stride << 16),
               0x7ffffffeu, 0x00027000u};
}

template<int B>
__device__ __forceinline__ GV<B> make_value(uint32_t x) {
  if constexpr (B <= 4) return x;
  else if constexpr (B == 8) return GV<B>{x, x + 1u};
  else if constexpr (B == 12) return GV<B>{x, x + 1u, x + 2u};
  else return GV<B>{x, x + 1u, x + 2u, x + 3u};
}

template<int B>
__device__ __forceinline__ uint32_t sum_value(GV<B> v) {
  if constexpr (B <= 4) return v;
  else {
    uint32_t s = 0;
#pragma unroll
    for (int i = 0; i < B / 4; ++i) s += v[i];
    return s;
  }
}

template<int B, bool Glc> __device__ __forceinline__ GV<B> global_load_async(const uint8_t* p);
#define DEFINE_LOAD(B, TYPE, OPCODE) \
template<> __device__ __forceinline__ TYPE global_load_async<B, false>(const uint8_t* p) { \
  TYPE v; const uint64_t a = reinterpret_cast<uint64_t>(p); \
  asm volatile(OPCODE " %0, %1, off" : "=v"(v) : "v"(a) : "memory"); return v; \
} \
template<> __device__ __forceinline__ TYPE global_load_async<B, true>(const uint8_t* p) { \
  TYPE v; const uint64_t a = reinterpret_cast<uint64_t>(p); \
  asm volatile(OPCODE " %0, %1, off glc" : "=v"(v) : "v"(a) : "memory"); return v; \
}
DEFINE_LOAD(1, uint32_t, "global_load_ubyte")
DEFINE_LOAD(2, uint32_t, "global_load_ushort")
DEFINE_LOAD(4, uint32_t, "global_load_dword")
DEFINE_LOAD(8, u32x2, "global_load_dwordx2")
DEFINE_LOAD(12, u32x3, "global_load_dwordx3")
DEFINE_LOAD(16, u32x4, "global_load_dwordx4")
#undef DEFINE_LOAD

template<int B, bool Glc> __device__ __forceinline__ GV<B> buffer_load_async(u32x4 rsrc, uint32_t offset);
#define DEFINE_BUFFER_LOAD(B, TYPE, OPCODE) \
template<> __device__ __forceinline__ TYPE buffer_load_async<B, false>(u32x4 rsrc, uint32_t offset) { \
  TYPE v; asm volatile(OPCODE " %0, %1, %2, 0 offen" : "=v"(v) : "v"(offset), "s"(rsrc) : "memory"); return v; \
} \
template<> __device__ __forceinline__ TYPE buffer_load_async<B, true>(u32x4 rsrc, uint32_t offset) { \
  TYPE v; asm volatile(OPCODE " %0, %1, %2, 0 offen glc" : "=v"(v) : "v"(offset), "s"(rsrc) : "memory"); return v; \
}
DEFINE_BUFFER_LOAD(1, uint32_t, "buffer_load_ubyte")
DEFINE_BUFFER_LOAD(2, uint32_t, "buffer_load_ushort")
DEFINE_BUFFER_LOAD(4, uint32_t, "buffer_load_dword")
DEFINE_BUFFER_LOAD(8, u32x2, "buffer_load_dwordx2")
DEFINE_BUFFER_LOAD(12, u32x3, "buffer_load_dwordx3")
DEFINE_BUFFER_LOAD(16, u32x4, "buffer_load_dwordx4")
#undef DEFINE_BUFFER_LOAD

template<int B> __device__ __forceinline__ void global_store_async(uint8_t* p, GV<B> v);
#define DEFINE_STORE(B, TYPE, OPCODE) \
template<> __device__ __forceinline__ void global_store_async<B>(uint8_t* p, TYPE v) { \
  const uint64_t a = reinterpret_cast<uint64_t>(p); \
  asm volatile(OPCODE " %0, %1, off" :: "v"(a), "v"(v) : "memory"); \
}
DEFINE_STORE(1, uint32_t, "global_store_byte")
DEFINE_STORE(2, uint32_t, "global_store_short")
DEFINE_STORE(4, uint32_t, "global_store_dword")
DEFINE_STORE(8, u32x2, "global_store_dwordx2")
DEFINE_STORE(12, u32x3, "global_store_dwordx3")
DEFINE_STORE(16, u32x4, "global_store_dwordx4")
#undef DEFINE_STORE

template<int B> __device__ __forceinline__ void buffer_store_async(u32x4 rsrc, uint32_t offset, GV<B> v);
#define DEFINE_BUFFER_STORE(B, TYPE, OPCODE) \
template<> __device__ __forceinline__ void buffer_store_async<B>(u32x4 rsrc, uint32_t offset, TYPE v) { \
  asm volatile(OPCODE " %0, %1, %2, 0 offen" :: "v"(v), "v"(offset), "s"(rsrc) : "memory"); \
}
DEFINE_BUFFER_STORE(1, uint32_t, "buffer_store_byte")
DEFINE_BUFFER_STORE(2, uint32_t, "buffer_store_short")
DEFINE_BUFFER_STORE(4, uint32_t, "buffer_store_dword")
DEFINE_BUFFER_STORE(8, u32x2, "buffer_store_dwordx2")
DEFINE_BUFFER_STORE(12, u32x3, "buffer_store_dwordx3")
DEFINE_BUFFER_STORE(16, u32x4, "buffer_store_dwordx4")
#undef DEFINE_BUFFER_STORE

__device__ __forceinline__ void wait_vmem() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }

extern "C" __global__ void vmem_wait_baseline_kernel(TickResult* out) {
  __shared__ uint32_t control[64];
  const uint32_t lane = threadIdx.x & 63u;
  control[lane] = lane;
  __syncthreads();
  const uint64_t t0 = clock64();
  wait_vmem();
  const uint64_t t1 = clock64();
  if (lane == 0) out[0] = TickResult{t1 - t0, t1 - t0, control[0], 0};
}

template<int B, bool Glc, bool Store, bool Buffer>
__global__ void completion_kernel(uint8_t* data, TickResult* out) {
  __shared__ uint32_t control[64];
  const uint32_t lane = threadIdx.x & 63u;
  control[lane] = lane;
  __syncthreads();
  GV<B> v = make_value<B>(lane + 1u);
  uint8_t* p = data + static_cast<uint64_t>(lane) * B;
  const u32x4 rsrc = make_rsrc(data);
  const uint32_t offset = lane * B;
  const uint64_t t0 = clock64();
  if constexpr (Store && Buffer) buffer_store_async<B>(rsrc, offset, v);
  else if constexpr (Store) global_store_async<B>(p, v);
  else if constexpr (Buffer) v = buffer_load_async<B, Glc>(rsrc, offset);
  else v = global_load_async<B, Glc>(p);
  wait_vmem();
  const uint64_t t1 = clock64();
  if (lane == 0) out[0] = TickResult{t1 - t0, t1 - t0, sum_value<B>(v), 0};
}

template<int B, bool Glc, bool Store, bool Buffer, int N>
__global__ void issue_kernel(uint8_t* data, int rounds, TickResult* out) {
  __shared__ uint32_t control[64];
  const uint32_t lane = threadIdx.x & 63u;
  control[lane] = lane;
  __syncthreads();
  GV<B> vals[N];
  const u32x4 rsrc = make_rsrc(data);
#pragma unroll
  for (int i = 0; i < N; ++i) vals[i] = make_value<B>(lane + i + 1u);
  uint64_t issue_total = 0, completion_total = 0;
  uint32_t sink = 0;
#pragma clang loop unroll(disable)
  for (int round = 0; round < rounds; ++round) {
  const uint64_t t0 = clock64();
#pragma unroll
  for (int i = 0; i < N; ++i) {
    uint8_t* p = data + (static_cast<uint64_t>(i) * 64u + lane) * B;
    const uint32_t offset = (i * 64u + lane) * B;
    if constexpr (Store && Buffer) buffer_store_async<B>(rsrc, offset, vals[i]);
    else if constexpr (Store) global_store_async<B>(p, vals[i]);
    else if constexpr (Buffer) vals[i] = buffer_load_async<B, Glc>(rsrc, offset);
    else vals[i] = global_load_async<B, Glc>(p);
  }
  const uint64_t ti = clock64();
  wait_vmem();
  const uint64_t td = clock64();
  issue_total += ti - t0;
  completion_total += td - t0;
#pragma unroll
  for (int i = 0; i < N; ++i) sink += sum_value<B>(vals[i]) + static_cast<uint32_t>(round);
  }
  if (lane == 0) out[0] = TickResult{issue_total, completion_total, sink, 0};
}

template<int B, bool Store, bool Buffer>
__global__ void stream_kernel(uint8_t* data, uint64_t bytes, uint32_t passes, uint32_t* sink_out) {
  const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t threads = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const uint64_t elements = bytes / B;
  const u32x4 rsrc = make_rsrc(data);
  uint32_t sink = 0;
  for (uint32_t pass = 0; pass < passes; ++pass) {
    for (uint64_t i = tid; i < elements; i += threads * 4u) {
      if constexpr (Store) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const uint64_t j = i + static_cast<uint64_t>(k) * threads;
          if (j < elements) {
            const GV<B> value = make_value<B>(static_cast<uint32_t>(j) + pass + 1u);
            if constexpr (Buffer) buffer_store_async<B>(rsrc, static_cast<uint32_t>(j * B), value);
            else global_store_async<B>(data + j * B, value);
          }
        }
      } else {
        GV<B> v0 = make_value<B>(0), v1 = v0, v2 = v0, v3 = v0;
        if (i < elements) {
          if constexpr (Buffer) v0 = buffer_load_async<B, false>(rsrc, static_cast<uint32_t>(i * B));
          else v0 = global_load_async<B, false>(data + i * B);
        }
        if (i + threads < elements) {
          if constexpr (Buffer) v1 = buffer_load_async<B, false>(rsrc, static_cast<uint32_t>((i + threads) * B));
          else v1 = global_load_async<B, false>(data + (i + threads) * B);
        }
        if (i + 2u * threads < elements) {
          if constexpr (Buffer) v2 = buffer_load_async<B, false>(rsrc, static_cast<uint32_t>((i + 2u * threads) * B));
          else v2 = global_load_async<B, false>(data + (i + 2u * threads) * B);
        }
        if (i + 3u * threads < elements) {
          if constexpr (Buffer) v3 = buffer_load_async<B, false>(rsrc, static_cast<uint32_t>((i + 3u * threads) * B));
          else v3 = global_load_async<B, false>(data + (i + 3u * threads) * B);
        }
        wait_vmem();
        sink ^= sum_value<B>(v0) + sum_value<B>(v1) + sum_value<B>(v2) + sum_value<B>(v3);
      }
    }
  }
  wait_vmem();
  if (tid == 0) sink_out[0] = sink;
}

extern "C" __global__ void prime_kernel(const uint8_t* p, uint64_t bytes, uint32_t* sink_out) {
  const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t threads = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  uint32_t sink = 0;
  for (uint64_t off = tid * 64u; off < bytes; off += threads * 64u) {
    uint32_t v;
    const uint64_t a = reinterpret_cast<uint64_t>(p + off);
    asm volatile("global_load_dword %0, %1, off glc" : "=v"(v) : "v"(a) : "memory");
    sink ^= v;
  }
  wait_vmem();
  if (tid == 0) sink_out[0] = sink;
}

static std::string arg(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i) if (!std::strcmp(argv[i], key)) return argv[i + 1];
  return def;
}
static int argi(int argc, char** argv, const char* key, int def) {
  const std::string d = std::to_string(def); return std::atoi(arg(argc, argv, key, d.c_str()).c_str());
}

static void prepare(const std::string& state, uint8_t* data, uint8_t* pollution, uint32_t* sink) {
  if (state == "L2-hot") hipLaunchKernelGGL(prime_kernel, dim3(80), dim3(256), 0, 0, data, 1ull << 20, sink);
  else hipLaunchKernelGGL(prime_kernel, dim3(320), dim3(256), 0, 0, pollution, 32ull << 20, sink);
  HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
}

template<int B, bool Store, bool Buffer>
static void launch_completion(bool glc, uint8_t* p, TickResult* out) {
  if (glc) hipLaunchKernelGGL((completion_kernel<B, true, Store, Buffer>), dim3(1), dim3(64), 0, 0, p, out);
  else hipLaunchKernelGGL((completion_kernel<B, false, Store, Buffer>), dim3(1), dim3(64), 0, 0, p, out);
}

template<int B, bool Store, bool Buffer, int N>
static void launch_issue_n(bool glc, uint8_t* p, int rounds, TickResult* out) {
  if (glc) hipLaunchKernelGGL((issue_kernel<B, true, Store, Buffer, N>), dim3(1), dim3(64), 0, 0, p, rounds, out);
  else hipLaunchKernelGGL((issue_kernel<B, false, Store, Buffer, N>), dim3(1), dim3(64), 0, 0, p, rounds, out);
}

template<int B, bool Store, bool Buffer>
static void launch_issue(int n, bool glc, uint8_t* p, int rounds, TickResult* out) {
  if (n == 1) launch_issue_n<B, Store, Buffer, 1>(glc, p, rounds, out);
  else if (n == 2) launch_issue_n<B, Store, Buffer, 2>(glc, p, rounds, out);
  else if (n == 4) launch_issue_n<B, Store, Buffer, 4>(glc, p, rounds, out);
  else if (n == 8) launch_issue_n<B, Store, Buffer, 8>(glc, p, rounds, out);
  else if (n == 16) launch_issue_n<B, Store, Buffer, 16>(glc, p, rounds, out);
  else launch_issue_n<B, Store, Buffer, 32>(glc, p, rounds, out);
}

template<int B, bool Buffer>
static void run_timing(const std::string& state, bool store, int repeats) {
  uint8_t *data = nullptr, *pollution = nullptr;
  uint32_t* sink = nullptr; TickResult* result = nullptr;
  HIP_CHECK(hipMalloc(&data, 64ull << 20)); HIP_CHECK(hipMalloc(&pollution, 32ull << 20));
  HIP_CHECK(hipMalloc(&sink, sizeof(uint32_t))); HIP_CHECK(hipMalloc(&result, sizeof(TickResult)));
  HIP_CHECK(hipMemset(data, 0x3c, 64ull << 20)); HIP_CHECK(hipMemset(pollution, 0x5a, 32ull << 20));
  const bool glc = state == "L2-hot" && !store;
  for (int r = 0; r < repeats; ++r) {
    prepare(state, data, pollution, sink);
    const uint64_t offset = state == "L2-hot" ? 0 : (static_cast<uint64_t>(r) * 4096u);
    if (store) launch_completion<B, true, Buffer>(false, data + offset, result);
    else launch_completion<B, false, Buffer>(glc, data + offset, result);
    HIP_CHECK(hipGetLastError());
    TickResult h{}; HIP_CHECK(hipMemcpy(&h, result, sizeof(h), hipMemcpyDeviceToHost));
    bw::emit("completion,%s,%s,%d,%s,%d,%d,%d,%llu,%llu,%u\n", Buffer ? "buffer" : "global", store ? "store" : "load", B,
                state.c_str(), 1, 1, r, static_cast<unsigned long long>(h.issue_ticks),
                static_cast<unsigned long long>(h.completion_ticks), h.value);
  }
  for (int n : {1, 2, 4, 8, 16, 32}) {
    const int rounds = 4096;
    for (int r = 0; r < repeats; ++r) {
      prepare(state, data, pollution, sink);
      if (store) launch_issue<B, true, Buffer>(n, false, data, rounds, result);
      else launch_issue<B, false, Buffer>(n, glc, data, rounds, result);
      HIP_CHECK(hipGetLastError());
      TickResult h{}; HIP_CHECK(hipMemcpy(&h, result, sizeof(h), hipMemcpyDeviceToHost));
      bw::emit("issue,%s,%s,%d,%s,%d,%d,%d,%llu,%llu,%u\n", Buffer ? "buffer" : "global", store ? "store" : "load", B,
                  state.c_str(), n, rounds, r, static_cast<unsigned long long>(h.issue_ticks),
                  static_cast<unsigned long long>(h.completion_ticks), h.value);
    }
  }
  HIP_CHECK(hipFree(result)); HIP_CHECK(hipFree(sink)); HIP_CHECK(hipFree(pollution)); HIP_CHECK(hipFree(data));
}

template<int B, bool Store, bool Buffer>
static void run_stream(uint64_t bytes, int repeats) {
  uint8_t* data = nullptr; uint32_t* sink = nullptr;
  HIP_CHECK(hipMalloc(&data, bytes)); HIP_CHECK(hipMalloc(&sink, sizeof(uint32_t)));
  HIP_CHECK(hipMemset(data, 0x3c, bytes));
  hipEvent_t a, b; HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b));
  const int blocks = 320, threads = 256; const uint32_t passes = 4;
  hipLaunchKernelGGL((stream_kernel<B, Store, Buffer>), dim3(blocks), dim3(threads), 0, 0, data, bytes, 1u, sink);
  HIP_CHECK(hipDeviceSynchronize());
  for (int r = 0; r < repeats; ++r) {
    HIP_CHECK(hipEventRecord(a));
    hipLaunchKernelGGL((stream_kernel<B, Store, Buffer>), dim3(blocks), dim3(threads), 0, 0, data, bytes, passes, sink);
    HIP_CHECK(hipEventRecord(b)); HIP_CHECK(hipEventSynchronize(b));
    float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    const uint64_t moved = bytes * passes;
    const double gbps = static_cast<double>(moved) / (static_cast<double>(ms) * 1.0e6);
    bw::emit("stream,%s,%s,%d,streaming,%u,%d,%.6f,%llu,%.6f\n", Buffer ? "buffer" : "global", Store ? "store" : "load", B,
                passes, r, ms, static_cast<unsigned long long>(moved), gbps);
  }
  HIP_CHECK(hipEventDestroy(b)); HIP_CHECK(hipEventDestroy(a)); HIP_CHECK(hipFree(sink)); HIP_CHECK(hipFree(data));
}

int main(int argc, char** argv) {
  bw::start();

  const std::string mode = arg(argc, argv, "--mode", "timing");
  const std::string op = arg(argc, argv, "--op", "load");
  const std::string state = arg(argc, argv, "--state", "L2-hot");
  const std::string path = arg(argc, argv, "--path", "buffer");
  const int width = argi(argc, argv, "--width", 1);
  const int repeats = argi(argc, argv, "--repeats", 30);
  const bool store = op == "store";
  if (mode == "timing") {
    bw::line("suite,repeat,issue_ticks,completion_ticks,value");
    TickResult* base_out=nullptr; HIP_CHECK(hipMalloc(&base_out,sizeof(TickResult)));
    for(int rep=0;rep<repeats;++rep){hipLaunchKernelGGL(vmem_wait_baseline_kernel,dim3(1),dim3(64),0,0,base_out); HIP_CHECK(hipGetLastError());TickResult h{};HIP_CHECK(hipMemcpy(&h,base_out,sizeof(h),hipMemcpyDeviceToHost));bw::emit("baseline,%d,%llu,%llu,%u\n",rep,(unsigned long long)h.issue_ticks,(unsigned long long)h.completion_ticks,h.value);}
    HIP_CHECK(hipFree(base_out));
  }
  if (mode == "baseline") {
    bw::line("suite,repeat,issue_ticks,completion_ticks,value");
    TickResult* d = nullptr; HIP_CHECK(hipMalloc(&d, sizeof(TickResult)));
    for (int r = 0; r < repeats; ++r) {
      hipLaunchKernelGGL(vmem_wait_baseline_kernel, dim3(1), dim3(64), 0, 0, d);
      HIP_CHECK(hipGetLastError()); TickResult h{}; HIP_CHECK(hipMemcpy(&h, d, sizeof(h), hipMemcpyDeviceToHost));
      bw::emit("baseline,%d,%llu,%llu,%u\n", r, (unsigned long long)h.issue_ticks, (unsigned long long)h.completion_ticks, h.value);
    }
    HIP_CHECK(hipFree(d));
  } else if (mode == "timing") {
    bw::line("suite,path,op,width,state,n,rounds,repeat,issue_ticks,completion_ticks,value");
#define RUN_W(B) if (width == B) { if (path == "buffer") run_timing<B, true>(state, store, repeats); else run_timing<B, false>(state, store, repeats); }
    RUN_W(1) else RUN_W(2) else RUN_W(4) else RUN_W(8) else RUN_W(12) else RUN_W(16) else return 4;
#undef RUN_W
  } else if (mode == "stream") {
    bw::line("suite,path,op,width,state,passes,repeat,event_ms,bytes,GBps");
    const uint64_t bytes = 256ull << 20;
#define RUN_S(B) if (width == B) { if (path == "buffer") { if (store) run_stream<B, true, true>(bytes, repeats); else run_stream<B, false, true>(bytes, repeats); } else { if (store) run_stream<B, true, false>(bytes, repeats); else run_stream<B, false, false>(bytes, repeats); } }
    RUN_S(1) else RUN_S(2) else RUN_S(4) else RUN_S(8) else RUN_S(12) else RUN_S(16) else return 4;
#undef RUN_S
  } else {
    std::fprintf(stderr, "Usage: %s --mode baseline|timing|stream --path buffer|global --op load|store --width 1|2|4|8|12|16 --state L2-hot|L2-cold|HBM --repeats N\n", argv[0]);
    return 1;
  }
  HIP_CHECK(hipDeviceSynchronize());
  return bw::finish("buffer_issue");
}
