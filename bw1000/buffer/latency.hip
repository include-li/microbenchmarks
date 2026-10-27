/* BW1000 single-file HIP program: buffer/latency
 * Compile: hipcc -O2 -std=c++17 --offload-arch=gfx936 latency.hip -o bench
 * Run: ./bench --path buffer --state L1 --width 1 --repeats 30
 * Minimal run (NOT a timing reproduction): ./bench --path buffer --state L1 --width 1 --repeats 1
 * No local headers, private libraries, scripts, data files or other source files required.
 * Only public HIP/DTK, its driver/runtime, and the C++ standard library are required.
 * Prints in-process summaries in timer tick and calibrated ns. BW1000_RAW=1 also prints samples.
 * Original measurement core: buffer/src/latency_bench.cpp; SHA256=c66aefa8ea0350415f2a0313ae12af33e5386f456d0fe7712ad4cec5f3f91fc4
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
    if(family=="buffer_latency"||family=="buffer_exact") {put("dependent_adjusted","load",num(r,"effective_ticks_per_load"));continue;}
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

struct TickResult { uint64_t ticks, steps; uint32_t value, pad; };

template<int B> struct VType;
template<> struct VType<1> { using type = uint32_t; };
template<> struct VType<2> { using type = uint32_t; };
template<> struct VType<4> { using type = uint32_t; };
template<> struct VType<8> { using type = u32x2; };
template<> struct VType<12> { using type = u32x3; };
template<> struct VType<16> { using type = u32x4; };
template<int B> using VT = typename VType<B>::type;

__device__ __forceinline__ u32x4 make_rsrc(const void* base, uint32_t stride = 0) {
  const uint64_t p = reinterpret_cast<uint64_t>(base);
  return u32x4{static_cast<uint32_t>(p),
               (static_cast<uint32_t>(p >> 32) & 0xffffu) | (stride << 16),
               0x7ffffffeu, 0x00027000u};
}

__device__ __forceinline__ void wait_vmem() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }

template<int B> __device__ __forceinline__ uint32_t first_word(VT<B> v) {
  if constexpr (B <= 4) return v;
  else return v[0];
}

template<int B, int Policy> __device__ __forceinline__ VT<B> global_load(const uint8_t* p);
#define DEFINE_GLOBAL(B, TYPE, OPCODE) \
template<> __device__ __forceinline__ TYPE global_load<B, 0>(const uint8_t* p) { TYPE v; const uint64_t a = reinterpret_cast<uint64_t>(p); asm volatile(OPCODE " %0, %1, off" : "=v"(v) : "v"(a) : "memory"); return v; } \
template<> __device__ __forceinline__ TYPE global_load<B, 1>(const uint8_t* p) { TYPE v; const uint64_t a = reinterpret_cast<uint64_t>(p); asm volatile(OPCODE " %0, %1, off glc" : "=v"(v) : "v"(a) : "memory"); return v; } \
template<> __device__ __forceinline__ TYPE global_load<B, 2>(const uint8_t* p) { TYPE v; const uint64_t a = reinterpret_cast<uint64_t>(p); asm volatile(OPCODE " %0, %1, off slc" : "=v"(v) : "v"(a) : "memory"); return v; } \
template<> __device__ __forceinline__ TYPE global_load<B, 3>(const uint8_t* p) { TYPE v; const uint64_t a = reinterpret_cast<uint64_t>(p); asm volatile(OPCODE " %0, %1, off glc slc" : "=v"(v) : "v"(a) : "memory"); return v; }
DEFINE_GLOBAL(1, uint32_t, "global_load_ubyte")
DEFINE_GLOBAL(2, uint32_t, "global_load_ushort")
DEFINE_GLOBAL(4, uint32_t, "global_load_dword")
DEFINE_GLOBAL(8, u32x2, "global_load_dwordx2")
DEFINE_GLOBAL(12, u32x3, "global_load_dwordx3")
DEFINE_GLOBAL(16, u32x4, "global_load_dwordx4")
#undef DEFINE_GLOBAL

template<int B, int Policy, int Mode> __device__ __forceinline__ VT<B> buffer_load(
    u32x4 raw, u32x4 stride64, u32x4 stride32, uint32_t index, uint32_t byte_offset);

#define DEFINE_BUFFER_MODE(B, TYPE, OPCODE, POLICY, MODS) \
template<> __device__ __forceinline__ TYPE buffer_load<B, POLICY, 0>(u32x4 raw, u32x4, u32x4, uint32_t, uint32_t off) { TYPE v; asm volatile(OPCODE " %0, %1, %2, 0 offen " MODS : "=v"(v) : "v"(off), "s"(raw) : "memory"); return v; } \
template<> __device__ __forceinline__ TYPE buffer_load<B, POLICY, 1>(u32x4, u32x4 stride64, u32x4, uint32_t index, uint32_t) { TYPE v; asm volatile(OPCODE " %0, %1, %2, 0 idxen " MODS : "=v"(v) : "v"(index), "s"(stride64) : "memory"); return v; } \
template<> __device__ __forceinline__ TYPE buffer_load<B, POLICY, 2>(u32x4, u32x4, u32x4 stride32, uint32_t index, uint32_t) { TYPE v; const u32x2 pair = u32x2{index, index * 32u}; asm volatile(OPCODE " %0, %1, %2, 0 idxen offen " MODS : "=v"(v) : "v"(pair), "s"(stride32) : "memory"); return v; }

#define DEFINE_BUFFER_ALL_POLICIES(B, TYPE, OPCODE) \
DEFINE_BUFFER_MODE(B, TYPE, OPCODE, 0, "") \
DEFINE_BUFFER_MODE(B, TYPE, OPCODE, 1, "glc") \
DEFINE_BUFFER_MODE(B, TYPE, OPCODE, 2, "slc") \
DEFINE_BUFFER_MODE(B, TYPE, OPCODE, 3, "glc slc")
DEFINE_BUFFER_ALL_POLICIES(1, uint32_t, "buffer_load_ubyte")
DEFINE_BUFFER_ALL_POLICIES(2, uint32_t, "buffer_load_ushort")
DEFINE_BUFFER_ALL_POLICIES(4, uint32_t, "buffer_load_dword")
DEFINE_BUFFER_ALL_POLICIES(8, u32x2, "buffer_load_dwordx2")
DEFINE_BUFFER_ALL_POLICIES(12, u32x3, "buffer_load_dwordx3")
DEFINE_BUFFER_ALL_POLICIES(16, u32x4, "buffer_load_dwordx4")
#undef DEFINE_BUFFER_ALL_POLICIES
#undef DEFINE_BUFFER_MODE

template<int B, bool Buffer, int Policy, int Mode>
__global__ void dependent_kernel(const uint8_t* base, uint32_t nodes, uint32_t stride,
                                 uint64_t warm_steps, uint64_t steps, TickResult* out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const uint32_t mask = nodes - 1u;
  const u32x4 raw = make_rsrc(base);
  const u32x4 stride64 = make_rsrc(base, 64);
  const u32x4 stride32 = make_rsrc(base, 32);
  uint32_t index = 0, sink = 0;
#pragma clang loop unroll(disable)
  for (uint64_t i = 0; i < warm_steps; ++i) {
    const uint32_t off = index * stride;
    VT<B> v;
    if constexpr (Buffer) v = buffer_load<B, Policy, Mode>(raw, stride64, stride32, index, off);
    else v = global_load<B, Policy>(base + off);
    wait_vmem();
    const uint32_t delta = first_word<B>(v);
    index = (index + delta) & mask;
    sink ^= index;
  }
  const uint64_t t0 = clock64();
#pragma clang loop unroll(disable)
  for (uint64_t i = 0; i < steps; ++i) {
    const uint32_t off = index * stride;
    VT<B> v;
    if constexpr (Buffer) v = buffer_load<B, Policy, Mode>(raw, stride64, stride32, index, off);
    else v = global_load<B, Policy>(base + off);
    wait_vmem();
    const uint32_t delta = first_word<B>(v);
    index = (index + delta) & mask;
    sink ^= index + static_cast<uint32_t>(i);
  }
  const uint64_t t1 = clock64();
  out[0] = TickResult{t1 - t0, steps, index ^ sink, 0};
  return;
}

extern "C" __global__ void dependent_baseline_kernel(uint32_t nodes, uint32_t stride,
                                                        uint32_t delta, uint64_t steps,
                                                        TickResult* out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const uint32_t mask = nodes - 1u;
  uint32_t index = 0, sink = 0;
  const uint64_t t0 = clock64();
#pragma clang loop unroll(disable)
  for (uint64_t i = 0; i < steps; ++i) {
    uint32_t off = index * stride;
    asm volatile("" : "+v"(off));
    index = (index + delta) & mask;
    sink ^= index + static_cast<uint32_t>(i);
  }
  const uint64_t t1 = clock64();
  out[0] = TickResult{t1 - t0, steps, index ^ sink, 0};
  return;
}

static std::string arg(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i) if (!std::strcmp(argv[i], key)) return argv[i + 1];
  return def;
}
static int argi(int argc, char** argv, const char* key, int def) {
  const std::string d = std::to_string(def); return std::atoi(arg(argc, argv, key, d.c_str()).c_str());
}

template<int B, bool Buffer, int Policy, int Mode>
static void launch_dep(const uint8_t* data, uint32_t nodes, uint32_t stride,
                       uint64_t warm_steps, uint64_t steps, TickResult* out) {
  hipLaunchKernelGGL((dependent_kernel<B, Buffer, Policy, Mode>), dim3(1), dim3(64), 0, 0,
                     data, nodes, stride, warm_steps, steps, out);
}

template<int B, bool Buffer, int Policy>
static void launch_mode(int mode, const uint8_t* data, uint32_t nodes, uint32_t stride,
                        uint64_t warm_steps, uint64_t steps, TickResult* out) {
  if (mode == 0) launch_dep<B, Buffer, Policy, 0>(data, nodes, stride, warm_steps, steps, out);
  else if (mode == 1) launch_dep<B, Buffer, Policy, 1>(data, nodes, stride, warm_steps, steps, out);
  else launch_dep<B, Buffer, Policy, 2>(data, nodes, stride, warm_steps, steps, out);
}

template<int B, bool Buffer>
static void launch_policy(int policy, int mode, const uint8_t* data, uint32_t nodes,
                          uint32_t stride, uint64_t warm_steps, uint64_t steps,
                          TickResult* out) {
  if (policy == 0) launch_mode<B, Buffer, 0>(mode, data, nodes, stride, warm_steps, steps, out);
  else if (policy == 1) launch_mode<B, Buffer, 1>(mode, data, nodes, stride, warm_steps, steps, out);
  else if (policy == 2) launch_mode<B, Buffer, 2>(mode, data, nodes, stride, warm_steps, steps, out);
  else launch_mode<B, Buffer, 3>(mode, data, nodes, stride, warm_steps, steps, out);
}

template<int B>
static void run(const std::string& path, const std::string& state,
                const std::string& policy_name, const std::string& mode_name, int repeats) {
  const uint32_t stride = 64;
  uint32_t nodes = state == "L1" ? 64u : state == "L2" ? 16384u : 524288u;
  const uint64_t bytes = static_cast<uint64_t>(nodes) * stride;
  uint32_t delta = B == 1 ? 251u : B == 2 ? 32767u : 131071u;
  delta &= nodes - 1u;
  if ((delta & 1u) == 0) delta |= 1u;
  std::vector<uint8_t> host(bytes, 0x6d);
  for (uint32_t i = 0; i < nodes; ++i) {
    const uint64_t off = static_cast<uint64_t>(i) * stride;
    if constexpr (B == 1) host[off] = static_cast<uint8_t>(delta);
    else if constexpr (B == 2) { const uint16_t q = static_cast<uint16_t>(delta); std::memcpy(host.data() + off, &q, 2); }
    else std::memcpy(host.data() + off, &delta, 4);
  }
  uint8_t* data = nullptr; TickResult* result = nullptr;
  HIP_CHECK(hipMalloc(&data, bytes)); HIP_CHECK(hipMalloc(&result, sizeof(TickResult)));
  HIP_CHECK(hipMemcpy(data, host.data(), bytes, hipMemcpyHostToDevice));
  const int policy = policy_name == "glc" ? 1 : policy_name == "slc" ? 2 : policy_name == "glc_slc" ? 3 : 0;
  const int mode = mode_name == "idxen" ? 1 : mode_name == "idxen_offen" ? 2 : 0;
  const uint64_t warm_steps = state == "L1" ? 4096u : static_cast<uint64_t>(nodes);
  const uint64_t steps = state == "HBM" ? 8192u : 4096u;
  bw::line("suite,path,width,state,policy,address_mode,repeat,steps,raw_ticks,baseline_ticks,effective_ticks,raw_ticks_per_load,baseline_ticks_per_step,effective_ticks_per_load,value");
  for (int r = -2; r < repeats; ++r) {
    hipLaunchKernelGGL(dependent_baseline_kernel, dim3(1), dim3(64), 0, 0,
                       nodes, stride, delta, steps, result);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    TickResult base{}; HIP_CHECK(hipMemcpy(&base, result, sizeof(base), hipMemcpyDeviceToHost));
    if (path == "buffer") launch_policy<B, true>(policy, mode, data, nodes, stride, warm_steps, steps, result);
    else launch_policy<B, false>(policy, mode, data, nodes, stride, warm_steps, steps, result);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
    TickResult got{}; HIP_CHECK(hipMemcpy(&got, result, sizeof(got), hipMemcpyDeviceToHost));
    if (r < 0) continue;
    const uint64_t effective = got.ticks > base.ticks ? got.ticks - base.ticks : 0;
    bw::emit("dependent,%s,%d,%s,%s,%s,%d,%llu,%llu,%llu,%llu,%.9f,%.9f,%.9f,%u\n",
                path.c_str(), B, state.c_str(), policy_name.c_str(), mode_name.c_str(), r,
                static_cast<unsigned long long>(steps), static_cast<unsigned long long>(got.ticks),
                static_cast<unsigned long long>(base.ticks), static_cast<unsigned long long>(effective),
                static_cast<double>(got.ticks) / steps, static_cast<double>(base.ticks) / steps,
                static_cast<double>(effective) / steps, got.value);
  }
  HIP_CHECK(hipFree(result)); HIP_CHECK(hipFree(data));
}

int main(int argc, char** argv) {
  bw::start();

  const std::string path = arg(argc, argv, "--path", "buffer");
  const std::string state = arg(argc, argv, "--state", "L1");
  const std::string policy = arg(argc, argv, "--policy", state == "L2" ? "glc" : "default");
  const std::string mode = arg(argc, argv, "--address-mode", "offen");
  const int width = argi(argc, argv, "--width", 4);
  const int repeats = argi(argc, argv, "--repeats", 30);
#define RUN(B) if (width == B) run<B>(path, state, policy, mode, repeats)
  RUN(1); else RUN(2); else RUN(4); else RUN(8); else RUN(12); else RUN(16); else return 4;
#undef RUN
  return bw::finish("buffer_latency");
}
