/* BW1000 single-file HIP program: wave/exchange
 * Compile: hipcc -O3 -std=c++17 --offload-arch=gfx936 exchange.hip -o bench
 * Run: ./bench --suite core --repeats 30
 * Minimal run (NOT a timing reproduction): ./bench --suite core --repeats 1
 * No local headers, private libraries, scripts, data files or other source files required.
 * Only public HIP/DTK, its driver/runtime, and the C++ standard library are required.
 * Prints in-process summaries in timer tick and calibrated ns. BW1000_RAW=1 also prints samples.
 * Original measurement core: wave/src/wave_bench.cpp; SHA256=cd779013a5d702050009842986b461946368c801c065ee8f533348eac0664d74
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
    if(family=="wave"||family=="alu") {
      if(s!="dependent"&&s!="issue"&&s!="combo")continue;
      double e=num(r,"effective_ticks"),n=num(r,"instruction_count"),streams=num(r,"streams",1);
      if(n>0)put(s,family=="wave"&&s=="combo"?"original instruction-count denominator":"native instruction (original count)",e/n);
      if(family=="alu") {put(s+"_sequence","whole sequence/lane",e/(512*streams));double elements=num(r,"elements");if(elements>0)put(s+"_element","logical element/lane",e/elements);}
      if(family=="wave"&&s=="issue"&&streams>=16&&num(r,"waves")==1)batches[{t,conditions(r,{"streams"}),"wave_fit"}][int(streams)].push_back(e/n);
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


namespace {

constexpr int kIters = 256;

struct Sample {
  uint64_t ticks;
  uint32_t value;
  uint32_t pad;
};

enum Op : int {
  BPermute = 0,
  Permute = 1,
  SwizzleIdentity = 2,
  SwizzleReverse = 3,
  SwizzleXor1 = 4,
  SwizzleRotate = 5,
  SwizzleBitBroadcast = 6,
  SwizzleBitClear = 7,
  DppQuadXor1 = 8,
  DppQuadRotate = 9,
  DppQuadBroadcast0 = 10,
  DppRowShr1 = 11,
  DppRowRor1 = 12,
  DppBroadcast15 = 13,
  DppBroadcast31 = 14,
};

enum Pattern : int {
  Self = 0,
  Xor1 = 1,
  Xor2 = 2,
  Xor4 = 3,
  Xor8 = 4,
  Xor16 = 5,
  Xor32 = 6,
  WaveBroadcast0 = 7,
  Irregular = 8,
  QuadRotate = 9,
  QuadBroadcast0 = 10,
};

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t e_ = (call);                                                     \
    if (e_ != hipSuccess) {                                                     \
      std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,       \
                   hipGetErrorString(e_));                                     \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

template <int P>
__device__ __forceinline__ uint32_t lane_index(uint32_t lane) {
  uint32_t source = lane;
  if constexpr (P == Xor1) source = lane ^ 1u;
  else if constexpr (P == Xor2) source = lane ^ 2u;
  else if constexpr (P == Xor4) source = lane ^ 4u;
  else if constexpr (P == Xor8) source = lane ^ 8u;
  else if constexpr (P == Xor16) source = lane ^ 16u;
  else if constexpr (P == Xor32) source = lane ^ 32u;
  else if constexpr (P == WaveBroadcast0) source = 0u;
  else if constexpr (P == Irregular) source = (lane * 13u + 7u) & 63u;
  else if constexpr (P == QuadRotate) source = (lane & ~3u) | ((lane + 1u) & 3u);
  else if constexpr (P == QuadBroadcast0) source = lane & ~3u;
  return source << 2;
}

template <int O>
__device__ __forceinline__ uint32_t issue_op(uint32_t index, uint32_t x) {
  uint32_t y;
  if constexpr (O == BPermute) {
    asm volatile("ds_bpermute_b32 %0, %1, %2" : "=v"(y) : "v"(index), "v"(x) : "memory");
  } else if constexpr (O == Permute) {
    asm volatile("ds_permute_b32 %0, %1, %2" : "=v"(y) : "v"(index), "v"(x) : "memory");
  } else if constexpr (O == SwizzleIdentity) {
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x80e4" : "=v"(y) : "v"(x) : "memory");
  } else if constexpr (O == SwizzleReverse) {
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x801b" : "=v"(y) : "v"(x) : "memory");
  } else if constexpr (O == SwizzleXor1) {
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x80b1" : "=v"(y) : "v"(x) : "memory");
  } else if constexpr (O == SwizzleRotate) {
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x8039" : "=v"(y) : "v"(x) : "memory");
  } else if constexpr (O == SwizzleBitBroadcast) {
    asm volatile("ds_swizzle_b32 %0, %1 offset:0xe4" : "=v"(y) : "v"(x) : "memory");
  } else if constexpr (O == SwizzleBitClear) {
    asm volatile("ds_swizzle_b32 %0, %1 offset:0x1b" : "=v"(y) : "v"(x) : "memory");
  } else if constexpr (O == DppQuadXor1) {
    asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
                 : "=v"(y) : "v"(x));
  } else if constexpr (O == DppQuadRotate) {
    asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[1,2,3,0] row_mask:0xf bank_mask:0xf"
                 : "=v"(y) : "v"(x));
  } else if constexpr (O == DppQuadBroadcast0) {
    asm volatile("v_mov_b32_dpp %0, %1 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf"
                 : "=v"(y) : "v"(x));
  } else if constexpr (O == DppRowShr1) {
    asm volatile("v_mov_b32_dpp %0, %1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1"
                 : "=v"(y) : "v"(x));
  } else if constexpr (O == DppRowRor1) {
    asm volatile("v_mov_b32_dpp %0, %1 row_ror:1 row_mask:0xf bank_mask:0xf"
                 : "=v"(y) : "v"(x));
  } else if constexpr (O == DppBroadcast15) {
    asm volatile("v_mov_b32_dpp %0, %1 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:1"
                 : "=v"(y) : "v"(x));
  } else {
    asm volatile("v_mov_b32_dpp %0, %1 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1"
                 : "=v"(y) : "v"(x));
  }
  return y;
}

template <int O>
__device__ __forceinline__ constexpr bool is_ds() {
  return O <= SwizzleBitClear;
}

template <int O>
__device__ __forceinline__ constexpr bool is_dpp() {
  return O >= DppQuadXor1;
}

__device__ __forceinline__ void wait_lgkm() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

template <int O>
__device__ __forceinline__ uint32_t dependent_op(uint32_t index, uint32_t x) {
  if constexpr (is_dpp<O>()) asm volatile("s_nop 1");
  uint32_t y = issue_op<O>(index, x);
  if constexpr (is_ds<O>()) wait_lgkm();
  return y;
}

template <int O>
__device__ __forceinline__ void baseline_step(uint32_t& x) {
  asm volatile("" : "+v"(x));
  if constexpr (is_dpp<O>()) asm volatile("s_nop 1");
  if constexpr (is_ds<O>()) wait_lgkm();
}

template <int O, int P, bool Baseline>
__global__ void dependent_kernel(Sample* out, uint32_t seed) {
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t wave = threadIdx.x >> 6;
  const uint32_t index = lane_index<P>(lane);
  uint32_t x = seed + lane * 0x101u + wave * 0x10001u;
  asm volatile("" ::: "memory");
  const uint64_t t0 = clock64();
#pragma unroll 16
  for (int i = 0; i < kIters; ++i) {
    if constexpr (Baseline) baseline_step<O>(x);
    else x = dependent_op<O>(index, x);
  }
  const uint64_t t1 = clock64();
  asm volatile("" ::: "memory");
  if (lane == 0) out[wave] = {t1 - t0, x, 0};
}

template <int O, int P, int Streams, bool Baseline>
__global__ void issue_kernel(Sample* out, uint32_t seed) {
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t wave = threadIdx.x >> 6;
  const uint32_t index = lane_index<P>(lane);
  uint32_t x[64];
#pragma unroll
  for (int j = 0; j < 64; ++j) x[j] = seed + lane * 0x101u + wave * 0x10001u + j * 0x01010101u;
  asm volatile("" ::: "memory");
  const uint64_t t0 = clock64();
#pragma unroll 8
  for (int i = 0; i < kIters; ++i) {
#pragma unroll
    for (int j = 0; j < Streams; ++j) {
      if constexpr (Baseline) {
        asm volatile("" : "+v"(x[j]));
        if constexpr (is_dpp<O>() && Streams == 1) asm volatile("s_nop 1");
      } else {
        if constexpr (is_dpp<O>() && Streams == 1) asm volatile("s_nop 1");
        x[j] = issue_op<O>(index, x[j]);
      }
    }
    if constexpr (is_ds<O>()) wait_lgkm();
  }
  const uint64_t t1 = clock64();
  asm volatile("" ::: "memory");
  uint32_t sink = 0;
#pragma unroll
  for (int j = 0; j < Streams; ++j) sink ^= x[j];
  if (lane == 0) out[wave] = {t1 - t0, sink, 0};
}

template <int O, int P, bool Baseline>
__global__ void combo_kernel(Sample* out, uint32_t seed) {
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t wave = threadIdx.x >> 6;
  const uint32_t index = lane_index<P>(lane);
  uint32_t x = seed + lane + wave * 257u;
  asm volatile("" ::: "memory");
  const uint64_t t0 = clock64();
#pragma unroll 16
  for (int i = 0; i < kIters; ++i) {
    if constexpr (Baseline) {
      asm volatile("" : "+v"(x));
      if constexpr (is_ds<O>()) wait_lgkm();
    } else {
      asm volatile("v_add_u32 %0, %1, 1" : "=v"(x) : "v"(x));
      x = dependent_op<O>(index, x);
      asm volatile("v_xor_b32 %0, 0x9e3779b9, %1" : "=v"(x) : "v"(x));
    }
  }
  const uint64_t t1 = clock64();
  asm volatile("" ::: "memory");
  if (lane == 0) out[wave] = {t1 - t0, x, 0};
}

template <int O, int P>
__global__ void correctness_kernel(uint32_t* out) {
  const uint32_t lane = threadIdx.x & 63u;
  uint32_t x = 1000u + lane;
  x = dependent_op<O>(lane_index<P>(lane), x);
  out[lane] = x;
}

template <typename LaunchTarget, typename LaunchBase>
void collect_case(const char* category, const char* instruction, const char* pattern,
                  int streams, int waves, int repeats, int instructions,
                  LaunchTarget launch_target, LaunchBase launch_base) {
  const size_t n = static_cast<size_t>(waves) * repeats;
  Sample *d_target = nullptr, *d_base = nullptr;
  HIP_CHECK(hipMalloc(&d_target, n * sizeof(Sample)));
  HIP_CHECK(hipMalloc(&d_base, n * sizeof(Sample)));
  for (int i = 0; i < 5; ++i) {
    launch_target(d_target, waves, 0x12345678u + i);
    launch_base(d_base, waves, 0x12345678u + i);
  }
  HIP_CHECK(hipDeviceSynchronize());
  for (int r = 0; r < repeats; ++r) {
    Sample* target_slot = d_target + static_cast<size_t>(r) * waves;
    Sample* base_slot = d_base + static_cast<size_t>(r) * waves;
    if ((r & 1) == 0) {
      launch_base(base_slot, waves, 0x9e370000u + r);
      launch_target(target_slot, waves, 0x9e370000u + r);
    } else {
      launch_target(target_slot, waves, 0x9e370000u + r);
      launch_base(base_slot, waves, 0x9e370000u + r);
    }
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<Sample> target(n), base(n);
  HIP_CHECK(hipMemcpy(target.data(), d_target, n * sizeof(Sample), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(base.data(), d_base, n * sizeof(Sample), hipMemcpyDeviceToHost));
  for (int r = 0; r < repeats; ++r) {
    for (int w = 0; w < waves; ++w) {
      const size_t i = static_cast<size_t>(r) * waves + w;
      const int64_t effective = static_cast<int64_t>(target[i].ticks) - static_cast<int64_t>(base[i].ticks);
      bw::emit("%s,%s,%s,%d,%d,%d,%d,%llu,%llu,%lld,%u\n",
                  category, instruction, pattern, streams, waves, r, instructions,
                  static_cast<unsigned long long>(target[i].ticks),
                  static_cast<unsigned long long>(base[i].ticks),
                  static_cast<long long>(effective), target[i].value);
    }
  }
  HIP_CHECK(hipFree(d_target));
  HIP_CHECK(hipFree(d_base));
}

template <int O, int P>
void run_dep(const char* instruction, const char* pattern, int repeats, int waves = 1) {
  auto target = [](Sample* p, int w, uint32_t seed) {
    dependent_kernel<O, P, false><<<1, w * 64>>>(p, seed);
  };
  auto base = [](Sample* p, int w, uint32_t seed) {
    dependent_kernel<O, P, true><<<1, w * 64>>>(p, seed);
  };
  collect_case("dependent", instruction, pattern, 1, waves, repeats, kIters, target, base);
}

template <int O, int P, int S>
void run_issue_one(const char* instruction, const char* pattern, int repeats, int waves = 1) {
  auto target = [](Sample* p, int w, uint32_t seed) {
    issue_kernel<O, P, S, false><<<1, w * 64>>>(p, seed);
  };
  auto base = [](Sample* p, int w, uint32_t seed) {
    issue_kernel<O, P, S, true><<<1, w * 64>>>(p, seed);
  };
  collect_case("issue", instruction, pattern, S, waves, repeats, kIters * S, target, base);
}

template <int O, int P>
void run_issue_set(const char* instruction, const char* pattern, int repeats, int waves = 1) {
  run_issue_one<O, P, 1>(instruction, pattern, repeats, waves);
  run_issue_one<O, P, 2>(instruction, pattern, repeats, waves);
  run_issue_one<O, P, 4>(instruction, pattern, repeats, waves);
  run_issue_one<O, P, 8>(instruction, pattern, repeats, waves);
}

template <int O, int P>
void run_issue_extension(const char* instruction, const char* pattern, int repeats, int waves = 1) {
  run_issue_one<O, P, 16>(instruction, pattern, repeats, waves);
  run_issue_one<O, P, 32>(instruction, pattern, repeats, waves);
  run_issue_one<O, P, 64>(instruction, pattern, repeats, waves);
}

template <int O, int P>
void run_combo(const char* instruction, const char* pattern, int repeats, int waves = 1) {
  auto target = [](Sample* p, int w, uint32_t seed) {
    combo_kernel<O, P, false><<<1, w * 64>>>(p, seed);
  };
  auto base = [](Sample* p, int w, uint32_t seed) {
    combo_kernel<O, P, true><<<1, w * 64>>>(p, seed);
  };
  collect_case("combo", instruction, pattern, 1, waves, repeats, kIters, target, base);
}

template <int O, int P>
void emit_correctness(const char* instruction, const char* pattern) {
  uint32_t* d = nullptr;
  HIP_CHECK(hipMalloc(&d, 64 * sizeof(uint32_t)));
  correctness_kernel<O, P><<<1, 64>>>(d);
  HIP_CHECK(hipDeviceSynchronize());
  uint32_t h[64];
  HIP_CHECK(hipMemcpy(h, d, sizeof(h), hipMemcpyDeviceToHost));
  for (int lane = 0; lane < 64; ++lane)
    bw::emit("correctness,%s,%s,0,1,%d,1,0,0,0,%u\n", instruction, pattern, lane, h[lane]);
  HIP_CHECK(hipFree(d));
}

void run_core(int repeats) {
  run_dep<BPermute, Self>("ds_bpermute_b32", "self", repeats);
  run_dep<BPermute, Xor1>("ds_bpermute_b32", "xor1", repeats);
  run_dep<BPermute, Xor2>("ds_bpermute_b32", "xor2", repeats);
  run_dep<BPermute, Xor4>("ds_bpermute_b32", "xor4", repeats);
  run_dep<BPermute, Xor8>("ds_bpermute_b32", "xor8", repeats);
  run_dep<BPermute, Xor16>("ds_bpermute_b32", "xor16", repeats);
  run_dep<BPermute, Xor32>("ds_bpermute_b32", "xor32", repeats);
  run_dep<BPermute, WaveBroadcast0>("ds_bpermute_b32", "wave_broadcast0", repeats);
  run_dep<BPermute, Irregular>("ds_bpermute_b32", "irregular_13x_plus7", repeats);
  run_dep<BPermute, QuadRotate>("ds_bpermute_b32", "quad_rotate", repeats);
  run_dep<BPermute, QuadBroadcast0>("ds_bpermute_b32", "quad_broadcast0", repeats);
  run_issue_set<BPermute, Xor1>("ds_bpermute_b32", "xor1", repeats);
  run_issue_set<BPermute, Xor32>("ds_bpermute_b32", "xor32", repeats);
  run_issue_set<BPermute, WaveBroadcast0>("ds_bpermute_b32", "wave_broadcast0", repeats);
  run_issue_set<BPermute, Irregular>("ds_bpermute_b32", "irregular_13x_plus7", repeats);
  run_issue_set<BPermute, QuadRotate>("ds_bpermute_b32", "quad_rotate", repeats);
  run_issue_set<BPermute, QuadBroadcast0>("ds_bpermute_b32", "quad_broadcast0", repeats);
  run_issue_extension<BPermute, Xor1>("ds_bpermute_b32", "xor1", repeats);

  run_dep<Permute, Self>("ds_permute_b32", "self", repeats);
  run_dep<Permute, Xor1>("ds_permute_b32", "xor1", repeats);
  run_dep<Permute, Xor2>("ds_permute_b32", "xor2", repeats);
  run_dep<Permute, Xor4>("ds_permute_b32", "xor4", repeats);
  run_dep<Permute, Xor8>("ds_permute_b32", "xor8", repeats);
  run_dep<Permute, Xor16>("ds_permute_b32", "xor16", repeats);
  run_dep<Permute, Xor32>("ds_permute_b32", "xor32", repeats);
  run_dep<Permute, Irregular>("ds_permute_b32", "irregular_13x_plus7", repeats);
  run_issue_set<Permute, Xor1>("ds_permute_b32", "xor1", repeats);
  run_issue_set<Permute, Xor32>("ds_permute_b32", "xor32", repeats);
  run_issue_set<Permute, Irregular>("ds_permute_b32", "irregular_13x_plus7", repeats);
  run_issue_extension<Permute, Xor1>("ds_permute_b32", "xor1", repeats);

  run_dep<SwizzleIdentity, Self>("ds_swizzle_b32", "quad_identity_0x80e4", repeats);
  run_dep<SwizzleReverse, Self>("ds_swizzle_b32", "quad_reverse_0x801b", repeats);
  run_dep<SwizzleXor1, Self>("ds_swizzle_b32", "quad_xor1_0x80b1", repeats);
  run_dep<SwizzleRotate, Self>("ds_swizzle_b32", "quad_rotate_0x8039", repeats);
  run_dep<SwizzleBitBroadcast, Self>("ds_swizzle_b32", "bitmode_and4_or7_xor0_0x00e4", repeats);
  run_dep<SwizzleBitClear, Self>("ds_swizzle_b32", "bitmode_and27_or0_xor0_0x001b", repeats);
  run_issue_set<SwizzleIdentity, Self>("ds_swizzle_b32", "quad_identity_0x80e4", repeats);
  run_issue_set<SwizzleReverse, Self>("ds_swizzle_b32", "quad_reverse_0x801b", repeats);
  run_issue_set<SwizzleXor1, Self>("ds_swizzle_b32", "quad_xor1_0x80b1", repeats);
  run_issue_set<SwizzleRotate, Self>("ds_swizzle_b32", "quad_rotate_0x8039", repeats);
  run_issue_set<SwizzleBitBroadcast, Self>("ds_swizzle_b32", "bitmode_and4_or7_xor0_0x00e4", repeats);
  run_issue_set<SwizzleBitClear, Self>("ds_swizzle_b32", "bitmode_and27_or0_xor0_0x001b", repeats);
  run_issue_extension<SwizzleXor1, Self>("ds_swizzle_b32", "quad_xor1_0x80b1", repeats);

  run_dep<DppQuadXor1, Self>("v_mov_b32_dpp", "quad_xor1", repeats);
  run_dep<DppQuadRotate, Self>("v_mov_b32_dpp", "quad_rotate", repeats);
  run_dep<DppQuadBroadcast0, Self>("v_mov_b32_dpp", "quad_broadcast0", repeats);
  run_dep<DppRowShr1, Self>("v_mov_b32_dpp", "row_shr1", repeats);
  run_dep<DppRowRor1, Self>("v_mov_b32_dpp", "row_ror1", repeats);
  run_dep<DppBroadcast15, Self>("v_mov_b32_dpp", "row_bcast15", repeats);
  run_dep<DppBroadcast31, Self>("v_mov_b32_dpp", "row_bcast31", repeats);
  run_issue_set<DppQuadXor1, Self>("v_mov_b32_dpp", "quad_xor1", repeats);
  run_issue_set<DppQuadRotate, Self>("v_mov_b32_dpp", "quad_rotate", repeats);
  run_issue_set<DppQuadBroadcast0, Self>("v_mov_b32_dpp", "quad_broadcast0", repeats);
  run_issue_set<DppRowShr1, Self>("v_mov_b32_dpp", "row_shr1", repeats);
  run_issue_set<DppRowRor1, Self>("v_mov_b32_dpp", "row_ror1", repeats);
  run_issue_set<DppBroadcast15, Self>("v_mov_b32_dpp", "row_bcast15", repeats);
  run_issue_set<DppBroadcast31, Self>("v_mov_b32_dpp", "row_bcast31", repeats);
}

void run_occupancy(int repeats) {
  for (int waves : {1, 2, 4, 8}) {
    run_dep<BPermute, Xor1>("ds_bpermute_b32", "xor1", repeats, waves);
    run_dep<SwizzleXor1, Self>("ds_swizzle_b32", "quad_xor1_0x80b1", repeats, waves);
    run_dep<DppQuadXor1, Self>("v_mov_b32_dpp", "quad_xor1", repeats, waves);
  }
}

void run_combos(int repeats) {
  run_combo<DppQuadXor1, Self>("valu_dpp_valu", "quad_xor1", repeats);
  run_combo<BPermute, Xor1>("valu_ds_bpermute_valu", "xor1", repeats);
  run_combo<SwizzleXor1, Self>("valu_ds_swizzle_valu", "quad_xor1_0x80b1", repeats);
}

void run_correctness() {
  emit_correctness<BPermute, Xor1>("ds_bpermute_b32", "xor1");
  emit_correctness<BPermute, QuadRotate>("ds_bpermute_b32", "quad_rotate");
  emit_correctness<BPermute, QuadBroadcast0>("ds_bpermute_b32", "quad_broadcast0");
  emit_correctness<Permute, Xor1>("ds_permute_b32", "xor1");
  emit_correctness<SwizzleIdentity, Self>("ds_swizzle_b32", "quad_identity_0x80e4");
  emit_correctness<SwizzleReverse, Self>("ds_swizzle_b32", "quad_reverse_0x801b");
  emit_correctness<SwizzleXor1, Self>("ds_swizzle_b32", "quad_xor1_0x80b1");
  emit_correctness<SwizzleRotate, Self>("ds_swizzle_b32", "quad_rotate_0x8039");
  emit_correctness<SwizzleBitBroadcast, Self>("ds_swizzle_b32", "bitmode_and4_or7_xor0_0x00e4");
  emit_correctness<SwizzleBitClear, Self>("ds_swizzle_b32", "bitmode_and27_or0_xor0_0x001b");
  emit_correctness<DppQuadXor1, Self>("v_mov_b32_dpp", "quad_xor1");
  emit_correctness<DppQuadRotate, Self>("v_mov_b32_dpp", "quad_rotate");
  emit_correctness<DppQuadBroadcast0, Self>("v_mov_b32_dpp", "quad_broadcast0");
}

}  // namespace

int main(int argc, char** argv) {
  bw::start();

  std::string suite = "core";
  int repeats = 40;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--suite") && i + 1 < argc) suite = argv[++i];
    else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = std::atoi(argv[++i]);
  }
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::fprintf(stderr, "device=%s arch=%s wave=%d CUs=%d suite=%s repeats=%d\n",
               prop.name, prop.gcnArchName, prop.warpSize, prop.multiProcessorCount,
               suite.c_str(), repeats);
  bw::line("category,instruction,pattern,streams,waves,repeat,instruction_count,measured_ticks,baseline_ticks,effective_ticks,value");
  if (suite == "core") run_core(repeats);
  else if (suite == "occupancy") run_occupancy(repeats);
  else if (suite == "combos") run_combos(repeats);
  else if (suite == "correctness") run_correctness();
  else if (suite == "profile") {
    run_dep<BPermute, Xor1>("ds_bpermute_b32", "xor1", repeats);
    run_dep<SwizzleXor1, Self>("ds_swizzle_b32", "quad_xor1_0x80b1", repeats);
    run_dep<DppQuadXor1, Self>("v_mov_b32_dpp", "quad_xor1", repeats);
  } else {
    std::fprintf(stderr, "unknown suite: %s\n", suite.c_str());
    return 1;
  }
  return bw::finish("wave");
}
