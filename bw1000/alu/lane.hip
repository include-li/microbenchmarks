/* BW1000 single-file HIP program: alu/lane
 * Compile: hipcc -O3 -std=c++17 --offload-arch=gfx936 lane.hip -o bench
 * Run: ./bench --suite core --repeats 30
 * Minimal run (NOT a timing reproduction): ./bench --suite core --repeats 1
 * No local headers, private libraries, scripts, data files or other source files required.
 * Only public HIP/DTK, its driver/runtime, and the C++ standard library are required.
 * Prints in-process summaries in timer tick and calibrated ns. BW1000_RAW=1 also prints samples.
 * Original measurement core: lane/src/lane_bench.cpp; SHA256=09f3b6b356589a299f826c666c41f944b918786006e6f4c421a261dc3dfe40de
 * Host-only additions: calibration, sample grouping, original normalization/statistics, defaults.
 * Historical published measurements remain in README/results.csv; new runs do not replace them.
 */
// Self-contained host-side CSV observation and statistics. No subprocesses or private includes.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

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

constexpr int kIters = 512;

struct Sample {
  uint64_t ticks;
  uint32_t value;
  uint32_t value2;
};

struct State {
  uint32_t a;
  uint32_t b;
  uint32_t c;
};

enum Op : int {
  VPerm = 0,
  F16RoundTrip = 1,
  F16PackRoundTrip = 2,
  PkAddF16 = 3,
  PkMulF16 = 4,
  PkFmaF16 = 5,
  ScalarAddF16Pair = 6,
  ScalarMulF16Pair = 7,
  ScalarFmaF16Pair = 8,
  PackPkAddUnpackF16 = 9,
  Bf16RoundTrip = 10,
  Bf16ScalarAddPair = 11,
  Bf16PackedContainerAddPair = 12,
  PermConvertPackedAlu = 13,
  Bf16ScalarMulPair = 14,
  Bf16PackedContainerMulPair = 15,
  Bf16ScalarFmaPair = 16,
  Bf16PackedContainerFmaPair = 17,
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

__device__ __forceinline__ uint32_t as_u32(float x) {
  union { float f; uint32_t u; } v{x};
  return v.u;
}

__device__ __forceinline__ float as_f32(uint32_t x) {
  union { float f; uint32_t u; } v;
  v.u = x;
  return v.f;
}

__device__ __forceinline__ uint32_t vperm(uint32_t x, uint32_t y, uint32_t selector) {
  uint32_t z;
  asm volatile("v_perm_b32 %0, %1, %2, %3" : "=v"(z) : "v"(x), "v"(y), "s"(selector));
  return z;
}

__device__ __forceinline__ uint32_t cvt_f16_f32(uint32_t x) {
  uint32_t y;
  asm volatile("v_cvt_f16_f32 %0, %1" : "=v"(y) : "v"(x));
  return y;
}

__device__ __forceinline__ uint32_t cvt_f32_f16_lo(uint32_t x) {
  uint32_t y;
  asm volatile("v_cvt_f32_f16 %0, %1" : "=v"(y) : "v"(x));
  return y;
}

__device__ __forceinline__ uint32_t cvt_f32_f16_hi(uint32_t x) {
  uint32_t y;
  asm volatile("v_cvt_f32_f16_sdwa %0, %1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1"
               : "=v"(y) : "v"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack_f16(uint32_t x, uint32_t y) {
  uint32_t z;
  asm volatile("v_pack_b32_f16 %0, %1, %2" : "=v"(z) : "v"(x), "v"(y));
  return z;
}

__device__ __forceinline__ uint32_t pk_add_f16(uint32_t x, uint32_t y) {
  uint32_t z;
  asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(z) : "v"(x), "v"(y));
  return z;
}

__device__ __forceinline__ uint32_t pk_mul_f16(uint32_t x, uint32_t y) {
  uint32_t z;
  asm volatile("v_pk_mul_f16 %0, %1, %2" : "=v"(z) : "v"(x), "v"(y));
  return z;
}

__device__ __forceinline__ uint32_t pk_fma_f16(uint32_t x, uint32_t y, uint32_t z0) {
  uint32_t z;
  asm volatile("v_pk_fma_f16 %0, %1, %2, %3" : "=v"(z) : "v"(x), "v"(y), "v"(z0));
  return z;
}

__device__ __forceinline__ uint32_t add_f16(uint32_t x, uint32_t y) {
  uint32_t z;
  asm volatile("v_add_f16 %0, %1, %2" : "=v"(z) : "v"(x), "v"(y));
  return z;
}

__device__ __forceinline__ uint32_t mul_f16(uint32_t x, uint32_t y) {
  uint32_t z;
  asm volatile("v_mul_f16 %0, %1, %2" : "=v"(z) : "v"(x), "v"(y));
  return z;
}

__device__ __forceinline__ uint32_t fma_f16(uint32_t x, uint32_t y, uint32_t z0) {
  uint32_t z;
  asm volatile("v_fma_f16 %0, %1, %2, %3" : "=v"(z) : "v"(x), "v"(y), "v"(z0));
  return z;
}

__device__ __forceinline__ hip_bfloat16 bf16_from_bits(uint16_t bits) {
  hip_bfloat16 x;
  x.data = bits;
  return x;
}

__device__ __forceinline__ uint16_t bf16_bits(hip_bfloat16 x) {
  return x.data;
}

template <int O>
__device__ __forceinline__ void step(State& s, uint32_t selector) {
  if constexpr (O == VPerm) {
    s.a = vperm(s.a, s.b, selector);
  } else if constexpr (O == F16RoundTrip) {
    s.a = cvt_f32_f16_lo(cvt_f16_f32(s.a));
  } else if constexpr (O == F16PackRoundTrip) {
    uint32_t p = pack_f16(s.a, s.b);
    s.a = cvt_f32_f16_lo(p);
    s.b = cvt_f32_f16_hi(p);
  } else if constexpr (O == PkAddF16) {
    s.a = pk_add_f16(s.a, s.b);
  } else if constexpr (O == PkMulF16) {
    s.a = pk_mul_f16(s.a, s.b);
  } else if constexpr (O == PkFmaF16) {
    s.a = pk_fma_f16(s.a, s.b, s.c);
  } else if constexpr (O == ScalarAddF16Pair) {
    s.a = add_f16(s.a, s.c);
    s.b = add_f16(s.b, s.c);
  } else if constexpr (O == ScalarMulF16Pair) {
    s.a = mul_f16(s.a, s.c);
    s.b = mul_f16(s.b, s.c);
  } else if constexpr (O == ScalarFmaF16Pair) {
    s.a = fma_f16(s.a, s.c, 0u);
    s.b = fma_f16(s.b, s.c, 0u);
  } else if constexpr (O == PackPkAddUnpackF16) {
    uint32_t p = vperm(s.b, s.a, selector);
    p = pk_add_f16(p, s.c);
    asm volatile("v_and_b32 %0, 0xffff, %1" : "=v"(s.a) : "v"(p));
    asm volatile("v_lshrrev_b32 %0, 16, %1" : "=v"(s.b) : "v"(p));
  } else if constexpr (O == Bf16RoundTrip) {
    asm volatile("" : "+v"(s.a));
    hip_bfloat16 h(as_f32(s.a));
    s.a = as_u32(static_cast<float>(h));
    asm volatile("" : "+v"(s.a));
  } else if constexpr (O == Bf16ScalarAddPair) {
    asm volatile("" : "+v"(s.a), "+v"(s.b), "+v"(s.c));
    hip_bfloat16 a = bf16_from_bits(static_cast<uint16_t>(s.a));
    hip_bfloat16 b = bf16_from_bits(static_cast<uint16_t>(s.b));
    hip_bfloat16 addend = bf16_from_bits(static_cast<uint16_t>(s.c));
    a = a + addend;
    b = b + addend;
    s.a = bf16_bits(a);
    s.b = bf16_bits(b);
    asm volatile("" : "+v"(s.a), "+v"(s.b), "+v"(s.c));
  } else if constexpr (O == Bf16PackedContainerAddPair) {
    asm volatile("" : "+v"(s.a), "+v"(s.b), "+v"(s.c));
    hip_bfloat16 a = bf16_from_bits(static_cast<uint16_t>(s.a));
    hip_bfloat16 b = bf16_from_bits(static_cast<uint16_t>(s.a >> 16));
    hip_bfloat16 add0 = bf16_from_bits(static_cast<uint16_t>(s.b));
    hip_bfloat16 add1 = bf16_from_bits(static_cast<uint16_t>(s.b >> 16));
    a = a + add0;
    b = b + add1;
    s.a = static_cast<uint32_t>(bf16_bits(a)) |
          (static_cast<uint32_t>(bf16_bits(b)) << 16);
    asm volatile("" : "+v"(s.a), "+v"(s.b), "+v"(s.c));
  } else if constexpr (O == Bf16ScalarMulPair) {
    asm volatile("" : "+v"(s.a), "+v"(s.b), "+v"(s.c));
    hip_bfloat16 a = bf16_from_bits(static_cast<uint16_t>(s.a));
    hip_bfloat16 b = bf16_from_bits(static_cast<uint16_t>(s.b));
    hip_bfloat16 multiplier = bf16_from_bits(static_cast<uint16_t>(s.c));
    a = a * multiplier;
    b = b * multiplier;
    s.a = bf16_bits(a);
    s.b = bf16_bits(b);
    asm volatile("" : "+v"(s.a), "+v"(s.b), "+v"(s.c));
  } else if constexpr (O == Bf16PackedContainerMulPair) {
    asm volatile("" : "+v"(s.a), "+v"(s.b), "+v"(s.c));
    hip_bfloat16 a = bf16_from_bits(static_cast<uint16_t>(s.a));
    hip_bfloat16 b = bf16_from_bits(static_cast<uint16_t>(s.a >> 16));
    hip_bfloat16 m0 = bf16_from_bits(static_cast<uint16_t>(s.b));
    hip_bfloat16 m1 = bf16_from_bits(static_cast<uint16_t>(s.b >> 16));
    a = a * m0;
    b = b * m1;
    s.a = static_cast<uint32_t>(bf16_bits(a)) |
          (static_cast<uint32_t>(bf16_bits(b)) << 16);
    asm volatile("" : "+v"(s.a), "+v"(s.b), "+v"(s.c));
  } else if constexpr (O == Bf16ScalarFmaPair) {
    asm volatile("" : "+v"(s.a), "+v"(s.b), "+v"(s.c));
    hip_bfloat16 a = bf16_from_bits(static_cast<uint16_t>(s.a));
    hip_bfloat16 b = bf16_from_bits(static_cast<uint16_t>(s.b));
    hip_bfloat16 multiplier = bf16_from_bits(static_cast<uint16_t>(s.c));
    hip_bfloat16 addend = bf16_from_bits(static_cast<uint16_t>(s.c >> 16));
    a = hip_bfloat16(static_cast<float>(a) * static_cast<float>(multiplier) + static_cast<float>(addend));
    b = hip_bfloat16(static_cast<float>(b) * static_cast<float>(multiplier) + static_cast<float>(addend));
    s.a = bf16_bits(a);
    s.b = bf16_bits(b);
    asm volatile("" : "+v"(s.a), "+v"(s.b), "+v"(s.c));
  } else if constexpr (O == Bf16PackedContainerFmaPair) {
    asm volatile("" : "+v"(s.a), "+v"(s.b), "+v"(s.c));
    hip_bfloat16 a = bf16_from_bits(static_cast<uint16_t>(s.a));
    hip_bfloat16 b = bf16_from_bits(static_cast<uint16_t>(s.a >> 16));
    hip_bfloat16 m0 = bf16_from_bits(static_cast<uint16_t>(s.b));
    hip_bfloat16 m1 = bf16_from_bits(static_cast<uint16_t>(s.b >> 16));
    hip_bfloat16 c0 = bf16_from_bits(static_cast<uint16_t>(s.c));
    hip_bfloat16 c1 = bf16_from_bits(static_cast<uint16_t>(s.c >> 16));
    a = hip_bfloat16(static_cast<float>(a) * static_cast<float>(m0) + static_cast<float>(c0));
    b = hip_bfloat16(static_cast<float>(b) * static_cast<float>(m1) + static_cast<float>(c1));
    s.a = static_cast<uint32_t>(bf16_bits(a)) |
          (static_cast<uint32_t>(bf16_bits(b)) << 16);
    asm volatile("" : "+v"(s.a), "+v"(s.b), "+v"(s.c));
  } else {
    uint32_t p = vperm(s.a, s.b, selector);
    uint32_t f = cvt_f32_f16_lo(p);
    uint32_t h = cvt_f16_f32(f);
    p = vperm(p, h, 0x07060500u);
    s.a = pk_add_f16(p, s.c);
  }
}

template <int O>
__device__ __forceinline__ State initial_state(uint32_t lane, uint32_t wave, uint32_t seed) {
  State s{};
  if constexpr (O == F16RoundTrip || O == F16PackRoundTrip || O == Bf16RoundTrip) {
    s.a = as_u32(1.0f + static_cast<float>((lane + seed) & 31u) * 0.001f);
    s.b = as_u32(2.0f + static_cast<float>((lane + wave + seed) & 15u) * 0.001f);
    s.c = 0;
  } else if constexpr (O == PkAddF16 || O == PackPkAddUnpackF16) {
    s.a = 0x3c003c00u | ((lane & 7u) << 16) | (lane & 7u);
    s.b = (O == PkAddF16) ? (0x04000400u | ((lane & 7u) << 16) | (lane & 7u))
                          : (0x4000u + (lane & 7u));
    s.c = 0x04000400u | ((lane & 7u) << 16) | (lane & 7u);
  } else if constexpr (O == PkMulF16 || O == PkFmaF16) {
    s.a = 0x3c003c00u | ((lane & 7u) << 16) | (lane & 7u);
    s.b = 0x3c003c00u | ((lane & 7u) << 16) | (lane & 7u);
    s.c = 0x04000400u | ((lane & 7u) << 16) | (lane & 7u);
  } else if constexpr (O == ScalarAddF16Pair) {
    s.a = 0x3c00u | (lane & 7u); s.b = 0x4000u | (lane & 7u); s.c = 0x0400u | (lane & 7u);
  } else if constexpr (O == ScalarMulF16Pair || O == ScalarFmaF16Pair) {
    s.a = 0x3c00u | (lane & 7u); s.b = 0x4000u | (lane & 7u); s.c = 0x3c00u | (lane & 7u);
  } else if constexpr (O == Bf16ScalarAddPair) {
    s.a = 0x3f80u | (lane & 7u); s.b = 0x4000u | (lane & 7u); s.c = 0x3b80u | (lane & 7u);
  } else if constexpr (O == Bf16PackedContainerAddPair) {
    s.a = (0x4000u | (lane & 7u)) << 16 | (0x3f80u | (lane & 7u));
    s.b = (0x3b80u | (lane & 7u)) << 16 | (0x3b80u | (lane & 7u));
    s.c = lane;
  } else if constexpr (O == Bf16ScalarMulPair) {
    s.a = 0x3f80u | (lane & 7u); s.b = 0x4000u | (lane & 7u); s.c = 0x3f80u | (lane & 7u);
  } else if constexpr (O == Bf16PackedContainerMulPair) {
    s.a = (0x4000u | (lane & 7u)) << 16 | (0x3f80u | (lane & 7u));
    s.b = (0x3f80u | (lane & 7u)) << 16 | (0x3f80u | (lane & 7u)); s.c = lane;
  } else if constexpr (O == Bf16ScalarFmaPair) {
    s.a = 0x3f80u | (lane & 7u); s.b = 0x4000u | (lane & 7u);
    s.c = (0x3b80u | (lane & 7u)) << 16 | (0x3f80u | (lane & 7u));
  } else if constexpr (O == Bf16PackedContainerFmaPair) {
    s.a = (0x4000u | (lane & 7u)) << 16 | (0x3f80u | (lane & 7u));
    s.b = (0x3f80u | (lane & 7u)) << 16 | (0x3f80u | (lane & 7u));
    s.c = (0x3b80u | (lane & 7u)) << 16 | (0x3b80u | (lane & 7u));
  } else {
    s.a = seed + lane * 0x01010101u + wave;
    s.b = 0x76543210u ^ (seed * 17u) ^ (lane * 0x01020408u);
    s.c = lane * 0x00010001u;
  }
  return s;
}

template <int O, bool Baseline>
__global__ void dependent_kernel(Sample* out, uint32_t selector, uint32_t seed) {
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t wave = threadIdx.x >> 6;
  State s = initial_state<O>(lane, wave, seed);
  asm volatile("" ::: "memory");
  const uint64_t t0 = clock64();
#pragma unroll 16
  for (int i = 0; i < kIters; ++i) {
    if constexpr (Baseline) {
      asm volatile("" : "+v"(s.a), "+v"(s.b), "+v"(s.c));
    } else {
      step<O>(s, selector);
    }
  }
  const uint64_t t1 = clock64();
  asm volatile("" ::: "memory");
  if (lane == 0) out[wave] = {t1 - t0, s.a, s.b};
}

template <int O, int Streams, bool Baseline>
__global__ void issue_kernel(Sample* out, uint32_t selector, uint32_t seed) {
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t wave = threadIdx.x >> 6;
  State s[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) s[j] = initial_state<O>(lane, wave, seed + j * 101u);
  asm volatile("" ::: "memory");
  const uint64_t t0 = clock64();
#pragma unroll 8
  for (int i = 0; i < kIters; ++i) {
#pragma unroll
    for (int j = 0; j < Streams; ++j) {
      if constexpr (Baseline) asm volatile("" : "+v"(s[j].a), "+v"(s[j].b), "+v"(s[j].c));
      else step<O>(s[j], selector);
    }
  }
  const uint64_t t1 = clock64();
  asm volatile("" ::: "memory");
  uint32_t a = 0, b = 0;
#pragma unroll
  for (int j = 0; j < Streams; ++j) { a ^= s[j].a; b ^= s[j].b; }
  if (lane == 0) out[wave] = {t1 - t0, a, b};
}

__device__ __forceinline__ uint2 pk_add_f32(uint2 x, uint2 y) {
  uint2 z;
  asm volatile("v_pk_add_f32 %0, %1, %2" : "=v"(z) : "v"(x), "v"(y));
  return z;
}

template <int Streams, bool Baseline>
__global__ void pk_add_f32_kernel(Sample* out, uint32_t seed) {
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t wave = threadIdx.x >> 6;
  uint2 x[8], y[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j].x = as_u32(1.0f + static_cast<float>((lane + seed + j) & 31u) * 0.001f);
    x[j].y = as_u32(2.0f + static_cast<float>((lane + seed + 3 * j) & 31u) * 0.001f);
    y[j].x = as_u32(0.001f + static_cast<float>((lane + j) & 7u) * 0.00001f);
    y[j].y = as_u32(0.002f + static_cast<float>((lane + 2 * j) & 7u) * 0.00001f);
  }
  asm volatile("" ::: "memory");
  const uint64_t t0 = clock64();
#pragma unroll 8
  for (int i = 0; i < kIters; ++i) {
#pragma unroll
    for (int j = 0; j < Streams; ++j) {
      if constexpr (Baseline) asm volatile("" : "+v"(x[j]));
      else x[j] = pk_add_f32(x[j], y[j]);
    }
  }
  const uint64_t t1 = clock64();
  asm volatile("" ::: "memory");
  uint32_t a = 0, b = 0;
#pragma unroll
  for (int j = 0; j < Streams; ++j) { a ^= x[j].x; b ^= x[j].y; }
  if (lane == 0) out[wave] = {t1 - t0, a, b};
}

template <typename LaunchTarget, typename LaunchBase>
void collect_case(const char* category, const char* instruction, const char* operation,
                  uint32_t selector, int streams, int waves, int repeats,
                  int instruction_count, int elements, LaunchTarget launch_target,
                  LaunchBase launch_base) {
  const size_t n = static_cast<size_t>(waves) * repeats;
  Sample *d_target = nullptr, *d_base = nullptr;
  HIP_CHECK(hipMalloc(&d_target, n * sizeof(Sample)));
  HIP_CHECK(hipMalloc(&d_base, n * sizeof(Sample)));
  for (int i = 0; i < 5; ++i) {
    launch_target(d_target, waves, 0x12340000u + i);
    launch_base(d_base, waves, 0x12340000u + i);
  }
  HIP_CHECK(hipDeviceSynchronize());
  for (int r = 0; r < repeats; ++r) {
    Sample* t = d_target + static_cast<size_t>(r) * waves;
    Sample* b = d_base + static_cast<size_t>(r) * waves;
    if ((r & 1) == 0) {
      launch_base(b, waves, 0x9e370000u + r);
      launch_target(t, waves, 0x9e370000u + r);
    } else {
      launch_target(t, waves, 0x9e370000u + r);
      launch_base(b, waves, 0x9e370000u + r);
    }
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<Sample> target(n), base(n);
  HIP_CHECK(hipMemcpy(target.data(), d_target, n * sizeof(Sample), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(base.data(), d_base, n * sizeof(Sample), hipMemcpyDeviceToHost));
  for (int r = 0; r < repeats; ++r) {
    for (int w = 0; w < waves; ++w) {
      size_t i = static_cast<size_t>(r) * waves + w;
      int64_t effective = static_cast<int64_t>(target[i].ticks) - static_cast<int64_t>(base[i].ticks);
      bw::emit("%s,%s,%s,0x%08x,%d,%d,%d,%d,%d,%llu,%llu,%lld,%u,%u\n",
                  category, instruction, operation, selector, streams, waves, r,
                  instruction_count, elements,
                  static_cast<unsigned long long>(target[i].ticks),
                  static_cast<unsigned long long>(base[i].ticks),
                  static_cast<long long>(effective), target[i].value, target[i].value2);
    }
  }
  HIP_CHECK(hipFree(d_target));
  HIP_CHECK(hipFree(d_base));
}

template <int O>
void run_dep(const char* instruction, const char* operation, uint32_t selector,
             int per_step_instructions, int elements, int repeats, int waves = 1) {
  auto target = [selector](Sample* p, int w, uint32_t seed) {
    dependent_kernel<O, false><<<1, w * 64>>>(p, selector, seed);
  };
  auto base = [selector](Sample* p, int w, uint32_t seed) {
    dependent_kernel<O, true><<<1, w * 64>>>(p, selector, seed);
  };
  collect_case("dependent", instruction, operation, selector, 1, waves, repeats,
               per_step_instructions > 0 ? kIters * per_step_instructions : 0,
               kIters * elements, target, base);
}

template <int O, int S>
void run_issue_one(const char* instruction, const char* operation, uint32_t selector,
                   int per_step_instructions, int elements, int repeats, int waves = 1) {
  auto target = [selector](Sample* p, int w, uint32_t seed) {
    issue_kernel<O, S, false><<<1, w * 64>>>(p, selector, seed);
  };
  auto base = [selector](Sample* p, int w, uint32_t seed) {
    issue_kernel<O, S, true><<<1, w * 64>>>(p, selector, seed);
  };
  collect_case("issue", instruction, operation, selector, S, waves, repeats,
               per_step_instructions > 0 ? kIters * S * per_step_instructions : 0,
               kIters * S * elements, target, base);
}

template <int O>
void run_issue_set(const char* instruction, const char* operation, uint32_t selector,
                   int per_step_instructions, int elements, int repeats, int waves = 1) {
  run_issue_one<O, 1>(instruction, operation, selector, per_step_instructions, elements, repeats, waves);
  run_issue_one<O, 2>(instruction, operation, selector, per_step_instructions, elements, repeats, waves);
  run_issue_one<O, 4>(instruction, operation, selector, per_step_instructions, elements, repeats, waves);
  run_issue_one<O, 8>(instruction, operation, selector, per_step_instructions, elements, repeats, waves);
}

template <int S>
void run_pk_add_f32_case(const char* category, int repeats, int waves = 1) {
  auto target = [](Sample* p, int w, uint32_t seed) {
    pk_add_f32_kernel<S, false><<<1, w * 64>>>(p, seed);
  };
  auto base = [](Sample* p, int w, uint32_t seed) {
    pk_add_f32_kernel<S, true><<<1, w * 64>>>(p, seed);
  };
  collect_case(category, "v_pk_add_f32", "two_fp32_intermediates", 0, S, waves, repeats,
               kIters * S, kIters * S * 2, target, base);
}

void run_core(int repeats) {
  struct Sel { const char* name; uint32_t value; } selectors[] = {
    {"src0_identity", 0x03020100u},
    {"src1_identity", 0x07060504u},
    {"src0_src1_mix", 0x05040100u},
    {"repeat_byte0", 0x00000000u},
    {"full_reverse_mix", 0x00050603u},
  };
  for (const auto& s : selectors) {
    run_dep<VPerm>("v_perm_b32", s.name, s.value, 1, 1, repeats);
    run_issue_set<VPerm>("v_perm_b32", s.name, s.value, 1, 1, repeats);
  }

  run_dep<F16RoundTrip>("v_cvt_f16_f32+v_cvt_f32_f16", "fp32_f16_fp32", 0, 2, 1, repeats);
  run_issue_set<F16RoundTrip>("v_cvt_f16_f32+v_cvt_f32_f16", "fp32_f16_fp32", 0, 2, 1, repeats);
  run_dep<F16PackRoundTrip>("v_pack_b32_f16+2xcvt", "two_fp32_pack_unpack", 0, 3, 2, repeats);
  run_issue_set<F16PackRoundTrip>("v_pack_b32_f16+2xcvt", "two_fp32_pack_unpack", 0, 3, 2, repeats);

  run_dep<PkAddF16>("v_pk_add_f16", "packed_add_two_fp16", 0, 1, 2, repeats);
  run_issue_set<PkAddF16>("v_pk_add_f16", "packed_add_two_fp16", 0, 1, 2, repeats);
  run_dep<PkMulF16>("v_pk_mul_f16", "packed_mul_two_fp16", 0, 1, 2, repeats);
  run_issue_set<PkMulF16>("v_pk_mul_f16", "packed_mul_two_fp16", 0, 1, 2, repeats);
  run_dep<PkFmaF16>("v_pk_fma_f16", "packed_fma_two_fp16", 0, 1, 2, repeats);
  run_issue_set<PkFmaF16>("v_pk_fma_f16", "packed_fma_two_fp16", 0, 1, 2, repeats);

  run_dep<ScalarAddF16Pair>("2xv_add_f16", "scalar_add_two_fp16", 0, 2, 2, repeats);
  run_issue_set<ScalarAddF16Pair>("2xv_add_f16", "scalar_add_two_fp16", 0, 2, 2, repeats);
  run_dep<ScalarMulF16Pair>("2xv_mul_f16", "scalar_mul_two_fp16", 0, 2, 2, repeats);
  run_issue_set<ScalarMulF16Pair>("2xv_mul_f16", "scalar_mul_two_fp16", 0, 2, 2, repeats);
  run_dep<ScalarFmaF16Pair>("2xv_fma_f16", "scalar_fma_two_fp16", 0, 2, 2, repeats);
  run_issue_set<ScalarFmaF16Pair>("2xv_fma_f16", "scalar_fma_two_fp16", 0, 2, 2, repeats);

  run_pk_add_f32_case<1>("dependent", repeats);
  run_pk_add_f32_case<2>("issue", repeats);
  run_pk_add_f32_case<4>("issue", repeats);
  run_pk_add_f32_case<8>("issue", repeats);

  run_dep<PackPkAddUnpackF16>("v_perm+v_pk_add+and+shift", "scalar_half_pack_add_unpack", 0x05040100u, 4, 2, repeats);
  run_issue_set<PackPkAddUnpackF16>("v_perm+v_pk_add+and+shift", "scalar_half_pack_add_unpack", 0x05040100u, 4, 2, repeats);

  run_dep<Bf16RoundTrip>("compiler_bf16_convert_sequence", "fp32_bf16_fp32", 0, 0, 1, repeats);
  run_issue_set<Bf16RoundTrip>("compiler_bf16_convert_sequence", "fp32_bf16_fp32", 0, 0, 1, repeats);
  run_dep<Bf16ScalarAddPair>("compiler_bf16_add_sequence", "two_scalar_bf16_add", 0, 0, 2, repeats);
  run_issue_set<Bf16ScalarAddPair>("compiler_bf16_add_sequence", "two_scalar_bf16_add", 0, 0, 2, repeats);
  run_dep<Bf16PackedContainerAddPair>("compiler_bf16_packed_container_sequence", "packed_container_two_bf16_add", 0, 0, 2, repeats);
  run_issue_set<Bf16PackedContainerAddPair>("compiler_bf16_packed_container_sequence", "packed_container_two_bf16_add", 0, 0, 2, repeats);
  run_dep<Bf16ScalarMulPair>("compiler_bf16_mul_sequence", "two_scalar_bf16_mul", 0, 0, 2, repeats);
  run_issue_one<Bf16ScalarMulPair, 4>("compiler_bf16_mul_sequence", "two_scalar_bf16_mul", 0, 0, 2, repeats);
  run_dep<Bf16PackedContainerMulPair>("compiler_bf16_packed_container_mul_sequence", "packed_container_two_bf16_mul", 0, 0, 2, repeats);
  run_issue_one<Bf16PackedContainerMulPair, 4>("compiler_bf16_packed_container_mul_sequence", "packed_container_two_bf16_mul", 0, 0, 2, repeats);
  run_dep<Bf16ScalarFmaPair>("compiler_bf16_fma_sequence", "two_scalar_bf16_fma", 0, 0, 2, repeats);
  run_issue_one<Bf16ScalarFmaPair, 4>("compiler_bf16_fma_sequence", "two_scalar_bf16_fma", 0, 0, 2, repeats);
  run_dep<Bf16PackedContainerFmaPair>("compiler_bf16_packed_container_fma_sequence", "packed_container_two_bf16_fma", 0, 0, 2, repeats);
  run_issue_one<Bf16PackedContainerFmaPair, 4>("compiler_bf16_packed_container_fma_sequence", "packed_container_two_bf16_fma", 0, 0, 2, repeats);
}

void run_combos(int repeats) {
  run_dep<PackPkAddUnpackF16>("pack_packed_alu_unpack", "fp16_layout_primitive", 0x05040100u, 4, 2, repeats);
  run_dep<PermConvertPackedAlu>("v_perm_convert_v_perm_pk_add", "permutation_convert_packed_alu", 0x05040100u, 5, 2, repeats);
}

void run_occupancy(int repeats) {
  for (int waves : {1, 2, 4, 8}) {
    run_dep<VPerm>("v_perm_b32", "src0_src1_mix", 0x05040100u, 1, 1, repeats, waves);
    run_dep<PkAddF16>("v_pk_add_f16", "packed_add_two_fp16", 0, 1, 2, repeats, waves);
    run_dep<Bf16RoundTrip>("compiler_bf16_convert_sequence", "fp32_bf16_fp32", 0, 0, 1, repeats, waves);
  }
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
  bw::line("category,instruction,operation,selector,streams,waves,repeat,instruction_count,elements,measured_ticks,baseline_ticks,effective_ticks,value,value2");
  if (suite == "core") run_core(repeats);
  else if (suite == "combos") run_combos(repeats);
  else if (suite == "occupancy") run_occupancy(repeats);
  else if (suite == "profile") {
    run_dep<VPerm>("v_perm_b32", "src0_src1_mix", 0x05040100u, 1, 1, repeats);
    run_dep<PkAddF16>("v_pk_add_f16", "packed_add_two_fp16", 0, 1, 2, repeats);
    run_dep<Bf16RoundTrip>("compiler_bf16_convert_sequence", "fp32_bf16_fp32", 0, 0, 1, repeats);
  } else {
    std::fprintf(stderr, "unknown suite: %s\n", suite.c_str());
    return 1;
  }
  return bw::finish("alu");
}
