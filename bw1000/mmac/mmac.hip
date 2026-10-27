/* BW1000 single-file HIP program: mmac/mmac
 * Compile: hipcc -O2 -std=c++17 --offload-arch=gfx936 mmac.hip -o bench
 * Run: ./bench bf16_acc1 30 5
 * Minimal run (NOT a timing reproduction): ./bench bf16_acc1 1 0
 * No local headers, private libraries, scripts, data files or other source files required.
 * Only public HIP/DTK, its driver/runtime, and the C++ standard library are required.
 * Prints in-process summaries in timer tick and calibrated ns. BW1000_RAW=1 also prints samples.
 * Original measurement core: mma/benchmark.cpp; SHA256=96bbe505aa05196c5f2eb86d76a2a61cc2a7bf87b94dd082730f2e5372ff5fc0
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
    if(family=="mma") {t=str(r,"config");if(t=="baseline")put("baseline","bracket",num(r,"raw_ticks"));else{put("raw","1024 native MMAC",num(r,"raw_ticks"));put("adjusted","native MMAC",(num(r,"raw_ticks")-base)/1024.0);}continue;}
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


using f32x4 = float __attribute__((ext_vector_type(4)));
using i32x4 = int __attribute__((ext_vector_type(4)));
using i16x4 = short __attribute__((ext_vector_type(4)));

static constexpr int kWaveSize = 64;
static constexpr int kMmacCount = 1024;

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t error__ = (expr);                                                \
    if (error__ != hipSuccess) {                                                \
      std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   hipGetErrorString(error__));                                 \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define REP2(X) X X
#define REP4(X) REP2(X) REP2(X)
#define REP8(X) REP4(X) REP4(X)
#define REP16(X) REP8(X) REP8(X)
#define REP32(X) REP16(X) REP16(X)
#define REP64(X) REP32(X) REP32(X)
#define REP128(X) REP64(X) REP64(X)
#define REP256(X) REP128(X) REP128(X)
#define REP512(X) REP256(X) REP256(X)
#define REP1024(X) REP512(X) REP512(X)

#define DEVICE_PREPARE()                                                        \
  asm volatile("" ::: "memory");                                               \
  __builtin_amdgcn_s_waitcnt(0);                                                \
  asm volatile("" ::: "memory")
#define DEVICE_COMPILER_BARRIER() asm volatile("" ::: "memory")

#define BF16_MMAC(ACC)                                                          \
  ACC = __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, ACC);
#define I8_MMAC(ACC)                                                            \
  ACC = __builtin_amdgcn_mmac_i32_16x16x32i8(a, b, ACC);

#define BF16_STEP1() BF16_MMAC(acc0)
#define BF16_STEP2() BF16_MMAC(acc0) BF16_MMAC(acc1)
#define BF16_STEP4() BF16_MMAC(acc0) BF16_MMAC(acc1) BF16_MMAC(acc2) BF16_MMAC(acc3)
#define BF16_STEP8()                                                            \
  BF16_MMAC(acc0) BF16_MMAC(acc1) BF16_MMAC(acc2) BF16_MMAC(acc3)              \
  BF16_MMAC(acc4) BF16_MMAC(acc5) BF16_MMAC(acc6) BF16_MMAC(acc7)

#define I8_STEP1() I8_MMAC(acc0)
#define I8_STEP2() I8_MMAC(acc0) I8_MMAC(acc1)
#define I8_STEP4() I8_MMAC(acc0) I8_MMAC(acc1) I8_MMAC(acc2) I8_MMAC(acc3)
#define I8_STEP8()                                                              \
  I8_MMAC(acc0) I8_MMAC(acc1) I8_MMAC(acc2) I8_MMAC(acc3)                      \
  I8_MMAC(acc4) I8_MMAC(acc5) I8_MMAC(acc6) I8_MMAC(acc7)

extern "C" __global__ __launch_bounds__(64, 1)
void baseline_kernel(const f32x4 *__restrict__ init,
                     unsigned long long *__restrict__ ticks,
                     f32x4 *__restrict__ out) {
  const int lane = static_cast<int>(threadIdx.x);
  const int slot = static_cast<int>(blockIdx.x) * kWaveSize + lane;
  f32x4 acc0 = init[slot * 8];
  DEVICE_PREPARE();
  const unsigned long long t0 = clock64();
  DEVICE_COMPILER_BARRIER();
  DEVICE_COMPILER_BARRIER();
  const unsigned long long t1 = clock64();
  DEVICE_COMPILER_BARRIER();
  ticks[slot] = t1 - t0;
  out[slot * 8] = acc0;
}

extern "C" __global__ __launch_bounds__(64, 1)
void bf16_acc1_kernel(const i16x4 *__restrict__ src_a,
                      const i16x4 *__restrict__ src_b,
                      const f32x4 *__restrict__ init,
                      unsigned long long *__restrict__ ticks,
                      f32x4 *__restrict__ out) {
  const int lane = static_cast<int>(threadIdx.x);
  const int slot = static_cast<int>(blockIdx.x) * kWaveSize + lane;
  const i16x4 a = src_a[lane];
  const i16x4 b = src_b[lane];
  f32x4 acc0 = init[slot * 8];
  DEVICE_PREPARE();
  const unsigned long long t0 = clock64();
  DEVICE_COMPILER_BARRIER();
  REP1024(BF16_STEP1())
  DEVICE_COMPILER_BARRIER();
  const unsigned long long t1 = clock64();
  DEVICE_COMPILER_BARRIER();
  ticks[slot] = t1 - t0;
  out[slot * 8] = acc0;
}

extern "C" __global__ __launch_bounds__(64, 1)
void bf16_acc2_kernel(const i16x4 *__restrict__ src_a,
                      const i16x4 *__restrict__ src_b,
                      const f32x4 *__restrict__ init,
                      unsigned long long *__restrict__ ticks,
                      f32x4 *__restrict__ out) {
  const int lane = static_cast<int>(threadIdx.x);
  const int slot = static_cast<int>(blockIdx.x) * kWaveSize + lane;
  const i16x4 a = src_a[lane];
  const i16x4 b = src_b[lane];
  f32x4 acc0 = init[slot * 8 + 0];
  f32x4 acc1 = init[slot * 8 + 1];
  DEVICE_PREPARE();
  const unsigned long long t0 = clock64();
  DEVICE_COMPILER_BARRIER();
  REP512(BF16_STEP2())
  DEVICE_COMPILER_BARRIER();
  const unsigned long long t1 = clock64();
  DEVICE_COMPILER_BARRIER();
  ticks[slot] = t1 - t0;
  out[slot * 8 + 0] = acc0;
  out[slot * 8 + 1] = acc1;
}

extern "C" __global__ __launch_bounds__(64, 1)
void bf16_acc4_kernel(const i16x4 *__restrict__ src_a,
                      const i16x4 *__restrict__ src_b,
                      const f32x4 *__restrict__ init,
                      unsigned long long *__restrict__ ticks,
                      f32x4 *__restrict__ out) {
  const int lane = static_cast<int>(threadIdx.x);
  const int slot = static_cast<int>(blockIdx.x) * kWaveSize + lane;
  const i16x4 a = src_a[lane];
  const i16x4 b = src_b[lane];
  f32x4 acc0 = init[slot * 8 + 0];
  f32x4 acc1 = init[slot * 8 + 1];
  f32x4 acc2 = init[slot * 8 + 2];
  f32x4 acc3 = init[slot * 8 + 3];
  DEVICE_PREPARE();
  const unsigned long long t0 = clock64();
  DEVICE_COMPILER_BARRIER();
  REP256(BF16_STEP4())
  DEVICE_COMPILER_BARRIER();
  const unsigned long long t1 = clock64();
  DEVICE_COMPILER_BARRIER();
  ticks[slot] = t1 - t0;
  out[slot * 8 + 0] = acc0;
  out[slot * 8 + 1] = acc1;
  out[slot * 8 + 2] = acc2;
  out[slot * 8 + 3] = acc3;
}

extern "C" __global__ __launch_bounds__(64, 1)
void bf16_acc8_kernel(const i16x4 *__restrict__ src_a,
                      const i16x4 *__restrict__ src_b,
                      const f32x4 *__restrict__ init,
                      unsigned long long *__restrict__ ticks,
                      f32x4 *__restrict__ out) {
  const int lane = static_cast<int>(threadIdx.x);
  const int slot = static_cast<int>(blockIdx.x) * kWaveSize + lane;
  const i16x4 a = src_a[lane];
  const i16x4 b = src_b[lane];
  f32x4 acc0 = init[slot * 8 + 0];
  f32x4 acc1 = init[slot * 8 + 1];
  f32x4 acc2 = init[slot * 8 + 2];
  f32x4 acc3 = init[slot * 8 + 3];
  f32x4 acc4 = init[slot * 8 + 4];
  f32x4 acc5 = init[slot * 8 + 5];
  f32x4 acc6 = init[slot * 8 + 6];
  f32x4 acc7 = init[slot * 8 + 7];
  DEVICE_PREPARE();
  const unsigned long long t0 = clock64();
  DEVICE_COMPILER_BARRIER();
  REP128(BF16_STEP8())
  DEVICE_COMPILER_BARRIER();
  const unsigned long long t1 = clock64();
  DEVICE_COMPILER_BARRIER();
  ticks[slot] = t1 - t0;
  out[slot * 8 + 0] = acc0;
  out[slot * 8 + 1] = acc1;
  out[slot * 8 + 2] = acc2;
  out[slot * 8 + 3] = acc3;
  out[slot * 8 + 4] = acc4;
  out[slot * 8 + 5] = acc5;
  out[slot * 8 + 6] = acc6;
  out[slot * 8 + 7] = acc7;
}

extern "C" __global__ __launch_bounds__(64, 1)
void i8_acc1_kernel(const long long *__restrict__ src_a,
                    const long long *__restrict__ src_b,
                    const i32x4 *__restrict__ init,
                    unsigned long long *__restrict__ ticks,
                    i32x4 *__restrict__ out) {
  const int lane = static_cast<int>(threadIdx.x);
  const int slot = static_cast<int>(blockIdx.x) * kWaveSize + lane;
  const long long a = src_a[lane];
  const long long b = src_b[lane];
  i32x4 acc0 = init[slot * 8];
  DEVICE_PREPARE();
  const unsigned long long t0 = clock64();
  DEVICE_COMPILER_BARRIER();
  REP1024(I8_STEP1())
  DEVICE_COMPILER_BARRIER();
  const unsigned long long t1 = clock64();
  DEVICE_COMPILER_BARRIER();
  ticks[slot] = t1 - t0;
  out[slot * 8] = acc0;
}

extern "C" __global__ __launch_bounds__(64, 1)
void i8_acc2_kernel(const long long *__restrict__ src_a,
                    const long long *__restrict__ src_b,
                    const i32x4 *__restrict__ init,
                    unsigned long long *__restrict__ ticks,
                    i32x4 *__restrict__ out) {
  const int lane = static_cast<int>(threadIdx.x);
  const int slot = static_cast<int>(blockIdx.x) * kWaveSize + lane;
  const long long a = src_a[lane];
  const long long b = src_b[lane];
  i32x4 acc0 = init[slot * 8 + 0];
  i32x4 acc1 = init[slot * 8 + 1];
  DEVICE_PREPARE();
  const unsigned long long t0 = clock64();
  DEVICE_COMPILER_BARRIER();
  REP512(I8_STEP2())
  DEVICE_COMPILER_BARRIER();
  const unsigned long long t1 = clock64();
  DEVICE_COMPILER_BARRIER();
  ticks[slot] = t1 - t0;
  out[slot * 8 + 0] = acc0;
  out[slot * 8 + 1] = acc1;
}

extern "C" __global__ __launch_bounds__(64, 1)
void i8_acc4_kernel(const long long *__restrict__ src_a,
                    const long long *__restrict__ src_b,
                    const i32x4 *__restrict__ init,
                    unsigned long long *__restrict__ ticks,
                    i32x4 *__restrict__ out) {
  const int lane = static_cast<int>(threadIdx.x);
  const int slot = static_cast<int>(blockIdx.x) * kWaveSize + lane;
  const long long a = src_a[lane];
  const long long b = src_b[lane];
  i32x4 acc0 = init[slot * 8 + 0];
  i32x4 acc1 = init[slot * 8 + 1];
  i32x4 acc2 = init[slot * 8 + 2];
  i32x4 acc3 = init[slot * 8 + 3];
  DEVICE_PREPARE();
  const unsigned long long t0 = clock64();
  DEVICE_COMPILER_BARRIER();
  REP256(I8_STEP4())
  DEVICE_COMPILER_BARRIER();
  const unsigned long long t1 = clock64();
  DEVICE_COMPILER_BARRIER();
  ticks[slot] = t1 - t0;
  out[slot * 8 + 0] = acc0;
  out[slot * 8 + 1] = acc1;
  out[slot * 8 + 2] = acc2;
  out[slot * 8 + 3] = acc3;
}

extern "C" __global__ __launch_bounds__(64, 1)
void i8_acc8_kernel(const long long *__restrict__ src_a,
                    const long long *__restrict__ src_b,
                    const i32x4 *__restrict__ init,
                    unsigned long long *__restrict__ ticks,
                    i32x4 *__restrict__ out) {
  const int lane = static_cast<int>(threadIdx.x);
  const int slot = static_cast<int>(blockIdx.x) * kWaveSize + lane;
  const long long a = src_a[lane];
  const long long b = src_b[lane];
  i32x4 acc0 = init[slot * 8 + 0];
  i32x4 acc1 = init[slot * 8 + 1];
  i32x4 acc2 = init[slot * 8 + 2];
  i32x4 acc3 = init[slot * 8 + 3];
  i32x4 acc4 = init[slot * 8 + 4];
  i32x4 acc5 = init[slot * 8 + 5];
  i32x4 acc6 = init[slot * 8 + 6];
  i32x4 acc7 = init[slot * 8 + 7];
  DEVICE_PREPARE();
  const unsigned long long t0 = clock64();
  DEVICE_COMPILER_BARRIER();
  REP128(I8_STEP8())
  DEVICE_COMPILER_BARRIER();
  const unsigned long long t1 = clock64();
  DEVICE_COMPILER_BARRIER();
  ticks[slot] = t1 - t0;
  out[slot * 8 + 0] = acc0;
  out[slot * 8 + 1] = acc1;
  out[slot * 8 + 2] = acc2;
  out[slot * 8 + 3] = acc3;
  out[slot * 8 + 4] = acc4;
  out[slot * 8 + 5] = acc5;
  out[slot * 8 + 6] = acc6;
  out[slot * 8 + 7] = acc7;
}

struct Buffers {
  i16x4 *a16 = nullptr;
  i16x4 *b16 = nullptr;
  long long *a8 = nullptr;
  long long *b8 = nullptr;
  f32x4 *f_init = nullptr;
  i32x4 *i_init = nullptr;
  f32x4 *f_out = nullptr;
  i32x4 *i_out = nullptr;
  unsigned long long *ticks = nullptr;
};

static void launch(const std::string &config, const Buffers &p, int blocks) {
  const dim3 grid(blocks);
  const dim3 block(kWaveSize);
  if (config == "baseline") {
    hipLaunchKernelGGL(baseline_kernel, grid, block, 0, 0, p.f_init, p.ticks,
                       p.f_out);
  } else if (config == "bf16_acc1") {
    hipLaunchKernelGGL(bf16_acc1_kernel, grid, block, 0, 0, p.a16, p.b16,
                       p.f_init, p.ticks, p.f_out);
  } else if (config == "bf16_acc2") {
    hipLaunchKernelGGL(bf16_acc2_kernel, grid, block, 0, 0, p.a16, p.b16,
                       p.f_init, p.ticks, p.f_out);
  } else if (config == "bf16_acc4") {
    hipLaunchKernelGGL(bf16_acc4_kernel, grid, block, 0, 0, p.a16, p.b16,
                       p.f_init, p.ticks, p.f_out);
  } else if (config == "bf16_acc8") {
    hipLaunchKernelGGL(bf16_acc8_kernel, grid, block, 0, 0, p.a16, p.b16,
                       p.f_init, p.ticks, p.f_out);
  } else if (config == "i8_acc1") {
    hipLaunchKernelGGL(i8_acc1_kernel, grid, block, 0, 0, p.a8, p.b8, p.i_init,
                       p.ticks, p.i_out);
  } else if (config == "i8_acc2") {
    hipLaunchKernelGGL(i8_acc2_kernel, grid, block, 0, 0, p.a8, p.b8, p.i_init,
                       p.ticks, p.i_out);
  } else if (config == "i8_acc4") {
    hipLaunchKernelGGL(i8_acc4_kernel, grid, block, 0, 0, p.a8, p.b8, p.i_init,
                       p.ticks, p.i_out);
  } else if (config == "i8_acc8") {
    hipLaunchKernelGGL(i8_acc8_kernel, grid, block, 0, 0, p.a8, p.b8, p.i_init,
                       p.ticks, p.i_out);
  } else {
    std::fprintf(stderr, "unknown config: %s\n", config.c_str());
    std::exit(2);
  }
  HIP_CHECK(hipGetLastError());
}

static std::uint64_t pack_i8(int lane, int salt) {
  std::uint64_t packed = 0;
  for (int i = 0; i < 8; ++i) {
    const std::int8_t value = static_cast<std::int8_t>(((lane + i + salt) % 7) - 3);
    packed |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(value)) << (8 * i);
  }
  return packed;
}

int main(int argc, char **argv) {
  bw::start();

  if (argc == 1) { static char a0[]="mmac",a1[]="bf16_acc1",a2[]="30",a3[]="5"; static char* defaults[]={a0,a1,a2,a3,nullptr}; argc=4; argv=defaults; }
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s CONFIG REPEATS WARMUPS\n", argv[0]);
    return 2;
  }
  const std::string config = argv[1];
  const int repeats = std::atoi(argv[2]);
  const int warmups = std::atoi(argv[3]);
  if (repeats < 1 || warmups < 0) {
    std::fprintf(stderr, "invalid repeats/warmups\n");
    return 2;
  }

  int device = 0;
  hipDeviceProp_t props{};
  HIP_CHECK(hipSetDevice(device));
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::fprintf(stderr,
               "config=%s device=%s gcnArchName=%s warpSize=%d repeats=%d "
               "warmups=%d mmac_count=%d blocks=%d\n",
               config.c_str(), props.name, props.gcnArchName, props.warpSize,
               repeats, warmups, kMmacCount, props.multiProcessorCount);

  const int blocks = props.multiProcessorCount;
  const std::size_t kSourceLanes = kWaveSize;
  const std::size_t kSlots = static_cast<std::size_t>(blocks) * kWaveSize;
  const std::size_t kAccValues = kSlots * 8;
  std::vector<i16x4> h_a16(kSourceLanes);
  std::vector<i16x4> h_b16(kSourceLanes);
  std::vector<long long> h_a8(kSourceLanes);
  std::vector<long long> h_b8(kSourceLanes);
  std::vector<f32x4> h_finit(kAccValues);
  std::vector<i32x4> h_iinit(kAccValues);
  for (int lane = 0; lane < kWaveSize; ++lane) {
    h_a16[lane] = i16x4{static_cast<short>(0x3f80), static_cast<short>(0x3f00),
                         static_cast<short>(0xbf00), static_cast<short>(0x3e80)};
    h_b16[lane] = i16x4{static_cast<short>(0x3f80), static_cast<short>(0xbf80),
                         static_cast<short>(0x3f00), static_cast<short>(0x3e00)};
    h_a8[lane] = static_cast<long long>(pack_i8(lane, 1));
    h_b8[lane] = static_cast<long long>(pack_i8(lane, 3));
  }
  for (std::size_t slot = 0; slot < kSlots; ++slot) {
    for (int acc = 0; acc < 8; ++acc) {
      const std::size_t index = slot * 8 + acc;
      h_finit[index] = f32x4{0.001f * (index + 1), 0.002f * (index + 1),
                              0.003f * (index + 1), 0.004f * (index + 1)};
      const int value = static_cast<int>(index);
      h_iinit[index] = i32x4{value + 1, value + 2, value + 3, value + 4};
    }
  }

  Buffers p;
  HIP_CHECK(hipMalloc(&p.a16, kSourceLanes * sizeof(i16x4)));
  HIP_CHECK(hipMalloc(&p.b16, kSourceLanes * sizeof(i16x4)));
  HIP_CHECK(hipMalloc(&p.a8, kSourceLanes * sizeof(long long)));
  HIP_CHECK(hipMalloc(&p.b8, kSourceLanes * sizeof(long long)));
  HIP_CHECK(hipMalloc(&p.f_init, kAccValues * sizeof(f32x4)));
  HIP_CHECK(hipMalloc(&p.i_init, kAccValues * sizeof(i32x4)));
  HIP_CHECK(hipMalloc(&p.f_out, kAccValues * sizeof(f32x4)));
  HIP_CHECK(hipMalloc(&p.i_out, kAccValues * sizeof(i32x4)));
  HIP_CHECK(hipMalloc(&p.ticks, kSlots * sizeof(unsigned long long)));
  HIP_CHECK(hipMemcpy(p.a16, h_a16.data(), kSourceLanes * sizeof(i16x4),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(p.b16, h_b16.data(), kSourceLanes * sizeof(i16x4),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(p.a8, h_a8.data(), kSourceLanes * sizeof(long long),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(p.b8, h_b8.data(), kSourceLanes * sizeof(long long),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(p.f_init, h_finit.data(), kAccValues * sizeof(f32x4),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(p.i_init, h_iinit.data(), kAccValues * sizeof(i32x4),
                      hipMemcpyHostToDevice));

  for (int i = 0; i < warmups; ++i) {
    launch(config, p, blocks);
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  std::vector<unsigned long long> h_ticks(kSlots);
  // Archived pre/post baseline, sampled in-process; no external analysis.
  auto sample_baseline = [&]() {
    for (int run=0;run<repeats;++run) {
      launch("baseline",p,blocks); HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(h_ticks.data(),p.ticks,kSlots*sizeof(unsigned long long),hipMemcpyDeviceToHost));
      auto ordered=h_ticks; auto middle=ordered.begin()+ordered.size()/2;
      std::nth_element(ordered.begin(),middle,ordered.end());
      bw::line("config,run,raw_ticks,lane_min,lane_max,event_ms");
      bw::emit("baseline,%d,%llu,0,0,0\n",run,*middle);
    }
  };
  sample_baseline();

  bw::emit("config,run,raw_ticks,lane_min,lane_max,event_ms\n");
  for (int run = 0; run < repeats; ++run) {
    HIP_CHECK(hipEventRecord(start));
    launch(config, p, blocks);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipMemcpy(h_ticks.data(), p.ticks,
                        kSlots * sizeof(unsigned long long),
                        hipMemcpyDeviceToHost));
    const auto minmax = std::minmax_element(h_ticks.begin(), h_ticks.end());
    std::vector<unsigned long long> ordered_ticks = h_ticks;
    const auto middle = ordered_ticks.begin() + ordered_ticks.size() / 2;
    std::nth_element(ordered_ticks.begin(), middle, ordered_ticks.end());
    bw::emit("%s,%d,%llu,%llu,%llu,%.9g\n", config.c_str(), run,
                *middle, *minmax.first, *minmax.second, elapsed_ms);
  }

  double checksum = 0.0;
  if (config == "baseline" || config.rfind("bf16_", 0) == 0) {
    std::vector<f32x4> result(kAccValues);
    HIP_CHECK(hipMemcpy(result.data(), p.f_out, kAccValues * sizeof(f32x4),
                        hipMemcpyDeviceToHost));
    for (const auto &value : result) {
      checksum += static_cast<double>(value[0]) + value[1] + value[2] + value[3];
    }
  } else {
    std::vector<i32x4> result(kAccValues);
    HIP_CHECK(hipMemcpy(result.data(), p.i_out, kAccValues * sizeof(i32x4),
                        hipMemcpyDeviceToHost));
    for (const auto &value : result) {
      checksum += static_cast<double>(value[0]) + value[1] + value[2] + value[3];
    }
  }
  std::fprintf(stderr, "checksum=%.17g\n", checksum);

  sample_baseline();
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(p.a16));
  HIP_CHECK(hipFree(p.b16));
  HIP_CHECK(hipFree(p.a8));
  HIP_CHECK(hipFree(p.b8));
  HIP_CHECK(hipFree(p.f_init));
  HIP_CHECK(hipFree(p.i_init));
  HIP_CHECK(hipFree(p.f_out));
  HIP_CHECK(hipFree(p.i_out));
  HIP_CHECK(hipFree(p.ticks));
  return bw::finish("mma");
}
