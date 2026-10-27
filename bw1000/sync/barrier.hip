/* BW1000 single-file HIP program: sync/barrier
 * Compile: hipcc -O2 -std=c++17 --offload-arch=gfx936 barrier.hip -o bench
 * Run: ./bench --mode barrier_base --waves 1 --inner-reps 64 --repeats 30
 * Minimal run (NOT a timing reproduction): ./bench --mode barrier_base --waves 1 --inner-reps 8 --repeats 1
 * No local headers, private libraries, scripts, data files or other source files required.
 * Only public HIP/DTK, its driver/runtime, and the C++ standard library are required.
 * Prints in-process summaries in timer tick and calibrated ns. BW1000_RAW=1 also prints samples.
 * Original measurement core: sync_wait/src/barrier_bench.cpp; SHA256=f39eac39a8aa81acb3aa3cf0bfdecb6ef5c15a7698d6b36409001d85d64a6ab9
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
    if(family=="barrier") {
      t="s_barrier";if(s=="barrier_base"||s=="barrier_control")batches[{t,conditions(r,{"variant","inner_reps"}),s}][int(num(r,"inner_reps"))].push_back(num(r,"t1")-num(r,"t0"));
      else {put(s=="barrier_skew"?"barrier_span":"clock_or_delay_interval","observed bracket",num(r,s=="barrier_skew"?"t2":"t1")-num(r,s=="barrier_skew"?"t1":"t0"));}continue;
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
// Common definitions intentionally inlined.



#define HIP_CHECK(call) do {                                                   \
  hipError_t e_ = (call);                                                      \
  if (e_ != hipSuccess) {                                                      \
    std::fprintf(stderr, "HIP error %s:%d: %s -> %s\n", __FILE__, __LINE__,  \
                 #call, hipGetErrorString(e_));                                \
    std::exit(2);                                                              \
  }                                                                            \
} while (0)

static constexpr uint32_t kWaveSize = 64;
static constexpr uint32_t kMaxWaves = 8;

struct TimingRecord {
  uint64_t t[8];
  uint32_t value;
  uint32_t aux;
};

inline std::string get_arg(int argc, char** argv, const char* key,
                           const char* fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (!std::strcmp(argv[i], key)) return argv[i + 1];
  }
  return fallback;
}

inline int get_int(int argc, char** argv, const char* key, int fallback) {
  const std::string s = get_arg(argc, argv, key, "");
  return s.empty() ? fallback : std::atoi(s.c_str());
}

__device__ __forceinline__ uint32_t dependent_delay(uint32_t x,
                                                     uint32_t iterations) {
#pragma clang loop unroll(disable)
  for (uint32_t i = 0; i < iterations; ++i) {
    asm volatile(
        "v_add_u32 %0, 0x9e3779b9, %0\n\t"
        "v_xor_b32 %0, 0x85ebca6b, %0\n\t"
        "v_add_u32 %0, 0xc2b2ae35, %0\n\t"
        "v_xor_b32 %0, 0x27d4eb2f, %0"
        : "+v"(x));
  }
  return x;
}

__device__ __forceinline__ void raw_barrier() {
  asm volatile("s_barrier" ::: "memory");
}

template<int K>
__device__ __forceinline__ void wait_lgkm() {
  static_assert(K >= 0 && K <= 15, "gfx936 lgkm threshold must fit 4 bits");
  if constexpr (K == 0) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  else if constexpr (K == 1) asm volatile("s_waitcnt lgkmcnt(1)" ::: "memory");
  else if constexpr (K == 2) asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
  else if constexpr (K == 4) asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
  else if constexpr (K == 8) asm volatile("s_waitcnt lgkmcnt(8)" ::: "memory");
  else if constexpr (K == 15) asm volatile("s_waitcnt lgkmcnt(15)" ::: "memory");
}

template<int K>
__device__ __forceinline__ void wait_vmcnt() {
  if constexpr (K == 0) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  else if constexpr (K == 1) asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
  else if constexpr (K == 2) asm volatile("s_waitcnt vmcnt(2)" ::: "memory");
  else if constexpr (K == 4) asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
  else if constexpr (K == 8) asm volatile("s_waitcnt vmcnt(8)" ::: "memory");
  else if constexpr (K == 15) asm volatile("s_waitcnt vmcnt(15)" ::: "memory");
  else if constexpr (K == 31) asm volatile("s_waitcnt vmcnt(31)" ::: "memory");
  else if constexpr (K == 32) asm volatile("s_waitcnt vmcnt(32)" ::: "memory");
}

__device__ __forceinline__ void wait_both() {
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
}

__device__ __forceinline__ void wait_all_zero() {
  asm volatile("s_waitcnt 0" ::: "memory");
}

inline void print_record_header() {
  bw::line("mode,repeat,wave,waves,variant,op,n,threshold,age,conflict,state,"
            "pattern,width,payload,skew,inner_reps,event_us,"
            "t0,t1,t2,t3,t4,t5,t6,t7,value,aux");
}

inline void print_record(const char* mode, int repeat, int wave, int waves,
                         const std::string& variant, const std::string& op,
                         int n, int threshold, int age, int conflict,
                         const std::string& state, const std::string& pattern,
                         int width, int payload, int skew, int inner_reps,
                         float event_us, const TimingRecord& r) {
  bw::emit("%s,%d,%d,%d,%s,%s,%d,%d,%d,%d,%s,%s,%d,%d,%d,%d,%.3f",
              mode, repeat, wave, waves, variant.c_str(), op.c_str(), n,
              threshold, age, conflict, state.c_str(), pattern.c_str(), width,
              payload, skew, inner_reps, event_us);
  for (int i = 0; i < 8; ++i) {
    bw::emit(",%llu", static_cast<unsigned long long>(r.t[i]));
  }
  bw::emit(",%u,%u\n", r.value, r.aux);
}




extern "C" __global__ void clock_overhead_kernel(TimingRecord* out) {
  const uint32_t lane = threadIdx.x & 63u;
  const uint64_t t0 = clock64();
  const uint64_t t1 = clock64();
  const uint64_t t2 = clock64();
  if (lane == 0) {
    TimingRecord r{};
    r.t[0] = t0;
    r.t[1] = t1;
    r.t[2] = t2;
    r.value = static_cast<uint32_t>(t2 - t0);
    out[0] = r;
  }
}

extern "C" __global__ void delay_kernel(uint32_t iterations,
                                         TimingRecord* out) {
  const uint32_t lane = threadIdx.x & 63u;
  uint32_t x = lane + 1u;
  wait_lgkm<0>();
  const uint64_t t0 = clock64();
  x = dependent_delay(x, iterations);
  const uint64_t t1 = clock64();
  if (lane == 0) {
    TimingRecord r{};
    r.t[0] = t0;
    r.t[1] = t1;
    r.value = x;
    out[0] = r;
  }
}

template<bool WithBarrier>
__global__ void barrier_loop_kernel(uint32_t repetitions, TimingRecord* out) {
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t wave = threadIdx.x >> 6;
  uint32_t x = lane + 1u + wave;
  wait_lgkm<0>();
  raw_barrier();
  const uint64_t t0 = clock64();
#pragma clang loop unroll(disable)
  for (uint32_t i = 0; i < repetitions; ++i) {
    asm volatile("v_add_u32 %0, 1, %0" : "+v"(x));
    if constexpr (WithBarrier) raw_barrier();
  }
  const uint64_t t1 = clock64();
  if (lane == 0) {
    TimingRecord r{};
    r.t[0] = t0;
    r.t[1] = t1;
    r.value = x;
    out[wave] = r;
  }
}

extern "C" __global__ void barrier_skew_kernel(uint32_t base_delay,
                                                TimingRecord* out) {
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t wave = threadIdx.x >> 6;
  uint32_t x = lane + 1u + wave;
  wait_lgkm<0>();
  raw_barrier();
  const uint64_t t0 = clock64();
  x = dependent_delay(x, base_delay * wave);
  const uint64_t t_arrive = clock64();
  raw_barrier();
  const uint64_t t_exit = clock64();
  if (lane == 0) {
    TimingRecord r{};
    r.t[0] = t0;
    r.t[1] = t_arrive;
    r.t[2] = t_exit;
    r.value = x;
    out[wave] = r;
  }
}

static void launch_mode(const std::string& mode, int waves, int inner_reps,
                        int delay, TimingRecord* d_out) {
  const dim3 block(waves * kWaveSize);
  if (mode == "clock") {
    hipLaunchKernelGGL(clock_overhead_kernel, dim3(1), dim3(64), 0, 0, d_out);
  } else if (mode == "delay" || mode == "clock_cal") {
    hipLaunchKernelGGL(delay_kernel, dim3(1), dim3(64), 0, 0,
                       static_cast<uint32_t>(delay), d_out);
  } else if (mode == "barrier_base") {
    hipLaunchKernelGGL((barrier_loop_kernel<true>), dim3(1), block, 0, 0,
                       static_cast<uint32_t>(inner_reps), d_out);
  } else if (mode == "barrier_control") {
    hipLaunchKernelGGL((barrier_loop_kernel<false>), dim3(1), block, 0, 0,
                       static_cast<uint32_t>(inner_reps), d_out);
  } else if (mode == "barrier_skew") {
    hipLaunchKernelGGL(barrier_skew_kernel, dim3(1), block, 0, 0,
                       static_cast<uint32_t>(delay), d_out);
  } else {
    std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    std::exit(4);
  }
  HIP_CHECK(hipGetLastError());
}

int main(int argc, char** argv) {
  bw::start();

  const std::string mode = get_arg(argc, argv, "--mode", "barrier_base");
  const int waves = get_int(argc, argv, "--waves", 1);
  const int repeats = get_int(argc, argv, "--repeats", 30);
  const int inner_reps = get_int(argc, argv, "--inner-reps", 64);
  const int delay = get_int(argc, argv, "--delay", 0);
  if (waves < 1 || waves > static_cast<int>(kMaxWaves)) return 5;

  TimingRecord* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, kMaxWaves * sizeof(TimingRecord)));
  std::vector<TimingRecord> h_out(kMaxWaves);
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  for (int warm = 0; warm < 3; ++warm) {
    launch_mode(mode, waves, inner_reps, delay, d_out);
  }
  HIP_CHECK(hipDeviceSynchronize());
  print_record_header();
  for (int rep = 0; rep < repeats; ++rep) {
    HIP_CHECK(hipMemset(d_out, 0, kMaxWaves * sizeof(TimingRecord)));
    HIP_CHECK(hipEventRecord(start));
    launch_mode(mode, waves, inner_reps, delay, d_out);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipMemcpy(h_out.data(), d_out,
                        kMaxWaves * sizeof(TimingRecord),
                        hipMemcpyDeviceToHost));
    const int rows = (mode == "clock" || mode == "delay" || mode == "clock_cal")
                         ? 1 : waves;
    for (int wave = 0; wave < rows; ++wave) {
      print_record(mode.c_str(), rep, wave, rows,
                   mode == "barrier_control" ? "control" : "measured",
                   "none", 0, 0, delay, 0, "none", "none", 0, 0, delay,
                   inner_reps, ms * 1000.0f, h_out[wave]);
    }
  }
  if(mode=="barrier_base") {
    for(int rep=0;rep<repeats;++rep){
      launch_mode("barrier_control",waves,inner_reps,delay,d_out);HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(h_out.data(),d_out,kMaxWaves*sizeof(TimingRecord),hipMemcpyDeviceToHost));
      for(int wave=0;wave<waves;++wave)print_record("barrier_control",rep,wave,waves,"control","none",0,0,delay,0,"none","none",0,0,delay,inner_reps,0,h_out[wave]);
    }
  }
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipFree(d_out));
  return bw::finish("barrier");
}
