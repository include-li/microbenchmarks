/* BW1000 single-file HIP program: sync/waitcnt
 * Compile: hipcc -O2 -std=c++17 --offload-arch=gfx936 waitcnt.hip -o bench
 * Run: ./bench --mode lgkm --op read --n 8 --threshold 0 --repeats 30
 * Minimal run (NOT a timing reproduction): ./bench --mode lgkm --op read --n 1 --threshold 0 --repeats 1
 * No local headers, private libraries, scripts, data files or other source files required.
 * Only public HIP/DTK, its driver/runtime, and the C++ standard library are required.
 * Prints in-process summaries in timer tick and calibrated ns. BW1000_RAW=1 also prints samples.
 * Original measurement core: sync_wait/src/waitcnt_bench.cpp; SHA256=3bf3939ca681fda0449b239fcf559d89dfda2c5e158c3aaada893a9691774126
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
    if(family=="waitcnt") {
      t="s_waitcnt "+s;const char*names[]={"issue","age","selected_wait_observed","remaining_wait_observed","consume"};const int a[]={0,1,2,3,4},b[]={1,2,3,4,5};
      for(int i=0;i<5;++i)put(names[i],"observed interval, includes timer effects",num(r,"t"+std::to_string(b[i]))-num(r,"t"+std::to_string(a[i])));
      put("completion","observed request+wait chain",num(r,"t4")-num(r,"t0"));continue;
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




using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
static constexpr uint32_t kLdsWords = 16384;

template<int Width> struct DsType;
template<> struct DsType<2> { using type = uint32_t; };
template<> struct DsType<4> { using type = uint32_t; };
template<> struct DsType<16> { using type = u32x4; };
template<int Width> using DsValue = typename DsType<Width>::type;

template<int Width>
__device__ __forceinline__ DsValue<Width> make_ds_value(uint32_t x) {
  if constexpr (Width == 16) return u32x4{x, x + 1u, x + 2u, x + 3u};
  else return x;
}

template<int Width>
__device__ __forceinline__ uint32_t sum_ds_value(DsValue<Width> x) {
  if constexpr (Width == 16) return x[0] + x[1] + x[2] + x[3];
  else return x;
}

template<int Width>
__device__ __forceinline__ DsValue<Width> ds_read(uint32_t addr) {
  DsValue<Width> v;
  if constexpr (Width == 2) {
    asm volatile("ds_read_u16 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  } else if constexpr (Width == 4) {
    asm volatile("ds_read_b32 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  } else {
    asm volatile("ds_read_b128 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  }
  return v;
}

template<int Width>
__device__ __forceinline__ void ds_write(uint32_t addr, DsValue<Width> v) {
  if constexpr (Width == 2) {
    asm volatile("ds_write_b16 %0, %1" :: "v"(addr), "v"(v) : "memory");
  } else if constexpr (Width == 4) {
    asm volatile("ds_write_b32 %0, %1" :: "v"(addr), "v"(v) : "memory");
  } else {
    asm volatile("ds_write_b128 %0, %1" :: "v"(addr), "v"(v) : "memory");
  }
}

template<int Width>
__device__ __forceinline__ uint32_t conflict_word(uint32_t lane,
                                                  uint32_t degree) {
  constexpr uint32_t words = Width == 16 ? 4u : 1u;
  uint32_t slots = 32u / (words * degree);
  if (slots == 0u) slots = 1u;
  const uint32_t slot = lane % slots;
  const uint32_t rank = lane / slots;
  return slot * words + 32u * (rank + 1u);
}

template<int I, int N, int Width, bool IsWrite>
__device__ __forceinline__ void issue_ds(uint32_t base_addr,
                                         DsValue<Width>* vals) {
  if constexpr (I < N) {
    const uint32_t addr = base_addr + I * 2048u;
    if constexpr (IsWrite) ds_write<Width>(addr, vals[I]);
    else vals[I] = ds_read<Width>(addr);
    issue_ds<I + 1, N, Width, IsWrite>(base_addr, vals);
  }
}

template<int N, int K, int Width, bool IsWrite, bool HasAge>
__global__ void lgkm_kernel(uint32_t degree, uint32_t age,
                            TimingRecord* out) {
  __shared__ uint32_t lds[kLdsWords];
  const uint32_t lane = threadIdx.x & 63u;
  for (uint32_t i = lane; i < kLdsWords; i += 64u) {
    lds[i] = 0x10000u + i;
  }
  wait_lgkm<0>();
  raw_barrier();
  DsValue<Width> vals[N > 0 ? N : 1];
#pragma unroll
  for (int i = 0; i < (N > 0 ? N : 1); ++i) {
    vals[i] = make_ds_value<Width>(lane + static_cast<uint32_t>(i) + 1u);
  }
  uint32_t base_addr = conflict_word<Width>(lane, degree) * 4u;
  asm volatile("" : "+v"(base_addr));
  uint32_t delay_sink = lane + 1u;
  const uint64_t t0 = clock64();
  issue_ds<0, N, Width, IsWrite>(base_addr, vals);
  const uint64_t t1 = clock64();
  uint64_t t2 = t1;
  if constexpr (HasAge) {
    delay_sink = dependent_delay(delay_sink, age);
    t2 = clock64();
  }
  wait_lgkm<K>();
  const uint64_t t3 = clock64();
  wait_lgkm<0>();
  const uint64_t t4 = clock64();
  uint32_t sink = delay_sink;
  if constexpr (!IsWrite) {
#pragma unroll
    for (int i = 0; i < N; ++i) sink += sum_ds_value<Width>(vals[i]);
  } else if constexpr (N > 0) {
    DsValue<Width> check = ds_read<Width>(base_addr);
    wait_lgkm<0>();
    sink += sum_ds_value<Width>(check);
  }
  const uint64_t t5 = clock64();
  if (lane == 0) {
    TimingRecord r{};
    r.t[0] = t0; r.t[1] = t1; r.t[2] = t2; r.t[3] = t3;
    r.t[4] = t4; r.t[5] = t5;
    r.value = sink;
    r.aux = N;
    out[0] = r;
  }
}

template<int I, int N>
__device__ __forceinline__ void make_global_addresses(const uint8_t* base,
                                                       uint32_t lane,
                                                       uint64_t* addrs) {
  if constexpr (I < N) {
    addrs[I] = reinterpret_cast<uint64_t>(
        base + static_cast<uint64_t>(I) * 4096u + lane * 4u);
    make_global_addresses<I + 1, N>(base, lane, addrs);
  }
}

template<int I, int N>
__device__ __forceinline__ void issue_global_loads(const uint64_t* addrs,
                                                    uint32_t* vals) {
  if constexpr (I < N) {
    asm volatile("global_load_dword %0, %1, off glc"
                 : "=v"(vals[I]) : "v"(addrs[I]) : "memory");
    issue_global_loads<I + 1, N>(addrs, vals);
  }
}

template<int N, int K, bool HasAge>
__global__ void vmem_kernel(const uint8_t* base, uint32_t age,
                            TimingRecord* out) {
  const uint32_t lane = threadIdx.x & 63u;
  uint32_t vals[N > 0 ? N : 1];
  uint64_t addrs[N > 0 ? N : 1];
#pragma unroll
  for (int i = 0; i < (N > 0 ? N : 1); ++i) vals[i] = 0;
  make_global_addresses<0, N>(base, lane, addrs);
  if constexpr (N > 0) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+v"(addrs[i]));
  }
  uint32_t delay_sink = lane + 1u;
  wait_lgkm<0>();
  const uint64_t t0 = clock64();
  issue_global_loads<0, N>(addrs, vals);
  const uint64_t t1 = clock64();
  uint64_t t2 = t1;
  if constexpr (HasAge) {
    delay_sink = dependent_delay(delay_sink, age);
    t2 = clock64();
  }
  wait_vmcnt<K>();
  const uint64_t t3 = clock64();
  wait_vmcnt<0>();
  const uint64_t t4 = clock64();
  uint32_t sink = delay_sink;
#pragma unroll
  for (int i = 0; i < N; ++i) sink += vals[i];
  const uint64_t t5 = clock64();
  if (lane == 0) {
    TimingRecord r{};
    r.t[0] = t0; r.t[1] = t1; r.t[2] = t2; r.t[3] = t3;
    r.t[4] = t4; r.t[5] = t5;
    r.value = sink;
    r.aux = N;
    out[0] = r;
  }
}

enum MixedKind : int { kLgkmOnly = 0, kVmOnly = 1, kCombined = 2, kWaitZero = 3 };

template<int Kind, bool HasAge>
__global__ void mixed_kernel(const uint8_t* base, uint32_t age,
                             uint32_t degree, TimingRecord* out) {
  __shared__ uint32_t lds[kLdsWords];
  const uint32_t lane = threadIdx.x & 63u;
  for (uint32_t i = lane; i < kLdsWords; i += 64u) lds[i] = i + 3u;
  wait_lgkm<0>();
  raw_barrier();
  uint32_t gv[8]{};
  uint32_t lv[8]{};
  uint64_t gaddrs[8];
  make_global_addresses<0, 8>(base, lane, gaddrs);
  uint32_t laddrs[8];
  const uint32_t lbase = conflict_word<4>(lane, degree) * 4u;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    laddrs[i] = lbase + i * 2048u;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    asm volatile("" : "+v"(gaddrs[i]));
    asm volatile("" : "+v"(laddrs[i]));
  }
  uint32_t delay_sink = lane + 1u;
  const uint64_t t0 = clock64();
  issue_global_loads<0, 8>(gaddrs, gv);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    lv[i] = ds_read<4>(laddrs[i]);
  }
  const uint64_t t1 = clock64();
  uint64_t t2 = t1;
  if constexpr (HasAge) {
    delay_sink = dependent_delay(delay_sink, age);
    t2 = clock64();
  }
  if constexpr (Kind == kLgkmOnly) wait_lgkm<0>();
  else if constexpr (Kind == kVmOnly) wait_vmcnt<0>();
  else if constexpr (Kind == kCombined) wait_both();
  else wait_all_zero();
  const uint64_t t3 = clock64();
  wait_both();
  const uint64_t t4 = clock64();
  uint32_t sink = delay_sink;
#pragma unroll
  for (int i = 0; i < 8; ++i) sink += gv[i] + lv[i];
  const uint64_t t5 = clock64();
  if (lane == 0) {
    TimingRecord r{};
    r.t[0] = t0; r.t[1] = t1; r.t[2] = t2; r.t[3] = t3;
    r.t[4] = t4; r.t[5] = t5;
    r.value = sink;
    r.aux = Kind;
    out[0] = r;
  }
}

extern "C" __global__ void touch_memory_kernel(const uint8_t* base,
                                                uint64_t bytes,
                                                uint32_t* sink_out) {
  const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t threads = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  uint32_t sink = 0;
  for (uint64_t off = tid * 64u; off < bytes; off += threads * 64u) {
    uint32_t v;
    const uint64_t addr = reinterpret_cast<uint64_t>(base + off);
    asm volatile("global_load_dword %0, %1, off glc"
                 : "=v"(v) : "v"(addr) : "memory");
    sink ^= v;
  }
  wait_vmcnt<0>();
  if (tid == 0) sink_out[0] = sink;
}

template<int N, bool IsWrite, bool HasAge>
static void launch_lgkm_k(int k, int width, uint32_t conflict, uint32_t age,
                          TimingRecord* out) {
#define LAUNCH_LGKM(W, K) \
  hipLaunchKernelGGL((lgkm_kernel<N, K, W, IsWrite, HasAge>), dim3(1), dim3(64), 0, 0, conflict, age, out)
  if (width == 2) {
    if (k != 0 || (N != 1 && N != 4)) std::exit(10);
    LAUNCH_LGKM(2, 0);
  } else if (width == 16) {
    if (k != 0 || (N != 1 && N != 4)) std::exit(10);
    LAUNCH_LGKM(16, 0);
  } else {
    switch (k) {
      case 0: LAUNCH_LGKM(4, 0); break;
      case 1: LAUNCH_LGKM(4, 1); break;
      case 2: LAUNCH_LGKM(4, 2); break;
      case 4: LAUNCH_LGKM(4, 4); break;
      case 8: LAUNCH_LGKM(4, 8); break;
      case 15: LAUNCH_LGKM(4, 15); break;
      default: std::exit(11);
    }
  }
#undef LAUNCH_LGKM
}

template<bool IsWrite, bool HasAge>
static void launch_lgkm_age(int n, int k, int width, uint32_t conflict,
                            uint32_t age, TimingRecord* out) {
  switch (n) {
    case 0: launch_lgkm_k<0, IsWrite, HasAge>(k, width, conflict, age, out); break;
    case 1: launch_lgkm_k<1, IsWrite, HasAge>(k, width, conflict, age, out); break;
    case 2: launch_lgkm_k<2, IsWrite, HasAge>(k, width, conflict, age, out); break;
    case 4: launch_lgkm_k<4, IsWrite, HasAge>(k, width, conflict, age, out); break;
    case 8: launch_lgkm_k<8, IsWrite, HasAge>(k, width, conflict, age, out); break;
    case 16: launch_lgkm_k<16, IsWrite, HasAge>(k, width, conflict, age, out); break;
    default: std::exit(12);
  }
}

template<bool IsWrite>
static void launch_lgkm(int n, int k, int width, uint32_t conflict,
                        uint32_t age, TimingRecord* out) {
  if (age == 0) launch_lgkm_age<IsWrite, false>(n, k, width, conflict, age, out);
  else launch_lgkm_age<IsWrite, true>(n, k, width, conflict, age, out);
}

template<int N, bool HasAge>
static void launch_vmem_k(int k, const uint8_t* base, uint32_t age,
                          TimingRecord* out) {
#define LAUNCH_VMEM(K) \
  hipLaunchKernelGGL((vmem_kernel<N, K, HasAge>), dim3(1), dim3(64), 0, 0, base, age, out)
  switch (k) {
    case 0: LAUNCH_VMEM(0); break;
    case 1: LAUNCH_VMEM(1); break;
    case 2: LAUNCH_VMEM(2); break;
    case 4: LAUNCH_VMEM(4); break;
    case 8: LAUNCH_VMEM(8); break;
    case 15: LAUNCH_VMEM(15); break;
    case 31: LAUNCH_VMEM(31); break;
    case 32: LAUNCH_VMEM(32); break;
    default: std::exit(13);
  }
#undef LAUNCH_VMEM
}

template<bool HasAge>
static void launch_vmem_age(int n, int k, const uint8_t* base, uint32_t age,
                            TimingRecord* out) {
  switch (n) {
    case 0: launch_vmem_k<0, HasAge>(k, base, age, out); break;
    case 1: launch_vmem_k<1, HasAge>(k, base, age, out); break;
    case 2: launch_vmem_k<2, HasAge>(k, base, age, out); break;
    case 4: launch_vmem_k<4, HasAge>(k, base, age, out); break;
    case 8: launch_vmem_k<8, HasAge>(k, base, age, out); break;
    case 16: launch_vmem_k<16, HasAge>(k, base, age, out); break;
    default: std::exit(14);
  }
}

static void launch_vmem(int n, int k, const uint8_t* base, uint32_t age,
                        TimingRecord* out) {
  if (age == 0) launch_vmem_age<false>(n, k, base, age, out);
  else launch_vmem_age<true>(n, k, base, age, out);
}

template<bool HasAge>
static void launch_mixed_age(int kind, const uint8_t* base, uint32_t age,
                             uint32_t conflict, TimingRecord* out) {
  switch (kind) {
    case kLgkmOnly:
      hipLaunchKernelGGL((mixed_kernel<kLgkmOnly, HasAge>), dim3(1), dim3(64), 0, 0,
                         base, age, conflict, out); break;
    case kVmOnly:
      hipLaunchKernelGGL((mixed_kernel<kVmOnly, HasAge>), dim3(1), dim3(64), 0, 0,
                         base, age, conflict, out); break;
    case kCombined:
      hipLaunchKernelGGL((mixed_kernel<kCombined, HasAge>), dim3(1), dim3(64), 0, 0,
                         base, age, conflict, out); break;
    case kWaitZero:
      hipLaunchKernelGGL((mixed_kernel<kWaitZero, HasAge>), dim3(1), dim3(64), 0, 0,
                         base, age, conflict, out); break;
    default: std::exit(15);
  }
}

static void launch_mixed(int kind, const uint8_t* base, uint32_t age,
                         uint32_t conflict, TimingRecord* out) {
  if (age == 0) launch_mixed_age<false>(kind, base, age, conflict, out);
  else launch_mixed_age<true>(kind, base, age, conflict, out);
}

static const char* mixed_name(int kind) {
  if (kind == kLgkmOnly) return "lgkm_only";
  if (kind == kVmOnly) return "vm_only";
  if (kind == kCombined) return "combined";
  return "waitcnt_0";
}

int main(int argc, char** argv) {
  bw::start();

  const std::string mode = get_arg(argc, argv, "--mode", "lgkm");
  const std::string op = get_arg(argc, argv, "--op", "read");
  const std::string state = get_arg(argc, argv, "--state", "L2-hot");
  const int n = get_int(argc, argv, "--n", 0);
  const int threshold = get_int(argc, argv, "--threshold", 0);
  const int age = get_int(argc, argv, "--age", 0);
  const int conflict = get_int(argc, argv, "--conflict", 1);
  const int width = get_int(argc, argv, "--width", 4);
  const int kind = get_int(argc, argv, "--kind", 2);
  const int repeats = get_int(argc, argv, "--repeats", 30);

  TimingRecord* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, sizeof(TimingRecord)));
  uint8_t* d_target = nullptr;
  uint8_t* d_pollution = nullptr;
  uint32_t* d_touch_sink = nullptr;
  static constexpr uint64_t target_bytes = 256ull << 20;
  static constexpr uint64_t pollution_bytes = 32ull << 20;
  if (mode != "lgkm") {
    HIP_CHECK(hipMalloc(&d_target, target_bytes));
    HIP_CHECK(hipMalloc(&d_pollution, pollution_bytes));
    HIP_CHECK(hipMalloc(&d_touch_sink, sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_target, 1, target_bytes));
    HIP_CHECK(hipMemset(d_pollution, 7, pollution_bytes));
  }

  auto prepare = [&](int rep) -> const uint8_t* {
    if (mode == "lgkm") return nullptr;
    const uint64_t slice = static_cast<uint64_t>(rep % 120) * (2ull << 20);
    const uint8_t* base = d_target + slice;
    if (state == "L2-hot") {
      hipLaunchKernelGGL(touch_memory_kernel, dim3(80), dim3(256), 0, 0,
                         base, 128ull << 10, d_touch_sink);
    } else {
      hipLaunchKernelGGL(touch_memory_kernel, dim3(320), dim3(256), 0, 0,
                         d_pollution, pollution_bytes, d_touch_sink);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    return base;
  };

  auto launch = [&](const uint8_t* base) {
    if (mode == "lgkm") {
      if (op == "write") launch_lgkm<true>(n, threshold, width, conflict, age, d_out);
      else launch_lgkm<false>(n, threshold, width, conflict, age, d_out);
    } else if (mode == "vmem") {
      launch_vmem(n, threshold, base, age, d_out);
    } else if (mode == "mixed") {
      launch_mixed(kind, base, age, conflict, d_out);
    } else {
      std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
      std::exit(16);
    }
    HIP_CHECK(hipGetLastError());
  };

  for (int warm = 0; warm < 3; ++warm) {
    const uint8_t* warm_base = prepare(warm);
    launch(warm_base);
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  print_record_header();
  for (int rep = 0; rep < repeats; ++rep) {
    const uint8_t* base = prepare(rep + 1);
    HIP_CHECK(hipMemset(d_out, 0, sizeof(TimingRecord)));
    HIP_CHECK(hipEventRecord(start));
    launch(base);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    TimingRecord h{};
    HIP_CHECK(hipMemcpy(&h, d_out, sizeof(h), hipMemcpyDeviceToHost));
    const std::string variant = mode == "mixed" ? mixed_name(kind) : "measured";
    print_record(mode.c_str(), rep, 0, 1, variant,
                 mode == "vmem" ? "read" : op, n, threshold, age, conflict,
                 mode == "lgkm" ? "none" : state, "none", width, 0, 0, 0,
                 ms * 1000.0f, h);
  }

  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipEventDestroy(start));
  if (d_touch_sink) HIP_CHECK(hipFree(d_touch_sink));
  if (d_pollution) HIP_CHECK(hipFree(d_pollution));
  if (d_target) HIP_CHECK(hipFree(d_target));
  HIP_CHECK(hipFree(d_out));
  return bw::finish("waitcnt");
}
