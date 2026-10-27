/* BW1000 single-file HIP program: lds/to_global
 * Compile: hipcc -O2 -std=c++17 --offload-arch=gfx936 to_global.hip -o bench
 * Run: ./bench --mode completion --state L2-hot --width 4 --count 1 --repeats 30
 * Minimal run (NOT a timing reproduction): ./bench --mode completion --state L2-hot --width 4 --count 1 --repeats 1
 * No local headers, private libraries, scripts, data files or other source files required.
 * Only public HIP/DTK, its driver/runtime, and the C++ standard library are required.
 * Prints in-process summaries in timer tick and calibrated ns. BW1000_RAW=1 also prints samples.
 * Original measurement core: vgpr/src/lds_global_bench.cpp; SHA256=fc62b956077a2a28b8e2e53ac456726d3aa7b3f2aa874ee26a337856eab3a50a
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
    if(family=="lds_global") {t=str(r,"path");if(s=="path") {put("completion_chain","whole Wave64 chain",num(r,"completion_ticks"));put("issue_chain","whole Wave64 chain",num(r,"issue_ticks"));put("completion_per_byte","byte",num(r,"cycles_per_byte"));}continue;}
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

// Host-side conversion only: clock64 ticks are not assumed to be core cycles.
static double mb_tick_mhz() { return bw::mhz; }


using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));

#define HIP_CHECK(call) do { \
  hipError_t e_ = (call); \
  if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s:%d: %s -> %s\n", __FILE__, __LINE__, #call, hipGetErrorString(e_)); \
    std::exit(2); \
  } \
} while (0)

static constexpr uint32_t kLdsWords = 16384;

struct PathResult {
  uint64_t issue_ticks;
  uint64_t completion_ticks;
  uint32_t value;
  uint32_t pad;
};

template<int W> struct VType;
template<> struct VType<1> { using type = uint32_t; };
template<> struct VType<2> { using type = u32x2; };
template<> struct VType<4> { using type = u32x4; };
template<int W> using V = typename VType<W>::type;

template<int W>
__device__ __forceinline__ V<W> make_value(uint32_t x) {
  if constexpr (W == 1) return x;
  else if constexpr (W == 2) return V<W>{x, x + 1u};
  else return V<W>{x, x + 1u, x + 2u, x + 3u};
}

template<int W>
__device__ __forceinline__ uint32_t sum_value(V<W> x) {
  if constexpr (W == 1) return x;
  else {
    uint32_t s = 0;
#pragma unroll
    for (int i = 0; i < W; ++i) s += x[i];
    return s;
  }
}

template<int W> __device__ __forceinline__ V<W> lds_read(uint32_t addr);
template<> __device__ __forceinline__ uint32_t lds_read<1>(uint32_t addr) {
  uint32_t v;
  asm volatile("ds_read_b32 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  return v;
}
template<> __device__ __forceinline__ u32x2 lds_read<2>(uint32_t addr) {
  u32x2 v;
  asm volatile("ds_read_b64 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  return v;
}
template<> __device__ __forceinline__ u32x4 lds_read<4>(uint32_t addr) {
  u32x4 v;
  asm volatile("ds_read_b128 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  return v;
}

template<int W> __device__ __forceinline__ void global_store(uint8_t* p, V<W> v);
template<> __device__ __forceinline__ void global_store<1>(uint8_t* p, uint32_t v) {
  const uint64_t a = reinterpret_cast<uint64_t>(p);
  asm volatile("global_store_dword %0, %1, off" :: "v"(a), "v"(v) : "memory");
}
template<> __device__ __forceinline__ void global_store<2>(uint8_t* p, u32x2 v) {
  const uint64_t a = reinterpret_cast<uint64_t>(p);
  asm volatile("global_store_dwordx2 %0, %1, off" :: "v"(a), "v"(v) : "memory");
}
template<> __device__ __forceinline__ void global_store<4>(uint8_t* p, u32x4 v) {
  const uint64_t a = reinterpret_cast<uint64_t>(p);
  asm volatile("global_store_dwordx4 %0, %1, off" :: "v"(a), "v"(v) : "memory");
}

__device__ __forceinline__ void wait_lds() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}
__device__ __forceinline__ void wait_global() {
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

template<int W>
__device__ __forceinline__ uint32_t conflict_word(uint32_t lane, uint32_t degree) {
  uint32_t slots = 32u / (W * degree);
  if (slots == 0u) slots = 1u;
  const uint32_t slot = lane % slots;
  const uint32_t rank = lane / slots;
  return slot * W + 32u * (rank + 1u);
}

template<int W>
__global__ void lds_global_path_kernel(uint8_t* out, uint32_t degree, uint32_t count,
                                       PathResult* result) {
  __shared__ uint32_t lds[kLdsWords];
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t word = conflict_word<W>(lane, degree);
#pragma unroll
  for (int i = 0; i < W; ++i) lds[word + i] = 0x100000u * lane + i + 1u;
  __syncthreads();
  uint32_t sink = 0;
  const uint64_t t0 = clock64();
#pragma clang loop unroll(disable)
  for (uint32_t i = 0; i < count; ++i) {
    V<W> v = lds_read<W>(word * 4u);
    wait_lds();
    global_store<W>(out + (static_cast<uint64_t>(i) * 64u + lane) * W * 4u, v);
    sink ^= sum_value<W>(v) + i;
  }
  const uint64_t t_issue = clock64();
  wait_global();
  const uint64_t t_done = clock64();
  if (lane == 0) result[0] = PathResult{t_issue - t0, t_done - t0, sink, 0};
}

template<int W>
__global__ void global_store_only_kernel(uint8_t* out, uint32_t count, PathResult* result) {
  __shared__ uint32_t launch_control[64];
  const uint32_t lane = threadIdx.x & 63u;
  launch_control[lane] = lane;
  __syncthreads();
  V<W> v = make_value<W>(lane + 1u);
  uint32_t sink = 0;
  const uint64_t t0 = clock64();
#pragma clang loop unroll(disable)
  for (uint32_t i = 0; i < count; ++i) {
    global_store<W>(out + (static_cast<uint64_t>(i) * 64u + lane) * W * 4u, v);
    sink ^= sum_value<W>(v) + i;
  }
  const uint64_t t_issue = clock64();
  wait_global();
  const uint64_t t_done = clock64();
  if (lane == 0) result[0] = PathResult{t_issue - t0, t_done - t0, sink, 0};
}

template<int W>
__global__ void path_stream_kernel(uint8_t* out, uint32_t degree, uint32_t count,
                                   uint32_t* sink_out) {
  __shared__ uint32_t lds[kLdsWords];
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t wave = threadIdx.x >> 6;
  const uint32_t wave_count = blockDim.x >> 6;
  const uint32_t word = wave * 4096u + conflict_word<W>(lane, degree);
#pragma unroll
  for (int i = 0; i < W; ++i) lds[word + i] = lane + i + 1u;
  __syncthreads();
  const uint64_t global_wave = static_cast<uint64_t>(blockIdx.x) * wave_count + wave;
  const uint64_t wave_base = global_wave * count * 64ull * W * 4ull;
  uint32_t sink = 0;
#pragma clang loop unroll(disable)
  for (uint32_t i = 0; i < count; ++i) {
    V<W> v = lds_read<W>(word * 4u);
    wait_lds();
    global_store<W>(out + wave_base + (static_cast<uint64_t>(i) * 64u + lane) * W * 4u, v);
    sink ^= sum_value<W>(v) + i;
  }
  wait_global();
  if (lane == 0) sink_out[global_wave] = sink;
}

extern "C" __global__ void prime_l2_kernel(const uint8_t* p, uint64_t bytes,
                                             uint32_t* sink_out) {
  const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t threads = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  uint32_t sink = 0;
  for (uint64_t off = tid * 64u; off < bytes; off += threads * 64u) {
    uint32_t v;
    const uint64_t a = reinterpret_cast<uint64_t>(p + off);
    asm volatile("global_load_dword %0, %1, off glc" : "=v"(v) : "v"(a) : "memory");
    sink ^= v;
  }
  wait_global();
  if (tid == 0) sink_out[0] = sink;
}

static std::string get_arg(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i) if (!std::strcmp(argv[i], key)) return argv[i + 1];
  return def;
}
static int get_int(int argc, char** argv, const char* key, int def) {
  const std::string fallback = std::to_string(def);
  return std::atoi(get_arg(argc, argv, key, fallback.c_str()).c_str());
}

template<int W>
static void launch_path(uint8_t* out, uint32_t degree, uint32_t count, PathResult* result) {
  hipLaunchKernelGGL((lds_global_path_kernel<W>), dim3(1), dim3(64), 0, 0,
                     out, degree, count, result);
}
template<int W>
static void launch_store_only(uint8_t* out, uint32_t count, PathResult* result) {
  hipLaunchKernelGGL((global_store_only_kernel<W>), dim3(1), dim3(64), 0, 0,
                     out, count, result);
}

static void prepare_state(const std::string& state, uint8_t* target, uint64_t target_bytes,
                          uint8_t* pollution, uint64_t pollution_bytes, uint32_t* sink) {
  if (state == "L2-hot") {
    const uint64_t bytes = target_bytes < (1ull << 20) ? target_bytes : (1ull << 20);
    hipLaunchKernelGGL(prime_l2_kernel, dim3(80), dim3(256), 0, 0, target, bytes, sink);
  } else if (state == "L2-cold") {
    hipLaunchKernelGGL(prime_l2_kernel, dim3(320), dim3(256), 0, 0,
                       pollution, pollution_bytes, sink);
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
}

template<int W>
static void run_completion(const std::string& state, int conflict, int count, int repeats,
                           bool store_only) {
  const uint64_t per_run = static_cast<uint64_t>(count) * 64u * W * 4u;
  const uint64_t target_bytes = std::max<uint64_t>(64ull << 20, per_run * repeats + (1ull << 20));
  const uint64_t pollution_bytes = 32ull << 20;
  uint8_t* target = nullptr;
  uint8_t* pollution = nullptr;
  uint32_t* sink = nullptr;
  PathResult* d_result = nullptr;
  HIP_CHECK(hipMalloc(&target, target_bytes));
  HIP_CHECK(hipMalloc(&pollution, pollution_bytes));
  HIP_CHECK(hipMalloc(&sink, sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(&d_result, sizeof(PathResult)));
  HIP_CHECK(hipMemset(target, 0, target_bytes));
  HIP_CHECK(hipMemset(pollution, 1, pollution_bytes));
  for (int r = 0; r < repeats; ++r) {
    prepare_state(state, target, target_bytes, pollution, pollution_bytes, sink);
    uint64_t offset = 0;
    if (state != "L2-hot") offset = (static_cast<uint64_t>(r) * per_run) % (target_bytes - per_run);
    if (store_only) launch_store_only<W>(target + offset, count, d_result);
    else launch_path<W>(target + offset, conflict, count, d_result);
    HIP_CHECK(hipGetLastError());
    PathResult h{};
    HIP_CHECK(hipMemcpy(&h, d_result, sizeof(h), hipMemcpyDeviceToHost));
    uint32_t check[4]{};
    HIP_CHECK(hipMemcpy(check, target + offset, W * sizeof(uint32_t), hipMemcpyDeviceToHost));
    int validation_errors = 0;
    for (int i = 0; i < W; ++i) if (check[i] != static_cast<uint32_t>(i + 1)) ++validation_errors;
    const uint64_t bytes = per_run;
    bw::emit("path,%s,%s,%s,%d,%d,%d,%d,%llu,%llu,%llu,%.9f,%d\n",
                store_only ? "global_store_only" : "lds_vgpr_global_store",
                W == 1 ? "dword" : (W == 2 ? "dwordx2" : "dwordx4"), state.c_str(),
                W * 4, conflict, count, r, static_cast<unsigned long long>(bytes),
                static_cast<unsigned long long>(h.issue_ticks),
                static_cast<unsigned long long>(h.completion_ticks),
                static_cast<double>(h.completion_ticks) / static_cast<double>(bytes), validation_errors);
  }
  HIP_CHECK(hipFree(d_result));
  HIP_CHECK(hipFree(sink));
  HIP_CHECK(hipFree(pollution));
  HIP_CHECK(hipFree(target));
}

template<int W>
static void run_streaming(int conflict, int blocks, int threads, int count, int repeats) {
  const uint64_t waves = static_cast<uint64_t>(blocks) * (threads / 64);
  const uint64_t bytes = waves * count * 64ull * W * 4ull;
  uint8_t* out = nullptr;
  uint32_t* sink = nullptr;
  HIP_CHECK(hipMalloc(&out, bytes));
  HIP_CHECK(hipMalloc(&sink, waves * sizeof(uint32_t)));
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  hipLaunchKernelGGL((path_stream_kernel<W>), dim3(blocks), dim3(threads), 0, 0,
                     out, conflict, 32u, sink);
  HIP_CHECK(hipDeviceSynchronize());
  for (int r = 0; r < repeats; ++r) {
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL((path_stream_kernel<W>), dim3(blocks), dim3(threads), 0, 0,
                       out, conflict, static_cast<uint32_t>(count), sink);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    const double gbps = static_cast<double>(bytes) / (static_cast<double>(ms) * 1.0e6);
    const double cycles_per_byte = static_cast<double>(ms) * (mb_tick_mhz() * 1000.0) / static_cast<double>(bytes);
    uint32_t check[4]{};
    HIP_CHECK(hipMemcpy(check, out, W * sizeof(uint32_t), hipMemcpyDeviceToHost));
    int validation_errors = 0;
    for (int i = 0; i < W; ++i) if (check[i] != static_cast<uint32_t>(i + 1)) ++validation_errors;
    bw::emit("streaming,lds_vgpr_global_store,%s,streaming,%d,%d,%d,%d,%d,%d,%llu,%.6f,%.6f,%.9f,%d\n",
                W == 1 ? "dword" : (W == 2 ? "dwordx2" : "dwordx4"), W * 4,
                conflict, count, blocks, threads, r, static_cast<unsigned long long>(bytes),
                ms, gbps, cycles_per_byte, validation_errors);
  }
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipFree(sink));
  HIP_CHECK(hipFree(out));
}

int main(int argc, char** argv) {
  bw::start();

  const std::string mode = get_arg(argc, argv, "--mode", "completion");
  const std::string state = get_arg(argc, argv, "--state", "L2-hot");
  const int width = get_int(argc, argv, "--width", 4);
  const int conflict = get_int(argc, argv, "--conflict", 1);
  const int count = get_int(argc, argv, "--count", 1);
  const int repeats = get_int(argc, argv, "--repeats", 30);
  const bool store_only = mode == "store_only";
  hipDeviceProp_t p{};
  HIP_CHECK(hipGetDeviceProperties(&p, 0));
  if (p.warpSize != 64) return 3;

  if (mode == "completion" || mode == "store_only" || mode == "profile") {
    bw::line("suite,path,global_instruction,state,width,conflict,count,repeat,bytes,issue_ticks,completion_ticks,cycles_per_byte,validation_errors");
    const int reps = mode == "profile" ? 1 : repeats;
    if (width == 4) run_completion<1>(state, conflict, count, reps, store_only);
    else if (width == 8) run_completion<2>(state, conflict, count, reps, store_only);
    else if (width == 16) run_completion<4>(state, conflict, count, reps, store_only);
    else return 4;
  } else if (mode == "streaming") {
    const int blocks = get_int(argc, argv, "--blocks", 320);
    const int threads = get_int(argc, argv, "--threads", 256);
    bw::line("suite,path,global_instruction,state,width,conflict,count,blocks,threads,repeat,bytes,event_ms,GBps,cycles_per_byte,validation_errors");
    if (width == 4) run_streaming<1>(conflict, blocks, threads, count, repeats);
    else if (width == 8) run_streaming<2>(conflict, blocks, threads, count, repeats);
    else if (width == 16) run_streaming<4>(conflict, blocks, threads, count, repeats);
    else return 4;
  } else {
    std::fprintf(stderr, "Usage: %s --mode completion|store_only|streaming|profile --state L2-hot|L2-cold|streaming --width 4|8|16 --conflict N --count N --repeats N\n", argv[0]);
    return 1;
  }
  HIP_CHECK(hipDeviceSynchronize());
  return bw::finish("lds_global");
}
