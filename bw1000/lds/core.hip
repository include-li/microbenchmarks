/* BW1000 single-file HIP program: lds/core
 * Compile: hipcc -O2 -std=c++17 --offload-arch=gfx936 core.hip -o bench
 * Run: ./bench --mode core --width 4 --repeats 30
 * Minimal run (NOT a timing reproduction): ./bench --mode core --width 4 --conflict 1 --repeats 1
 * No local headers, private libraries, scripts, data files or other source files required.
 * Only public HIP/DTK, its driver/runtime, and the C++ standard library are required.
 * Prints in-process summaries in timer tick and calibrated ns. BW1000_RAW=1 also prints samples.
 * Original measurement core: vgpr/src/lds_bench.cpp; SHA256=d27ae46cdb7b0dfd866c053877086d485feb10215b5ff7545973d01688d2e0de
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

static constexpr int kWave = 64;
static constexpr int kLdsWords = 16384;

struct TickResult {
  uint64_t issue_ticks;
  uint64_t completion_ticks;
  uint32_t value;
  uint32_t pad;
};

template<int W> struct DType;
template<> struct DType<1> { using type = uint32_t; };
template<> struct DType<2> { using type = u32x2; };
template<> struct DType<3> { using type = u32x3; };
template<> struct DType<4> { using type = u32x4; };
template<int W> using DValue = typename DType<W>::type;

template<int W>
__device__ __forceinline__ DValue<W> make_value(uint32_t x) {
  if constexpr (W == 1) {
    return x;
  } else if constexpr (W == 2) {
    return DValue<W>{x, x + 1u};
  } else if constexpr (W == 3) {
    return DValue<W>{x, x + 1u, x + 2u};
  } else {
    return DValue<W>{x, x + 1u, x + 2u, x + 3u};
  }
}

template<int W>
__device__ __forceinline__ uint32_t value_word0(DValue<W> x) {
  if constexpr (W == 1) return x;
  else return x[0];
}

template<int W>
__device__ __forceinline__ uint32_t value_sum(DValue<W> x) {
  if constexpr (W == 1) {
    return x;
  } else {
    uint32_t s = 0;
#pragma unroll
    for (int i = 0; i < W; ++i) s += x[i];
    return s;
  }
}

template<int W> __device__ __forceinline__ DValue<W> ds_read(uint32_t addr);
template<> __device__ __forceinline__ uint32_t ds_read<1>(uint32_t addr) {
  uint32_t v;
  asm volatile("ds_read_b32 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  return v;
}
template<> __device__ __forceinline__ u32x2 ds_read<2>(uint32_t addr) {
  u32x2 v;
  asm volatile("ds_read_b64 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  return v;
}
template<> __device__ __forceinline__ u32x3 ds_read<3>(uint32_t addr) {
  u32x3 v;
  asm volatile("ds_read_b96 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  return v;
}
template<> __device__ __forceinline__ u32x4 ds_read<4>(uint32_t addr) {
  u32x4 v;
  asm volatile("ds_read_b128 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  return v;
}

template<int W> __device__ __forceinline__ void ds_write(uint32_t addr, DValue<W> v);
template<> __device__ __forceinline__ void ds_write<1>(uint32_t addr, uint32_t v) {
  asm volatile("ds_write_b32 %0, %1" :: "v"(addr), "v"(v) : "memory");
}
template<> __device__ __forceinline__ void ds_write<2>(uint32_t addr, u32x2 v) {
  asm volatile("ds_write_b64 %0, %1" :: "v"(addr), "v"(v) : "memory");
}
template<> __device__ __forceinline__ void ds_write<3>(uint32_t addr, u32x3 v) {
  asm volatile("ds_write_b96 %0, %1" :: "v"(addr), "v"(v) : "memory");
}
template<> __device__ __forceinline__ void ds_write<4>(uint32_t addr, u32x4 v) {
  asm volatile("ds_write_b128 %0, %1" :: "v"(addr), "v"(v) : "memory");
}

__device__ __forceinline__ void wait_lds() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

template<int W>
__device__ __forceinline__ uint32_t conflict_word(uint32_t lane, uint32_t degree,
                                                   uint32_t align_bytes) {
  const uint32_t banks_per_lane = W;
  uint32_t slots = 32u / (banks_per_lane * degree);
  if (slots == 0u) slots = 1u;
  const uint32_t slot = lane % slots;
  const uint32_t rank = lane / slots;
  return (align_bytes >> 2) + slot * banks_per_lane + 32u * (rank + 1u);
}

extern "C" __global__ void wait_baseline_kernel(TickResult* out) {
  __shared__ uint32_t lds[kLdsWords];
  lds[threadIdx.x] = threadIdx.x;
  __syncthreads();
  const uint64_t t0 = clock64();
  wait_lds();
  const uint64_t t1 = clock64();
  if (threadIdx.x == 0) out[0] = TickResult{t1 - t0, t1 - t0, lds[0], 0};
}

template<int W, bool IsWrite>
__global__ void stride_completion_kernel(uint32_t stride_words, uint32_t align_bytes,
                                         TickResult* out) {
  __shared__ uint32_t lds[kLdsWords];
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t word = (align_bytes >> 2) + lane * stride_words;
#pragma unroll
  for (int i = 0; i < W; ++i) lds[word + i] = 0x10000u * lane + i;
  __syncthreads();
  DValue<W> v = make_value<W>(lane + 1u);
  const uint64_t t0 = clock64();
  if constexpr (IsWrite) ds_write<W>(word * 4u + (align_bytes & 3u), v);
  else v = ds_read<W>(word * 4u + (align_bytes & 3u));
  wait_lds();
  const uint64_t t1 = clock64();
  if (lane == 0) out[0] = TickResult{t1 - t0, t1 - t0, value_sum<W>(v), 0};
}

template<int W, bool IsWrite>
__global__ void byte_stride_completion_kernel(uint32_t stride_bytes, TickResult* out) {
  __shared__ uint32_t lds[kLdsWords];
  const uint32_t lane = threadIdx.x & 63u;
  for (uint32_t i = lane; i < 4096u; i += 64u) lds[i] = i * 0x01010101u;
  __syncthreads();
  const uint32_t byte_addr = 256u + lane * stride_bytes;
  DValue<W> v = make_value<W>(lane + 1u);
  const uint64_t t0 = clock64();
  if constexpr (IsWrite) ds_write<W>(byte_addr, v);
  else v = ds_read<W>(byte_addr);
  wait_lds();
  const uint64_t t1 = clock64();
  if (lane == 0) out[0] = TickResult{t1 - t0, t1 - t0, value_sum<W>(v), 0};
}

template<int W, bool IsWrite>
__global__ void conflict_completion_kernel(uint32_t degree, uint32_t broadcast,
                                           uint32_t align_bytes, TickResult* out) {
  __shared__ uint32_t lds[kLdsWords];
  const uint32_t lane = threadIdx.x & 63u;
  uint32_t word = conflict_word<W>(lane, degree, align_bytes);
  if (broadcast) word = (align_bytes >> 2) + 32u;
#pragma unroll
  for (int i = 0; i < W; ++i) lds[word + i] = 0x10000u * lane + i;
  __syncthreads();
  DValue<W> v = make_value<W>(lane + 1u);
  const uint32_t byte_addr = word * 4u + (align_bytes & 3u);
  const uint64_t t0 = clock64();
  if constexpr (IsWrite) ds_write<W>(byte_addr, v);
  else v = ds_read<W>(byte_addr);
  wait_lds();
  const uint64_t t1 = clock64();
  if (lane == 0) out[0] = TickResult{t1 - t0, t1 - t0, value_sum<W>(v), 0};
}

template<int W, bool IsWrite>
__global__ void phase_pair_kernel(uint32_t probe_lane, uint32_t align_bytes,
                                  TickResult* out) {
  __shared__ uint32_t lds[kLdsWords];
  const uint32_t lane = threadIdx.x & 63u;
  uint32_t word = conflict_word<W>(lane, 1u, align_bytes);
  if (lane == probe_lane) word = (align_bytes >> 2) + 32u * (400u + probe_lane);
#pragma unroll
  for (int i = 0; i < W; ++i) lds[word + i] = 0x10000u * lane + i;
  __syncthreads();
  DValue<W> v = make_value<W>(lane + 1u);
  const uint32_t byte_addr = word * 4u + (align_bytes & 3u);
  const uint64_t t0 = clock64();
  if constexpr (IsWrite) ds_write<W>(byte_addr, v);
  else v = ds_read<W>(byte_addr);
  wait_lds();
  const uint64_t t1 = clock64();
  if (lane == 0) out[0] = TickResult{t1 - t0, t1 - t0, value_sum<W>(v), 0};
}

template<int W, bool IsWrite>
__global__ void phase_active_pair_kernel(uint32_t ref_lane, uint32_t probe_lane, TickResult* out) {
  __shared__ uint32_t lds[kLdsWords];
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t ref_word = 32u * (64u + ref_lane);
  const uint32_t probe_word = 32u * (192u + probe_lane);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < W; ++i) {
      lds[ref_word + i] = i + 1u;
      lds[probe_word + i] = i + 17u;
    }
  }
  __syncthreads();
  DValue<W> v = make_value<W>(lane + 1u);
  const uint64_t t0 = clock64();
  if (lane == ref_lane || lane == probe_lane) {
    const uint32_t byte_addr = (lane == ref_lane ? ref_word : probe_word) * 4u;
    if constexpr (IsWrite) ds_write<W>(byte_addr, v);
    else v = ds_read<W>(byte_addr);
    wait_lds();
  }
  const uint64_t t1 = clock64();
  if (lane == 0) out[0] = TickResult{t1 - t0, t1 - t0, value_sum<W>(v), 0};
}

template<int W>
__global__ void dependent_read_kernel(uint32_t degree, uint32_t steps,
                                      uint32_t align_bytes, TickResult* out) {
  __shared__ uint32_t lds[kLdsWords];
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t a = conflict_word<W>(lane, degree, align_bytes);
  const uint32_t b = a + 32u * 128u;
  lds[a] = b * 4u + (align_bytes & 3u);
  lds[b] = a * 4u + (align_bytes & 3u);
#pragma unroll
  for (int i = 1; i < W; ++i) {
    lds[a + i] = lane + i;
    lds[b + i] = lane + i + 17u;
  }
  __syncthreads();
  uint32_t p = a * 4u + (align_bytes & 3u);
  uint32_t sink = 0;
  const uint64_t t0 = clock64();
#pragma clang loop unroll(disable)
  for (uint32_t i = 0; i < steps; ++i) {
    DValue<W> v = ds_read<W>(p);
    wait_lds();
    p = value_word0<W>(v);
    sink ^= value_sum<W>(v) + i;
  }
  const uint64_t t1 = clock64();
  if (lane == 0) out[0] = TickResult{t1 - t0, t1 - t0, sink ^ p, 0};
}

template<int W>
__global__ void dependent_baseline_kernel(uint32_t degree, uint32_t steps,
                                          uint32_t align_bytes, TickResult* out) {
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t a = conflict_word<W>(lane, degree, align_bytes) * 4u + (align_bytes & 3u);
  const uint32_t b = a + 32u * 128u * 4u;
  uint32_t p = a;
  uint32_t sink = 0;
  const uint64_t t0 = clock64();
#pragma clang loop unroll(disable)
  for (uint32_t i = 0; i < steps; ++i) {
    p = (p == a) ? b : a;
    sink ^= p + i;
    asm volatile("" : "+v"(p), "+v"(sink));
  }
  const uint64_t t1 = clock64();
  if (lane == 0) out[0] = TickResult{t1 - t0, t1 - t0, sink ^ p, 0};
}

template<int W, bool IsWrite, int N>
__global__ void issue_kernel(uint32_t degree, uint32_t align_bytes, TickResult* out) {
  __shared__ uint32_t lds[kLdsWords];
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t word = conflict_word<W>(lane, degree, align_bytes);
#pragma unroll
  for (int i = 0; i < W; ++i) lds[word + i] = lane + i;
  __syncthreads();
  const uint32_t byte_addr = word * 4u + (align_bytes & 3u);
  DValue<W> vals[N];
#pragma unroll
  for (int i = 0; i < N; ++i) vals[i] = make_value<W>(lane + i + 1u);
  const uint64_t t0 = clock64();
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const uint32_t independent_addr = byte_addr + static_cast<uint32_t>(i) * 4096u;
    if constexpr (IsWrite) ds_write<W>(independent_addr, vals[i]);
    else vals[i] = ds_read<W>(independent_addr);
  }
  const uint64_t t_issue = clock64();
  wait_lds();
  const uint64_t t_done = clock64();
  uint32_t sink = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) sink += value_sum<W>(vals[i]);
  if (lane == 0) out[0] = TickResult{t_issue - t0, t_done - t0, sink, 0};
}

extern "C" __global__ void issue_baseline_kernel(TickResult* out) {
  const uint64_t t0 = clock64();
  const uint64_t t_issue = clock64();
  wait_lds();
  const uint64_t t_done = clock64();
  if (threadIdx.x == 0) out[0] = TickResult{t_issue - t0, t_done - t0, 0, 0};
}

template<int W, bool IsWrite>
__global__ void throughput_kernel(uint32_t degree, uint32_t iters, uint32_t* sink_out) {
  __shared__ uint32_t lds[kLdsWords];
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t wave = threadIdx.x >> 6;
  const uint32_t base = wave * 4096u;
  const uint32_t word = base + conflict_word<W>(lane, degree, 0u);
#pragma unroll
  for (int i = 0; i < W; ++i) lds[word + i] = lane + i;
  __syncthreads();
  DValue<W> v = make_value<W>(lane + 1u);
  uint32_t sink = 0;
#pragma clang loop unroll(disable)
  for (uint32_t i = 0; i < iters; ++i) {
    if constexpr (IsWrite) ds_write<W>(word * 4u, v);
    else {
      v = ds_read<W>(word * 4u);
      sink += value_sum<W>(v);
    }
  }
  wait_lds();
  if (lane == 0) sink_out[blockIdx.x * (blockDim.x / 64u) + wave] = sink;
}

static std::string arg_value(int argc, char** argv, const char* key,
                             const char* def = "") {
  for (int i = 1; i + 1 < argc; ++i) if (!std::strcmp(argv[i], key)) return argv[i + 1];
  return def;
}

static int arg_int(int argc, char** argv, const char* key, int def) {
  return std::atoi(arg_value(argc, argv, key, std::to_string(def).c_str()).c_str());
}

template<int W, bool IsWrite>
static void launch_stride(uint32_t stride, uint32_t align, TickResult* d) {
  hipLaunchKernelGGL((stride_completion_kernel<W, IsWrite>), dim3(1), dim3(64), 0, 0,
                     stride, align, d);
}

template<int W, bool IsWrite>
static void launch_byte_stride(uint32_t stride, TickResult* d) {
  hipLaunchKernelGGL((byte_stride_completion_kernel<W, IsWrite>), dim3(1), dim3(64), 0, 0,
                     stride, d);
}

template<int W, bool IsWrite>
static void launch_conflict(uint32_t degree, uint32_t broadcast, uint32_t align,
                            TickResult* d) {
  hipLaunchKernelGGL((conflict_completion_kernel<W, IsWrite>), dim3(1), dim3(64), 0, 0,
                     degree, broadcast, align, d);
}

template<int W, bool IsWrite>
static void launch_phase(uint32_t probe, uint32_t align, TickResult* d) {
  hipLaunchKernelGGL((phase_pair_kernel<W, IsWrite>), dim3(1), dim3(64), 0, 0,
                     probe, align, d);
}

template<int W, bool IsWrite>
static void launch_phase_active(uint32_t probe, TickResult* d) {
  hipLaunchKernelGGL((phase_active_pair_kernel<W, IsWrite>), dim3(1), dim3(64), 0, 0,
                     0u, probe, d);
}

template<int W, bool IsWrite>
static void launch_phase_active_ref(uint32_t ref, uint32_t probe, TickResult* d) {
  hipLaunchKernelGGL((phase_active_pair_kernel<W, IsWrite>), dim3(1), dim3(64), 0, 0,
                     ref, probe, d);
}

template<int W>
static void launch_dependent(uint32_t degree, uint32_t steps, uint32_t align,
                             bool baseline, TickResult* d) {
  if (baseline) {
    hipLaunchKernelGGL((dependent_baseline_kernel<W>), dim3(1), dim3(64), 0, 0,
                       degree, steps, align, d);
  } else {
    hipLaunchKernelGGL((dependent_read_kernel<W>), dim3(1), dim3(64), 0, 0,
                       degree, steps, align, d);
  }
}

template<int W, bool IsWrite, int N>
static void launch_issue_n(uint32_t degree, uint32_t align, TickResult* d) {
  hipLaunchKernelGGL((issue_kernel<W, IsWrite, N>), dim3(1), dim3(64), 0, 0,
                     degree, align, d);
}

template<int W, bool IsWrite>
static void launch_issue(int n, uint32_t degree, uint32_t align, TickResult* d) {
  if (n == 1) launch_issue_n<W, IsWrite, 1>(degree, align, d);
  else if (n == 2) launch_issue_n<W, IsWrite, 2>(degree, align, d);
  else if (n == 4) launch_issue_n<W, IsWrite, 4>(degree, align, d);
  else if (n == 8) launch_issue_n<W, IsWrite, 8>(degree, align, d);
  else launch_issue_n<W, IsWrite, 16>(degree, align, d);
}

static TickResult collect(TickResult* d) {
  TickResult h{};
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(&h, d, sizeof(h), hipMemcpyDeviceToHost));
  return h;
}

template<typename Launch>
static void repeat_launch(int repeats, Launch launch, const char* suite, const char* instruction,
                          const char* op, int width, int param, const char* param_name,
                          int n = 1, int align = 0) {
  TickResult* d = nullptr;
  HIP_CHECK(hipMalloc(&d, sizeof(TickResult)));
  for (int r = 0; r < repeats; ++r) {
    launch(d);
    TickResult h = collect(d);
    bw::emit("%s,%s,%s,%d,%s,%d,%d,%d,%d,%llu,%llu,%u\n",
                suite, instruction, op, width, param_name, param, n, align, r,
                static_cast<unsigned long long>(h.issue_ticks),
                static_cast<unsigned long long>(h.completion_ticks), h.value);
  }
  HIP_CHECK(hipFree(d));
}

template<int W>
static void run_stride_width(bool write, int repeats) {
  const int strides[] = {1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33,
                         63, 64, 65, 95, 96, 97, 127, 128, 129, 255, 256, 257};
  const char* inst_r = W == 1 ? "ds_read_b32" : (W == 2 ? "ds_read_b64" : "ds_read_b128");
  const char* inst_w = W == 1 ? "ds_write_b32" : (W == 2 ? "ds_write_b64" : "ds_write_b128");
  for (int stride : strides) {
    if (write) repeat_launch(repeats, [=](TickResult* d){ launch_stride<W, true>(stride, 0, d); },
                             "bank_stride", inst_w, "write", W * 4, stride, "stride_words");
    else repeat_launch(repeats, [=](TickResult* d){ launch_stride<W, false>(stride, 0, d); },
                       "bank_stride", inst_r, "read", W * 4, stride, "stride_words");
  }
}

static void run_byte_stride(bool write, int repeats) {
  const int strides[] = {1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16, 17, 24, 28, 31, 32, 33,
                         60, 63, 64, 65, 124, 127, 128, 129, 252, 255, 256, 257};
  for (int stride : strides) {
    if (write) repeat_launch(repeats, [=](TickResult* d){ launch_byte_stride<1, true>(stride, d); },
                             "bank_byte_stride", "ds_write_b32", "write", 4, stride, "stride_bytes");
    else repeat_launch(repeats, [=](TickResult* d){ launch_byte_stride<1, false>(stride, d); },
                       "bank_byte_stride", "ds_read_b32", "read", 4, stride, "stride_bytes");
  }
}

template<int W>
static void run_phase_width(bool write, int repeats) {
  const char* inst_r = W == 1 ? "ds_read_b32" : (W == 2 ? "ds_read_b64" : "ds_read_b128");
  const char* inst_w = W == 1 ? "ds_write_b32" : (W == 2 ? "ds_write_b64" : "ds_write_b128");
  for (int probe = 1; probe < 64; ++probe) {
    if (write) repeat_launch(repeats, [=](TickResult* d){ launch_phase<W, true>(probe, 0, d); },
                             "phase_pair", inst_w, "write", W * 4, probe, "probe_lane");
    else repeat_launch(repeats, [=](TickResult* d){ launch_phase<W, false>(probe, 0, d); },
                       "phase_pair", inst_r, "read", W * 4, probe, "probe_lane");
  }
}

template<int W>
static void run_phase_active_width(bool write, int repeats) {
  const char* inst_r = W == 1 ? "ds_read_b32" : (W == 2 ? "ds_read_b64" : "ds_read_b128");
  const char* inst_w = W == 1 ? "ds_write_b32" : (W == 2 ? "ds_write_b64" : "ds_write_b128");
  for (int probe = 1; probe < 64; ++probe) {
    if (write) repeat_launch(repeats, [=](TickResult* d){ launch_phase_active<W, true>(probe, d); },
                             "phase_active_pair", inst_w, "write", W * 4, probe, "probe_lane");
    else repeat_launch(repeats, [=](TickResult* d){ launch_phase_active<W, false>(probe, d); },
                       "phase_active_pair", inst_r, "read", W * 4, probe, "probe_lane");
  }
}

template<int W>
static void run_phase_matrix_width(bool write) {
  const char* inst_r = W == 1 ? "ds_read_b32" : (W == 2 ? "ds_read_b64" : "ds_read_b128");
  const char* inst_w = W == 1 ? "ds_write_b32" : (W == 2 ? "ds_write_b64" : "ds_write_b128");
  TickResult* d = nullptr;
  HIP_CHECK(hipMalloc(&d, sizeof(TickResult)));
  for (int ref = 0; ref < 64; ++ref) {
    for (int probe = 0; probe < 64; ++probe) {
      if (probe == ref) continue;
      if (write) launch_phase_active_ref<W, true>(ref, probe, d);
      else launch_phase_active_ref<W, false>(ref, probe, d);
      TickResult h = collect(d);
      bw::emit("phase_matrix,%s,%s,%d,ref_probe,%d,%d,0,0,%llu,%llu,%u\n",
                  write ? inst_w : inst_r, write ? "write" : "read", W * 4,
                  ref, probe, static_cast<unsigned long long>(h.issue_ticks),
                  static_cast<unsigned long long>(h.completion_ticks), h.value);
    }
  }
  HIP_CHECK(hipFree(d));
}

template<int W>
static void run_core_width(int repeats, int only_conflict = 0) {
  const char* inst_r = W == 1 ? "ds_read_b32" : (W == 2 ? "ds_read_b64" : (W == 3 ? "ds_read_b96" : "ds_read_b128"));
  const char* inst_w = W == 1 ? "ds_write_b32" : (W == 2 ? "ds_write_b64" : (W == 3 ? "ds_write_b96" : "ds_write_b128"));
  const int conflicts[] = {1, 2, 4, 8};
  for (int c : conflicts) {
    if (only_conflict && c != only_conflict) continue;
    if (32 / (W * c) == 0) continue;
    repeat_launch(repeats, [=](TickResult* d){ launch_conflict<W, false>(c, 0, 0, d); },
                  "completion", inst_r, "read", W * 4, c, "conflict");
    repeat_launch(repeats, [=](TickResult* d){ launch_conflict<W, true>(c, 0, 0, d); },
                  "completion", inst_w, "write", W * 4, c, "conflict");
    for (int n : {1, 2, 4, 8, 16}) {
      repeat_launch(repeats, [=](TickResult* d){ launch_issue<W, false>(n, c, 0, d); },
                    "issue", inst_r, "read", W * 4, c, "conflict", n);
      repeat_launch(repeats, [=](TickResult* d){ launch_issue<W, true>(n, c, 0, d); },
                    "issue", inst_w, "write", W * 4, c, "conflict", n);
    }
    TickResult* d = nullptr;
    HIP_CHECK(hipMalloc(&d, sizeof(TickResult)));
    const uint32_t steps = 1024;
    for (int r = 0; r < repeats; ++r) {
      launch_dependent<W>(c, steps, 0, true, d);
      TickResult base = collect(d);
      launch_dependent<W>(c, steps, 0, false, d);
      TickResult dep = collect(d);
      bw::emit("dependent,%s,read,%d,conflict,%d,%u,0,%d,%llu,%llu,%u\n",
                  inst_r, W * 4, c, steps, r,
                  static_cast<unsigned long long>(dep.completion_ticks),
                  static_cast<unsigned long long>(base.completion_ticks), dep.value);
    }
    HIP_CHECK(hipFree(d));
  }
  if (!only_conflict) {
    repeat_launch(repeats, [=](TickResult* d){ launch_conflict<W, false>(1, 1, 0, d); },
                  "broadcast", inst_r, "read", W * 4, 1, "same_address");
  }
}

template<int W>
static void run_alignment_width(int repeats) {
  const char* inst_r = W == 2 ? "ds_read_b64" : (W == 3 ? "ds_read_b96" : "ds_read_b128");
  const char* inst_w = W == 2 ? "ds_write_b64" : (W == 3 ? "ds_write_b96" : "ds_write_b128");
  for (int align : {0, 1, 2, 3, 4, 8, 12}) {
    repeat_launch(repeats, [=](TickResult* d){ launch_conflict<W, false>(1, 0, align, d); },
                  "alignment", inst_r, "read", W * 4, 1, "conflict", 1, align);
    repeat_launch(repeats, [=](TickResult* d){ launch_conflict<W, true>(1, 0, align, d); },
                  "alignment", inst_w, "write", W * 4, 1, "conflict", 1, align);
  }
}

template<int W, bool IsWrite>
static void run_throughput_case(int blocks, int threads, int degree, int iters, int repeats) {
  uint32_t* d = nullptr;
  HIP_CHECK(hipMalloc(&d, static_cast<size_t>(blocks) * (threads / 64) * sizeof(uint32_t)));
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  hipLaunchKernelGGL((throughput_kernel<W, IsWrite>), dim3(blocks), dim3(threads), 0, 0,
                     degree, 100u, d);
  HIP_CHECK(hipDeviceSynchronize());
  for (int r = 0; r < repeats; ++r) {
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL((throughput_kernel<W, IsWrite>), dim3(blocks), dim3(threads), 0, 0,
                       degree, static_cast<uint32_t>(iters), d);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    const uint64_t waves = static_cast<uint64_t>(blocks) * (threads / 64);
    const uint64_t bytes = waves * 64ull * iters * W * 4ull;
    const double gbps = static_cast<double>(bytes) / (static_cast<double>(ms) * 1.0e6);
    bw::emit("throughput,%s,%s,%d,conflict,%d,%d,%d,%d,%d,%.6f,%.6f,%llu\n",
                IsWrite ? (W == 1 ? "ds_write_b32" : (W == 2 ? "ds_write_b64" : "ds_write_b128"))
                        : (W == 1 ? "ds_read_b32" : (W == 2 ? "ds_read_b64" : "ds_read_b128")),
                IsWrite ? "write" : "read", W * 4, degree, blocks, threads, iters, r,
                ms, gbps, static_cast<unsigned long long>(bytes));
  }
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d));
}

int main(int argc, char** argv) {
  bw::start();

  const std::string mode = arg_value(argc, argv, "--mode", "core");
  const std::string op = arg_value(argc, argv, "--op", "read");
  const int width = arg_int(argc, argv, "--width", 4);
  const int repeats = arg_int(argc, argv, "--repeats", 30);
  const int conflict = arg_int(argc, argv, "--conflict", 0);
  const bool write = op == "write";
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  if (prop.warpSize != 64) {
    std::fprintf(stderr, "requires Wave64, got %d\n", prop.warpSize);
    return 3;
  }

  if (mode != "baseline") {
    bw::line("suite,instruction,op,width,param_name,param,n,alignment,repeat,issue_ticks,completion_ticks,value");
    repeat_launch(repeats, [](TickResult* d){hipLaunchKernelGGL(wait_baseline_kernel,dim3(1),dim3(64),0,0,d);},"baseline","s_waitcnt_lgkmcnt0","none",0,0,"none");
  }
  if (mode == "baseline") {
    bw::line("suite,instruction,op,width,param_name,param,n,alignment,repeat,issue_ticks,completion_ticks,value");
    repeat_launch(repeats, [](TickResult* d){ hipLaunchKernelGGL(wait_baseline_kernel, dim3(1), dim3(64), 0, 0, d); },
                  "baseline", "s_waitcnt_lgkmcnt0", "none", 0, 0, "none");
    repeat_launch(repeats, [](TickResult* d){ hipLaunchKernelGGL(issue_baseline_kernel, dim3(1), dim3(64), 0, 0, d); },
                  "baseline", "clock_issue_wait", "none", 0, 0, "none");
  } else if (mode == "bank_stride") {
    bw::line("suite,instruction,op,width,param_name,param,n,alignment,repeat,issue_ticks,completion_ticks,value");
    if (width == 4) run_stride_width<1>(write, repeats);
    else if (width == 8) run_stride_width<2>(write, repeats);
    else if (width == 16) run_stride_width<4>(write, repeats);
    else return 4;
  } else if (mode == "bank_byte_stride") {
    bw::line("suite,instruction,op,width,param_name,param,n,alignment,repeat,issue_ticks,completion_ticks,value");
    if (width != 4) return 4;
    run_byte_stride(write, repeats);
  } else if (mode == "phase") {
    bw::line("suite,instruction,op,width,param_name,param,n,alignment,repeat,issue_ticks,completion_ticks,value");
    if (width == 4) run_phase_width<1>(write, repeats);
    else if (width == 8) run_phase_width<2>(write, repeats);
    else if (width == 16) run_phase_width<4>(write, repeats);
    else return 4;
  } else if (mode == "phase_active") {
    bw::line("suite,instruction,op,width,param_name,param,n,alignment,repeat,issue_ticks,completion_ticks,value");
    if (width == 4) run_phase_active_width<1>(write, repeats);
    else if (width == 8) run_phase_active_width<2>(write, repeats);
    else if (width == 16) run_phase_active_width<4>(write, repeats);
    else return 4;
  } else if (mode == "phase_matrix") {
    bw::line("suite,instruction,op,width,param_name,ref,probe,alignment,repeat,issue_ticks,completion_ticks,value");
    if (width == 4) run_phase_matrix_width<1>(write);
    else if (width == 8) run_phase_matrix_width<2>(write);
    else if (width == 16) run_phase_matrix_width<4>(write);
    else return 4;
  } else if (mode == "core" || mode == "profile") {
    bw::line("suite,instruction,op,width,param_name,param,n,alignment_or_steps,repeat,issue_or_dep_ticks,completion_or_baseline_ticks,value");
    const int reps = mode == "profile" ? 1 : repeats;
    if (width == 4) run_core_width<1>(reps, conflict);
    else if (width == 8) run_core_width<2>(reps, conflict);
    else if (width == 12) run_core_width<3>(reps, conflict);
    else if (width == 16) run_core_width<4>(reps, conflict);
    else return 4;
  } else if (mode == "alignment") {
    bw::line("suite,instruction,op,width,param_name,param,n,alignment,repeat,issue_ticks,completion_ticks,value");
    if (width == 8) run_alignment_width<2>(repeats);
    else if (width == 12) run_alignment_width<3>(repeats);
    else if (width == 16) run_alignment_width<4>(repeats);
    else return 4;
  } else if (mode == "throughput") {
    const int blocks = arg_int(argc, argv, "--blocks", 80);
    const int threads = arg_int(argc, argv, "--threads", 64);
    const int iters = arg_int(argc, argv, "--iters", 4096);
    const int c = conflict ? conflict : 1;
    bw::line("suite,instruction,op,width,param_name,conflict,blocks,threads,iters,repeat,event_ms,GBps,bytes");
    if (width == 4 && write) run_throughput_case<1, true>(blocks, threads, c, iters, repeats);
    else if (width == 4) run_throughput_case<1, false>(blocks, threads, c, iters, repeats);
    else if (width == 8 && write) run_throughput_case<2, true>(blocks, threads, c, iters, repeats);
    else if (width == 8) run_throughput_case<2, false>(blocks, threads, c, iters, repeats);
    else if (width == 16 && write) run_throughput_case<4, true>(blocks, threads, c, iters, repeats);
    else if (width == 16) run_throughput_case<4, false>(blocks, threads, c, iters, repeats);
    else return 4;
  } else {
    std::fprintf(stderr, "Usage: %s --mode baseline|bank_stride|bank_byte_stride|phase|phase_active|phase_matrix|core|alignment|throughput|profile [--width 4|8|12|16] [--op read|write] [--conflict N] [--repeats N]\n", argv[0]);
    return 1;
  }
  HIP_CHECK(hipDeviceSynchronize());
  return bw::finish("lds_core");
}
