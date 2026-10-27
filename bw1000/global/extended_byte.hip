/* BW1000 single-file HIP program: global/extended_byte
 * Compile: hipcc -O3 -std=c++17 --offload-arch=gfx936 extended_byte.hip -o bench
 * Run: ./bench 30
 * Minimal run (NOT a timing reproduction): ./bench 1
 * No local headers, private libraries, scripts, data files or other source files required.
 * Only public HIP/DTK, its driver/runtime, and the C++ standard library are required.
 * Prints in-process summaries in timer tick and calibrated ns. BW1000_RAW=1 also prints samples.
 * Original measurement core: LDS/src/global_extended.cpp; SHA256=2e927ce9c3e5eb43da4798931b3d156c22df142c4a8bffba03a6fa10ab23eb6f
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
    if(family=="global_extended") {if(s!="stream")put(s,"original operation count",num(r,"cycles_per_operation"));else std::printf("BANDWIDTH,%s,%.9f,GB/s\n",t.c_str(),num(r,"bandwidth_GBps"));continue;}
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

#ifndef GMEM_BYTES
#define GMEM_BYTES 1
#endif

using u32x3 = uint32_t __attribute__((ext_vector_type(3)));

#define HIP_CHECK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) {    \
  std::fprintf(stderr, "HIP error %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
  std::exit(2); } } while (0)

#if GMEM_BYTES == 1
using Value = uint32_t;
constexpr const char* kLoadName = "global_load_ubyte";
constexpr const char* kStoreName = "global_store_byte";
__device__ __forceinline__ Value load_default(const void* p) { Value v; uint64_t a=reinterpret_cast<uint64_t>(p); asm volatile("global_load_ubyte %0, %1, off" : "=v"(v) : "v"(a) : "memory"); return v; }
__device__ __forceinline__ Value load_glc(const void* p) { Value v; uint64_t a=reinterpret_cast<uint64_t>(p); asm volatile("global_load_ubyte %0, %1, off glc" : "=v"(v) : "v"(a) : "memory"); return v; }
__device__ __forceinline__ void store_value(void* p, Value v) { uint64_t a=reinterpret_cast<uint64_t>(p); asm volatile("global_store_byte %0, %1, off" :: "v"(a), "v"(v) : "memory"); }
#elif GMEM_BYTES == 2
using Value = uint32_t;
constexpr const char* kLoadName = "global_load_ushort";
constexpr const char* kStoreName = "global_store_short";
__device__ __forceinline__ Value load_default(const void* p) { Value v; uint64_t a=reinterpret_cast<uint64_t>(p); asm volatile("global_load_ushort %0, %1, off" : "=v"(v) : "v"(a) : "memory"); return v; }
__device__ __forceinline__ Value load_glc(const void* p) { Value v; uint64_t a=reinterpret_cast<uint64_t>(p); asm volatile("global_load_ushort %0, %1, off glc" : "=v"(v) : "v"(a) : "memory"); return v; }
__device__ __forceinline__ void store_value(void* p, Value v) { uint64_t a=reinterpret_cast<uint64_t>(p); asm volatile("global_store_short %0, %1, off" :: "v"(a), "v"(v) : "memory"); }
#elif GMEM_BYTES == 12
using Value = u32x3;
constexpr const char* kLoadName = "global_load_dwordx3";
constexpr const char* kStoreName = "global_store_dwordx3";
__device__ __forceinline__ Value load_default(const void* p) { Value v; uint64_t a=reinterpret_cast<uint64_t>(p); asm volatile("global_load_dwordx3 %0, %1, off" : "=v"(v) : "v"(a) : "memory"); return v; }
__device__ __forceinline__ Value load_glc(const void* p) { Value v; uint64_t a=reinterpret_cast<uint64_t>(p); asm volatile("global_load_dwordx3 %0, %1, off glc" : "=v"(v) : "v"(a) : "memory"); return v; }
__device__ __forceinline__ void store_value(void* p, Value v) { uint64_t a=reinterpret_cast<uint64_t>(p); asm volatile("global_store_dwordx3 %0, %1, off" :: "v"(a), "v"(v) : "memory"); }
#else
#error GMEM_BYTES must be 1, 2, or 12
#endif

struct TickResult { uint64_t issue, completion, baseline; uint32_t value, pad; };

__device__ __forceinline__ uint32_t fold(Value v) {
#if GMEM_BYTES == 12
  return v[0] ^ v[1] ^ v[2];
#else
  return v;
#endif
}

__device__ __forceinline__ Value make_value(uint32_t x) {
#if GMEM_BYTES == 12
  return {x, x + 1, x + 2};
#else
  return x;
#endif
}

template <bool Glc>
__device__ __forceinline__ Value load_policy(const void* p) {
  if constexpr (Glc) return load_glc(p);
  return load_default(p);
}

template <bool Glc>
__global__ void chase_kernel(const uint8_t* data, uint32_t nodes, int steps,
                             TickResult* out) {
  if (blockIdx.x || threadIdx.x) return;
  uint32_t index = 0, sink = 0;
  const uint32_t mask = nodes - 1;
  const uint64_t t0 = clock64();
#pragma clang loop unroll(disable)
  for (int i = 0; i < steps; ++i) {
    Value v = load_policy<Glc>(data + static_cast<size_t>(index) * 64u);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    uint32_t delta = fold(v) | 1u;
    index = (index + delta) & mask;
    sink ^= index + static_cast<uint32_t>(i);
  }
  const uint64_t t1 = clock64();
  out[0] = {t1 - t0, t1 - t0, 0, sink ^ index, 0};
}

__global__ void chase_baseline(uint32_t nodes, int steps, TickResult* out) {
  if (blockIdx.x || threadIdx.x) return;
  uint32_t index = 0, sink = 0, mask = nodes - 1;
  const uint64_t t0 = clock64();
#pragma clang loop unroll(disable)
  for (int i = 0; i < steps; ++i) { index = (index + 1u) & mask; sink ^= index + static_cast<uint32_t>(i); }
  const uint64_t t1 = clock64();
  out[0] = {t1 - t0, t1 - t0, 0, sink ^ index, 0};
}

#define DO_N(N, STMT) do { if ((N)>0) STMT(0); if ((N)>1) STMT(1); if ((N)>2) STMT(2); if ((N)>3) STMT(3); if ((N)>4) STMT(4); if ((N)>5) STMT(5); if ((N)>6) STMT(6); if ((N)>7) STMT(7); if ((N)>8) STMT(8); if ((N)>9) STMT(9); if ((N)>10) STMT(10); if ((N)>11) STMT(11); if ((N)>12) STMT(12); if ((N)>13) STMT(13); if ((N)>14) STMT(14); if ((N)>15) STMT(15); } while (0)

template <int N, bool Store, bool Glc>
__global__ void issue_kernel(uint8_t* data, int rounds, TickResult* out) {
  if (blockIdx.x || threadIdx.x >= 64) return;
  uint32_t lane = threadIdx.x;
  Value values[16]{};
  Value input = make_value(lane + 1);
  uint32_t sink = lane;
  uint64_t issue = 0, complete = 0;
#pragma clang loop unroll(disable)
  for (int it = 0; it < rounds; ++it) {
    uint64_t t0 = clock64();
    if constexpr (Store) {
#define STORE_ONE(j) store_value(data + (static_cast<size_t>(j) * 64u + lane) * GMEM_BYTES, input)
      DO_N(N, STORE_ONE);
#undef STORE_ONE
    } else {
#define LOAD_ONE(j) values[j] = load_policy<Glc>(data + (static_cast<size_t>(j) * 64u + lane) * GMEM_BYTES)
      DO_N(N, LOAD_ONE);
#undef LOAD_ONE
    }
    uint64_t t1 = clock64();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    uint64_t t2 = clock64();
    issue += t1 - t0; complete += t2 - t0;
    if constexpr (!Store) {
#define FOLD_ONE(j) sink ^= fold(values[j]) + static_cast<uint32_t>(it)
      DO_N(N, FOLD_ONE);
#undef FOLD_ONE
    }
  }
  if (lane == 0) out[0] = {issue, complete, 0, sink, 0};
}

template <bool Store>
__global__ void stream_kernel(uint8_t* data, size_t items, int items_per_thread,
                              uint32_t* sink_out) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  uint32_t sink = static_cast<uint32_t>(tid);
  Value input = make_value(sink + 1);
#pragma clang loop unroll(disable)
  for (int i = 0; i < items_per_thread; ++i) {
    size_t item = tid + static_cast<size_t>(i) * stride;
    if (item >= items) break;
    if constexpr (Store) store_value(data + item * GMEM_BYTES, input);
    else {
      Value v = load_default(data + item * GMEM_BYTES);
      // Inline assembly does not inform the scheduler that the VGPR is pending.
      // Consume only after VM completion, before address/output VGPR reuse.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      sink ^= fold(v);
    }
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  if (threadIdx.x == 0) sink_out[blockIdx.x] = sink;
}

template <class F>
static float event_launch(F launch) {
  hipEvent_t a,b; HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b));
  HIP_CHECK(hipEventRecord(a)); launch(); HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventRecord(b)); HIP_CHECK(hipEventSynchronize(b));
  float ms=0; HIP_CHECK(hipEventElapsedTime(&ms,a,b)); HIP_CHECK(hipEventDestroy(b)); HIP_CHECK(hipEventDestroy(a)); return ms;
}

static TickResult copy_result(TickResult* d) { TickResult h{}; HIP_CHECK(hipMemcpy(&h,d,sizeof(h),hipMemcpyDeviceToHost)); return h; }

static void pollute(uint32_t* p, size_t words) {
  HIP_CHECK(hipMemset(p, 0x5a, words * 4));
  HIP_CHECK(hipDeviceSynchronize());
}

template <int N, bool Store, bool Glc>
static float issue_launch(uint8_t* data, int rounds, TickResult* out) {
  return event_launch([&] { hipLaunchKernelGGL((issue_kernel<N,Store,Glc>),dim3(1),dim3(64),0,0,data,rounds,out); });
}

template <bool Store, bool Glc>
static float issue_dispatch(int n, uint8_t* data, int rounds, TickResult* out) {
  if (n==1) return issue_launch<1,Store,Glc>(data,rounds,out);
  if (n==4) return issue_launch<4,Store,Glc>(data,rounds,out);
  if (n==8) return issue_launch<8,Store,Glc>(data,rounds,out);
  return issue_launch<16,Store,Glc>(data,rounds,out);
}

int main(int argc, char** argv) {
  bw::start();

  int repeats = argc > 1 ? std::atoi(argv[1]) : 30;
  constexpr size_t l2_bytes = 1u*1024u*1024u;
  constexpr size_t hbm_bytes = 32u*1024u*1024u;
  constexpr size_t stream_bytes = 256u*1024u*1024u;
  uint8_t *l2=nullptr,*hbm=nullptr,*stream=nullptr; uint32_t *pollution=nullptr,*sink=nullptr; TickResult* result=nullptr;
  HIP_CHECK(hipMalloc(&l2,l2_bytes)); HIP_CHECK(hipMalloc(&hbm,hbm_bytes)); HIP_CHECK(hipMalloc(&stream,stream_bytes));
  HIP_CHECK(hipMalloc(&pollution,64u*1024u*1024u)); HIP_CHECK(hipMalloc(&sink,1280*sizeof(uint32_t))); HIP_CHECK(hipMalloc(&result,sizeof(TickResult)));
  HIP_CHECK(hipMemset(l2,1,l2_bytes)); HIP_CHECK(hipMemset(hbm,1,hbm_bytes)); HIP_CHECK(hipMemset(stream,1,stream_bytes));
  bw::emit("suite,operation,instruction,width_bytes,l2_state,access_pattern,independent_instructions,repeat,ticks,baseline_ticks,adjusted_ticks,cycles_per_operation,event_ms,bandwidth_GBps\n");
  const int steps=16384;
  for (int state=0; state<2; ++state) {
    uint8_t* data=state? hbm:l2; size_t bytes=state?hbm_bytes:l2_bytes; uint32_t nodes=static_cast<uint32_t>(bytes/64);
    for (int r=0;r<repeats;++r) {
      if (state) pollute(pollution,16u*1024u*1024u);
      else { HIP_CHECK(hipMemset(l2,1,l2_bytes)); HIP_CHECK(hipDeviceSynchronize()); }
      hipLaunchKernelGGL(chase_baseline,dim3(1),dim3(64),0,0,nodes,steps,result); HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize()); TickResult base=copy_result(result);
      float ms;
      if (state) ms=event_launch([&]{hipLaunchKernelGGL((chase_kernel<false>),dim3(1),dim3(64),0,0,data,nodes,steps,result);});
      else ms=event_launch([&]{hipLaunchKernelGGL((chase_kernel<true>),dim3(1),dim3(64),0,0,data,nodes,steps,result);});
      TickResult h=copy_result(result); uint64_t adjusted=h.completion>base.completion?h.completion-base.completion:0;
      bw::emit("latency,load,%s,%d,%s,dependent,1,%d,%llu,%llu,%llu,%.9f,%.6f,0\n",kLoadName,GMEM_BYTES,state?"cold":"hot",r,(unsigned long long)h.completion,(unsigned long long)base.completion,(unsigned long long)adjusted,static_cast<double>(adjusted)/steps,ms);
    }
  }
  for (int n : {1,4,8,16}) for (int r=0;r<repeats;++r) {
    float ms=issue_dispatch<false,true>(n,l2,4096,result); TickResult h=copy_result(result);
    bw::emit("issue,load,%s,%d,hot,coalesced,%d,%d,%llu,0,%llu,%.9f,%.6f,0\n",kLoadName,GMEM_BYTES,n,r,(unsigned long long)h.issue,(unsigned long long)h.issue,static_cast<double>(h.issue)/(4096*n),ms);
    ms=issue_dispatch<true,false>(n,l2,4096,result); h=copy_result(result);
    bw::emit("issue,store,%s,%d,hot,coalesced,%d,%d,%llu,0,%llu,%.9f,%.6f,0\n",kStoreName,GMEM_BYTES,n,r,(unsigned long long)h.issue,(unsigned long long)h.issue,static_cast<double>(h.issue)/(4096*n),ms);
    bw::emit("completion,store,%s,%d,hot,coalesced,%d,%d,%llu,0,%llu,%.9f,%.6f,0\n",kStoreName,GMEM_BYTES,n,r,(unsigned long long)h.completion,(unsigned long long)h.completion,static_cast<double>(h.completion)/(4096*n),ms);
    pollute(pollution,16u*1024u*1024u);
    ms=issue_dispatch<false,false>(n,hbm,4096,result); h=copy_result(result);
    bw::emit("issue,load,%s,%d,cold,coalesced,%d,%d,%llu,0,%llu,%.9f,%.6f,0\n",kLoadName,GMEM_BYTES,n,r,(unsigned long long)h.issue,(unsigned long long)h.issue,static_cast<double>(h.issue)/(4096*n),ms);
    ms=issue_dispatch<true,false>(n,hbm,4096,result); h=copy_result(result);
    bw::emit("issue,store,%s,%d,cold,coalesced,%d,%d,%llu,0,%llu,%.9f,%.6f,0\n",kStoreName,GMEM_BYTES,n,r,(unsigned long long)h.issue,(unsigned long long)h.issue,static_cast<double>(h.issue)/(4096*n),ms);
    bw::emit("completion,store,%s,%d,cold,coalesced,%d,%d,%llu,0,%llu,%.9f,%.6f,0\n",kStoreName,GMEM_BYTES,n,r,(unsigned long long)h.completion,(unsigned long long)h.completion,static_cast<double>(h.completion)/(4096*n),ms);
  }
  const int grid=1280,block=256; size_t items=stream_bytes/GMEM_BYTES; int per_thread=static_cast<int>((items+static_cast<size_t>(grid*block)-1)/(grid*block));
  for (int r=0;r<repeats;++r) {
    float ms=event_launch([&]{hipLaunchKernelGGL((stream_kernel<false>),dim3(grid),dim3(block),0,0,stream,items,per_thread,sink);});
    // Independent host replay of all lane-0 reduction outputs, outside timing.
    std::vector<uint32_t> observed(grid);
    HIP_CHECK(hipMemcpy(observed.data(),sink,grid*sizeof(uint32_t),hipMemcpyDeviceToHost));
    size_t validation_errors=0;
    for (int b=0;b<grid;++b) {
      uint32_t tid=static_cast<uint32_t>(b*block), expected=tid;
      for (size_t item=tid;item<items;item+=static_cast<size_t>(grid*block)) {
        uint32_t word;
        if (r==0) {
#if GMEM_BYTES == 1
          word=0x01u;
#elif GMEM_BYTES == 2
          word=0x0101u;
#else
          word=0x01010101u;
#endif
        } else {
#if GMEM_BYTES == 1
          word=(tid+1u)&0xffu;
#elif GMEM_BYTES == 2
          word=(tid+1u)&0xffffu;
#else
          word=(tid+1u)^(tid+2u)^(tid+3u);
#endif
        }
        expected ^= word;
      }
      validation_errors += observed[b]!=expected;
    }
    if (validation_errors) { std::fprintf(stderr,"VALIDATION stream_load errors=%zu\n",validation_errors); return 3; }
    double gbps=static_cast<double>(items*GMEM_BYTES)/(ms*1.0e6);
    bw::emit("stream,load,%s,%d,streaming,coalesced,0,%d,0,0,0,0,%.6f,%.6f\n",kLoadName,GMEM_BYTES,r,ms,gbps);
    ms=event_launch([&]{hipLaunchKernelGGL((stream_kernel<true>),dim3(grid),dim3(block),0,0,stream,items,per_thread,sink);});
    gbps=static_cast<double>(items*GMEM_BYTES)/(ms*1.0e6);
    bw::emit("stream,store,%s,%d,streaming,coalesced,0,%d,0,0,0,0,%.6f,%.6f\n",kStoreName,GMEM_BYTES,r,ms,gbps);
  }
  // Verify the final store contents for every transferred byte, outside timing.
  std::vector<uint8_t> stored(items*GMEM_BYTES);
  HIP_CHECK(hipMemcpy(stored.data(),stream,stored.size(),hipMemcpyDeviceToHost));
  size_t store_errors=0;
  for (size_t item=0;item<items;++item) {
    uint32_t x=static_cast<uint32_t>(item%static_cast<size_t>(grid*block))+1u;
    for (int byte=0;byte<GMEM_BYTES;++byte) {
      uint32_t word=x+static_cast<uint32_t>(byte/4);
      uint8_t expected=static_cast<uint8_t>(word>>(8*(byte%4)));
      store_errors += stored[item*GMEM_BYTES+byte]!=expected;
    }
  }
  std::fprintf(stderr,"VALIDATION stream_load checks=%d stream_store_bytes=%zu errors=%zu\n",repeats*grid,stored.size(),store_errors);
  if (store_errors) return 3;
  HIP_CHECK(hipFree(result)); HIP_CHECK(hipFree(sink)); HIP_CHECK(hipFree(pollution)); HIP_CHECK(hipFree(stream)); HIP_CHECK(hipFree(hbm)); HIP_CHECK(hipFree(l2));
  return bw::finish("global_extended");
}
