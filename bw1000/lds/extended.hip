/* BW1000 single-file HIP program: lds/extended
 * Compile: hipcc -O2 -std=c++17 --offload-arch=gfx936 extended.hip -o bench
 * Run: ./bench --repeats 30
 * Minimal run (NOT a timing reproduction): ./bench --repeats 1
 * No local headers, private libraries, scripts, data files or other source files required.
 * Only public HIP/DTK, its driver/runtime, and the C++ standard library are required.
 * Prints in-process summaries in timer tick and calibrated ns. BW1000_RAW=1 also prints samples.
 * Original measurement core: vgpr/src/lds_extended_bench.cpp; SHA256=98f92e1e0508ad72026fd82470e4ad7267962f048754c409598855534aa232f9
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
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));

#define HIP_CHECK(call) do { hipError_t e_=(call); if(e_!=hipSuccess){ \
  std::fprintf(stderr,"HIP error %s:%d: %s -> %s\n",__FILE__,__LINE__,#call,hipGetErrorString(e_)); std::exit(2);} } while(0)

enum Kind : int {
  ReadU8, ReadI8, ReadU16, ReadI16, WriteB8, WriteB16,
  Read2B32, Read2B64, Read2St64B32, Read2St64B64,
  Write2B32, Write2B64, Write2St64B32, Write2St64B64
};

template<Kind K> struct KType { using type = uint32_t; };
template<> struct KType<Read2B32> { using type = u32x2; };
template<> struct KType<Read2B64> { using type = u32x4; };
template<> struct KType<Read2St64B32> { using type = u32x2; };
template<> struct KType<Read2St64B64> { using type = u32x4; };
template<> struct KType<Write2B32> { using type = u32x2; };
template<> struct KType<Write2B64> { using type = u32x4; };
template<> struct KType<Write2St64B32> { using type = u32x2; };
template<> struct KType<Write2St64B64> { using type = u32x4; };
template<Kind K> using KV = typename KType<K>::type;

struct Result { uint64_t issue_ticks, completion_ticks; uint32_t value, pad; };

template<Kind K>
__device__ __forceinline__ KV<K> make_value(uint32_t x) {
  if constexpr (K == Read2B32 || K == Read2St64B32 || K == Write2B32 || K == Write2St64B32)
    return KV<K>{x, x + 1u};
  else if constexpr (K == Read2B64 || K == Read2St64B64 || K == Write2B64 || K == Write2St64B64)
    return KV<K>{x, x + 1u, x + 2u, x + 3u};
  else return x;
}

template<Kind K>
__device__ __forceinline__ uint32_t sum_value(KV<K> v) {
  if constexpr (K == Read2B32 || K == Read2St64B32 || K == Write2B32 || K == Write2St64B32)
    return v[0] + v[1];
  else if constexpr (K == Read2B64 || K == Read2St64B64 || K == Write2B64 || K == Write2St64B64)
    return v[0] + v[1] + v[2] + v[3];
  else return v;
}

template<Kind K>
__device__ __forceinline__ KV<K> execute_op(uint32_t addr, KV<K> v) {
  if constexpr (K == ReadU8) asm volatile("ds_read_u8 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  else if constexpr (K == ReadI8) asm volatile("ds_read_i8 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  else if constexpr (K == ReadU16) asm volatile("ds_read_u16 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  else if constexpr (K == ReadI16) asm volatile("ds_read_i16 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  else if constexpr (K == WriteB8) asm volatile("ds_write_b8 %0, %1" :: "v"(addr), "v"(v) : "memory");
  else if constexpr (K == WriteB16) asm volatile("ds_write_b16 %0, %1" :: "v"(addr), "v"(v) : "memory");
  else if constexpr (K == Read2B32) asm volatile("ds_read2_b32 %0, %1 offset0:0 offset1:1" : "=v"(v) : "v"(addr) : "memory");
  else if constexpr (K == Read2B64) asm volatile("ds_read2_b64 %0, %1 offset0:0 offset1:1" : "=v"(v) : "v"(addr) : "memory");
  else if constexpr (K == Read2St64B32) asm volatile("ds_read2st64_b32 %0, %1 offset0:0 offset1:1" : "=v"(v) : "v"(addr) : "memory");
  else if constexpr (K == Read2St64B64) asm volatile("ds_read2st64_b64 %0, %1 offset0:0 offset1:1" : "=v"(v) : "v"(addr) : "memory");
  else if constexpr (K == Write2B32) asm volatile("ds_write2_b32 %0, %1, %2 offset0:0 offset1:1" :: "v"(addr), "v"(v[0]), "v"(v[1]) : "memory");
  else if constexpr (K == Write2B64) {
    u32x2 a={v[0],v[1]}, b={v[2],v[3]};
    asm volatile("ds_write2_b64 %0, %1, %2 offset0:0 offset1:1" :: "v"(addr), "v"(a), "v"(b) : "memory");
  } else if constexpr (K == Write2St64B32) asm volatile("ds_write2st64_b32 %0, %1, %2 offset0:0 offset1:1" :: "v"(addr), "v"(v[0]), "v"(v[1]) : "memory");
  else if constexpr (K == Write2St64B64) {
    u32x2 a={v[0],v[1]}, b={v[2],v[3]};
    asm volatile("ds_write2st64_b64 %0, %1, %2 offset0:0 offset1:1" :: "v"(addr), "v"(a), "v"(b) : "memory");
  }
  return v;
}

template<Kind K>
__host__ __device__ __forceinline__ constexpr uint32_t footprint_banks() {
  if constexpr (K == Read2B32 || K == Write2B32) return 2;
  if constexpr (K == Read2B64 || K == Write2B64) return 4;
  if constexpr (K == Read2St64B64 || K == Write2St64B64) return 2;
  return 1;
}

template<Kind K>
__device__ __forceinline__ uint32_t conflict_addr(uint32_t lane, uint32_t degree, uint32_t align) {
  const uint32_t footprint = footprint_banks<K>();
  uint32_t slots = 32u / (footprint * degree);
  if (!slots) slots = 1;
  const uint32_t word = (lane % slots) * footprint + 32u * (lane / slots + 1u);
  return word * 4u + align;
}

template<Kind K>
__global__ void completion_kernel(uint32_t degree, uint32_t align, Result* out) {
  __shared__ uint32_t lds[16384];
  const uint32_t lane = threadIdx.x & 63u;
  for (uint32_t i = lane; i < 8192u; i += 64u) lds[i] = i + 1u;
  __syncthreads();
  KV<K> v = make_value<K>(lane + 1u);
  const uint32_t addr = conflict_addr<K>(lane, degree, align);
  const uint64_t t0 = clock64();
  v = execute_op<K>(addr, v);
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  const uint64_t t1 = clock64();
  if (lane == 0) out[0] = Result{t1-t0,t1-t0,sum_value<K>(v),0};
}

template<Kind K, int N>
__global__ void issue_kernel(uint32_t degree, uint32_t align, Result* out) {
  __shared__ uint32_t lds[16384];
  const uint32_t lane = threadIdx.x & 63u;
  for (uint32_t i = lane; i < 8192u; i += 64u) lds[i] = i + 1u;
  __syncthreads();
  KV<K> vals[N];
#pragma unroll
  for (int i=0;i<N;++i) vals[i]=make_value<K>(lane+i+1u);
  const uint32_t addr=conflict_addr<K>(lane,degree,align);
  const uint64_t t0=clock64();
#pragma unroll
  for(int i=0;i<N;++i) vals[i]=execute_op<K>(addr+static_cast<uint32_t>(i)*4096u,vals[i]);
  const uint64_t ti=clock64();
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  const uint64_t td=clock64();
  uint32_t sink=0;
#pragma unroll
  for(int i=0;i<N;++i) sink+=sum_value<K>(vals[i]);
  if(lane==0) out[0]=Result{ti-t0,td-t0,sink,0};
}

static const char* kind_name(Kind k) {
  static const char* names[]={"ds_read_u8","ds_read_i8","ds_read_u16","ds_read_i16","ds_write_b8","ds_write_b16",
    "ds_read2_b32","ds_read2_b64","ds_read2st64_b32","ds_read2st64_b64","ds_write2_b32","ds_write2_b64","ds_write2st64_b32","ds_write2st64_b64"};
  return names[static_cast<int>(k)];
}
static int kind_bytes(Kind k) {
  if(k==ReadU8||k==ReadI8||k==WriteB8) return 1;
  if(k==ReadU16||k==ReadI16||k==WriteB16) return 2;
  if(k==Read2B32||k==Read2St64B32||k==Write2B32||k==Write2St64B32) return 8;
  return 16;
}

template<Kind K>
static Result collect_completion(int degree,int align,Result* d){
  hipLaunchKernelGGL((completion_kernel<K>),dim3(1),dim3(64),0,0,degree,align,d);
  HIP_CHECK(hipGetLastError()); Result h{}; HIP_CHECK(hipMemcpy(&h,d,sizeof(h),hipMemcpyDeviceToHost)); return h;
}
template<Kind K,int N>
static Result collect_issue_n(int degree,int align,Result* d){
  hipLaunchKernelGGL((issue_kernel<K,N>),dim3(1),dim3(64),0,0,degree,align,d);
  HIP_CHECK(hipGetLastError()); Result h{}; HIP_CHECK(hipMemcpy(&h,d,sizeof(h),hipMemcpyDeviceToHost)); return h;
}
template<Kind K>
static Result collect_issue(int n,int degree,int align,Result* d){
  if(n==1)return collect_issue_n<K,1>(degree,align,d); if(n==2)return collect_issue_n<K,2>(degree,align,d);
  if(n==4)return collect_issue_n<K,4>(degree,align,d); if(n==8)return collect_issue_n<K,8>(degree,align,d);
  return collect_issue_n<K,16>(degree,align,d);
}

template<Kind K>
static void run_kind(int repeats){
  Result* d=nullptr; HIP_CHECK(hipMalloc(&d,sizeof(Result)));
  for(int c: {1,2,4,8}){
    if(32/(static_cast<int>(footprint_banks<K>())*c)==0) continue;
    for(int r=0;r<repeats;++r){Result h=collect_completion<K>(c,0,d); bw::emit("completion,%s,%d,%d,1,0,%d,%llu,%llu,%u\n",kind_name(K),kind_bytes(K),c,r,(unsigned long long)h.issue_ticks,(unsigned long long)h.completion_ticks,h.value);}
    for(int n: {1,2,4,8,16}) for(int r=0;r<repeats;++r){Result h=collect_issue<K>(n,c,0,d); bw::emit("issue,%s,%d,%d,%d,0,%d,%llu,%llu,%u\n",kind_name(K),kind_bytes(K),c,n,r,(unsigned long long)h.issue_ticks,(unsigned long long)h.completion_ticks,h.value);}
  }
  for(int align: {0,1,2,3}) for(int r=0;r<repeats;++r){Result h=collect_completion<K>(1,align,d); bw::emit("alignment,%s,%d,1,1,%d,%d,%llu,%llu,%u\n",kind_name(K),kind_bytes(K),align,r,(unsigned long long)h.issue_ticks,(unsigned long long)h.completion_ticks,h.value);}
  HIP_CHECK(hipFree(d));
}

// The archived extended reducer subtracts the same LDS wait baseline as core.
__global__ void standalone_wait_baseline(Result*out){
  __shared__ uint32_t lds[16384];lds[threadIdx.x]=threadIdx.x;__syncthreads();
  const uint64_t a=clock64();asm volatile("s_waitcnt lgkmcnt(0)":::"memory");const uint64_t b=clock64();
  if(threadIdx.x==0)*out=Result{b-a,b-a,lds[0],0};
}
int main(int argc,char** argv){
  bw::start();

  int repeats=30; for(int i=1;i+1<argc;++i)if(!std::strcmp(argv[i],"--repeats"))repeats=std::atoi(argv[i+1]);
  bw::line("suite,instruction,width,conflict,n,alignment,repeat,issue_ticks,completion_ticks,value");
  Result* base_out=nullptr;HIP_CHECK(hipMalloc(&base_out,sizeof(Result)));
  for(int rep=0;rep<repeats;++rep){hipLaunchKernelGGL(standalone_wait_baseline,dim3(1),dim3(64),0,0,base_out);HIP_CHECK(hipGetLastError());Result h{};HIP_CHECK(hipMemcpy(&h,base_out,sizeof(h),hipMemcpyDeviceToHost));bw::emit("baseline,s_waitcnt_lgkmcnt0,0,0,0,0,%d,%llu,%llu,%u\n",rep,(unsigned long long)h.issue_ticks,(unsigned long long)h.completion_ticks,h.value);}
  HIP_CHECK(hipFree(base_out));
  bw::line("suite,instruction,width,conflict,n,alignment,repeat,issue_ticks,completion_ticks,value");
  run_kind<ReadU8>(repeats); run_kind<ReadI8>(repeats); run_kind<ReadU16>(repeats); run_kind<ReadI16>(repeats);
  run_kind<WriteB8>(repeats); run_kind<WriteB16>(repeats); run_kind<Read2B32>(repeats); run_kind<Read2B64>(repeats);
  run_kind<Read2St64B32>(repeats); run_kind<Read2St64B64>(repeats); run_kind<Write2B32>(repeats); run_kind<Write2B64>(repeats);
  run_kind<Write2St64B32>(repeats); run_kind<Write2St64B64>(repeats); HIP_CHECK(hipDeviceSynchronize()); return bw::finish("lds_extended");
}
