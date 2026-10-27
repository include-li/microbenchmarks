/* BW1000 single-file HIP program: sync/empty_wait
 * Compile: hipcc -O3 -std=c++17 --offload-arch=gfx936 empty_wait.hip -o bench
 * Run: ./bench 
 * Minimal run (NOT a timing reproduction): ./bench 
 * No local headers, private libraries, scripts, data files or other source files required.
 * Only public HIP/DTK, its driver/runtime, and the C++ standard library are required.
 * Prints in-process summaries in timer tick and calibrated ns. BW1000_RAW=1 also prints samples.
 * Original measurement core: @empty; SHA256=1a79f91038d9b2bdbeb33690325f5a0fe5661dea0cbf2314030511e0851ce080
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
#include <initializer_list>
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
    if(family=="empty_wait") {
      t="s_waitcnt "+str(r,"counter");s=str(r,"method");
      if(s=="event_wait"||s=="event_control")batches[{t,"threshold="+str(r,"threshold"),s}][int(num(r,"rounds"))].push_back(num(r,"event_ms")*1e6);
      else {put(s,"observed bracket, NOT bare wait latency",num(r,"ticks"));if(s=="clock_drained")batches[{t,"threshold="+str(r,"threshold"),s}][int(num(r,"count"))].push_back(num(r,"ticks"));}
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
#define CHECK(x) do {hipError_t e=(x);if(e!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));std::exit(1);}}while(0)
struct Sample{uint64_t t[5];uint32_t value;};
template<bool VM,int K> __device__ __forceinline__ void wait_op(){
 if constexpr(VM)asm volatile("s_waitcnt vmcnt(%0)"::"n"(K):"memory");
 else asm volatile("s_waitcnt lgkmcnt(%0)"::"n"(K):"memory");
 return;
}
template<bool VM,int K> __global__ void legacy_probe(Sample*out){
 uint64_t a=clock64(),b=clock64();wait_op<VM,K>();uint64_t c=clock64();wait_op<VM,0>();uint64_t d=clock64(),e=clock64();
 if(threadIdx.x==0)*out=Sample{{a,b,c,d,e},0};
 return;
}
template<bool VM,int K,int N> __global__ void drained_probe(Sample*out){
 uint64_t a=clock64();wait_op<false,0>();
if constexpr(VM)asm volatile(".rept %0\n s_waitcnt vmcnt(%1)\n .endr"::"n"(N),"n"(K):"memory");
 else asm volatile(".rept %0\n s_waitcnt lgkmcnt(%1)\n .endr"::"n"(N),"n"(K):"memory");
 uint64_t b=clock64();wait_op<false,0>();
 if(threadIdx.x==0)*out=Sample{{a,b,0,0,0},0};
 return;
}
template<bool VM,int K,bool DoWait> __global__ void event_probe(Sample*out,unsigned rounds){
 uint32_t x=0x1234u+threadIdx.x;
#pragma clang loop unroll(disable)
 for(unsigned r=0;r<rounds;++r){
  if constexpr(DoWait){
if constexpr(VM)asm volatile(".rept 256\n s_waitcnt vmcnt(%0)\n .endr"::"n"(K):"memory");
   else asm volatile(".rept 256\n s_waitcnt lgkmcnt(%0)\n .endr"::"n"(K):"memory");
  }
  x^=r;asm volatile("":"+v"(x)::"memory");
 }
 if(threadIdx.x==0)out->value=x;
 return;
}
template<bool VM,int K,int N> void run_drained(Sample*d){
 for(int r=-5;r<30;++r){drained_probe<VM,K,N><<<1,64>>>(d);CHECK(hipGetLastError());CHECK(hipDeviceSynchronize());Sample h{};CHECK(hipMemcpy(&h,d,sizeof(h),hipMemcpyDeviceToHost));if(r>=0)bw::emit("clock_drained,%s,%d,%d,0,%d,%llu,0,0\n",VM?"vmcnt":"lgkmcnt",K,N,r,(unsigned long long)(h.t[1]-h.t[0]));}
}
template<bool VM,int K> void run_group(Sample*d){
 for(int r=-5;r<30;++r){legacy_probe<VM,K><<<1,64>>>(d);CHECK(hipGetLastError());CHECK(hipDeviceSynchronize());Sample h{};CHECK(hipMemcpy(&h,d,sizeof(h),hipMemcpyDeviceToHost));if(r>=0){bw::emit("legacy_clock_gap,%s,%d,0,0,%d,%llu,0,0\n",VM?"vmcnt":"lgkmcnt",K,r,(unsigned long long)(h.t[1]-h.t[0]));bw::emit("legacy_selected_wait,%s,%d,1,0,%d,%llu,0,0\n",VM?"vmcnt":"lgkmcnt",K,r,(unsigned long long)(h.t[2]-h.t[1]));}}
 run_drained<VM,K,0>(d);run_drained<VM,K,1>(d);run_drained<VM,K,2>(d);run_drained<VM,K,4>(d);run_drained<VM,K,8>(d);run_drained<VM,K,16>(d);run_drained<VM,K,64>(d);
 hipEvent_t a,b;CHECK(hipEventCreate(&a));CHECK(hipEventCreate(&b));
 for(unsigned rounds: {32u,128u,512u,2048u}){
  for(int r=-5;r<30;++r){
   for(int kind=0;kind<2;++kind){CHECK(hipEventRecord(a));if(kind)event_probe<VM,K,true><<<1,64>>>(d,rounds);else event_probe<VM,K,false><<<1,64>>>(d,rounds);CHECK(hipGetLastError());CHECK(hipEventRecord(b));CHECK(hipEventSynchronize(b));float ms=0;CHECK(hipEventElapsedTime(&ms,a,b));Sample h{};CHECK(hipMemcpy(&h,d,sizeof(h),hipMemcpyDeviceToHost));if(h.value!=0x1234u){std::fprintf(stderr,"event probe output mismatch %u\n",h.value);std::exit(2);}if(r>=0)bw::emit("%s,%s,%d,256,%u,%d,0,%.9f,%u\n",kind?"event_wait":"event_control",VM?"vmcnt":"lgkmcnt",K,rounds,r,ms,h.value);}
  }
 }
 CHECK(hipEventDestroy(a));CHECK(hipEventDestroy(b));
}
int main(){
  bw::start();
int n=0;CHECK(hipGetDeviceCount(&n));if(n<1){std::fprintf(stderr,"expected at least one visible GPU, got %d\n",n);return 3;}char pci[64];CHECK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));std::fprintf(stderr,"gpu_pci=%s; diagnostic only, no main benchmark data replaced\n",pci);Sample*d=nullptr;CHECK(hipMalloc(&d,sizeof(Sample)));bw::line("method,counter,threshold,count,rounds,repeat,ticks,event_ms,value");run_group<false,0>(d);run_group<false,1>(d);run_group<false,2>(d);run_group<false,15>(d);run_group<true,0>(d);run_group<true,32>(d);CHECK(hipFree(d));return bw::finish("empty_wait");}
