// Measures what each preset actually does, per profile and output type:
//   1. EQ of a centred (L=R) tone and side-vs-mid balance,
//   2. output level against input level (leveler / compressor / limiter),
//   3. voice-to-background change for a centred voice over a wide background.
// Build: clang++ -std=c++17 -O2 -arch arm64 -I core tools/preset_check.cpp core/spatial.cpp -o /tmp/preset_check
// Run:   /tmp/preset_check [0|1]   (1 = reverb on)
#include "spatial.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
#include <random>
using namespace ally;
const float SR=48000;
struct Cfg{const char* n; Profile p; DeviceKind d;};
// Render 'secs' of a generator; return output (interleaved).
template<class G> std::vector<float> render(Renderer& r, G gen, double secs){
  int N=int(secs*SR)/kHop*kHop; std::vector<float> in(2*kHop), out(2*N);
  for(int b=0;b<N/kHop;++b){ for(int i=0;i<kHop;++i){ float l,rr; gen(b*kHop+i,l,rr); in[2*i]=l; in[2*i+1]=rr;} r.process_interleaved(in.data(), &out[2*b*kHop], kHop, 2);} return out; }
double rms(const std::vector<float>& v,int from,int ch){ double s=0; int n=0; for(size_t i=2*from+ch;i<v.size();i+=2){s+=v[i]*v[i];++n;} return std::sqrt(s/n);}
double db(double x){return 20*std::log10(x+1e-12);}
static int eq_and_dynamics(int argc,char**argv){
  bool reverb = argc>1 && argv[1][0]=='1';
  Cfg cfgs[]={{"game  hp ",Profile::Game,DeviceKind::Headphones},{"movie hp ",Profile::Movie,DeviceKind::Headphones},{"music hp ",Profile::Music,DeviceKind::Headphones},{"speech hp",Profile::Speech,DeviceKind::Headphones},
              {"game  spk",Profile::Game,DeviceKind::Speakers},{"movie spk",Profile::Movie,DeviceKind::Speakers},{"music spk",Profile::Music,DeviceKind::Speakers},{"speech spk",Profile::Speech,DeviceKind::Speakers},
              {"music gl ",Profile::Music,DeviceKind::Glasses},{"speech gl",Profile::Speech,DeviceKind::Glasses}};
  const float freqs[]={40,80,160,315,630,1250,2500,5000,8000,12500,16000};
  std::printf("reverb=%d\nEQ (mid signal L=R, sine -26 dBFS, dB of output power vs input, relative to 1 kHz) | level@1k | side/mid@1k\n", reverb);
  std::printf("%-10s", "cfg"); for(float f:freqs) std::printf("%7.0f",f); std::printf("   1k_abs  side-mid\n");
  for(auto&c:cfgs){
    std::printf("%-10s",c.n); double ref=0; std::vector<double> vals;
    auto meas=[&](float f,bool side){ Renderer r; r.init(SR); r.set_profile(c.p); r.set_device(c.d); r.set_reverb(reverb); r.set_neural(false);
      float a=0.05f; auto out=render(r,[&](int n,float&l,float&rr){ float s=a*std::sin(2*M_PI*f*n/SR); l=s; rr=side?-s:s;},1.5);
      int from=int(1.0*SR); double o=std::hypot(rms(out,from,0),rms(out,from,1)); double in=a/std::sqrt(2)*std::sqrt(2); return db(o/in);} ;
    ref=meas(1000,false); for(float f:freqs) std::printf("%7.1f",meas(f,false)-ref);
    std::printf("  %7.1f  %7.1f\n", ref, meas(1000,true)-ref);
  }
  std::printf("\nDynamics: decorrelated pink-ish noise, input RMS dBFS -> output RMS dBFS (after 6 s)\n%-10s","cfg");
  const float lv[]={-40,-30,-20,-12,-6};
  for(float l:lv) std::printf("%8.0f",l); std::printf("   peak@-6\n");
  for(auto&c:cfgs){ std::printf("%-10s",c.n); double pk=0;
    for(float L:lv){ Renderer r; r.init(SR); r.set_profile(c.p); r.set_device(c.d); r.set_reverb(reverb);
      std::mt19937 rng(3); std::normal_distribution<float> nd(0,1); float bl[2][3]={{0}};
      float g=std::pow(10,L/20)/0.33f;
      auto out=render(r,[&](int,float&l,float&rr){ float w[2]={nd(rng),nd(rng)}; float o[2]; for(int k=0;k<2;++k){ bl[k][0]=0.99f*bl[k][0]+0.1f*w[k]; bl[k][1]=0.9f*bl[k][1]+0.3f*w[k]; o[k]=(bl[k][0]+bl[k][1]+0.3f*w[k]);} l=g*o[0]*0.25f; rr=g*o[1]*0.25f;},8);
      int from=6*SR; std::printf("%8.1f", db(std::sqrt((rms(out,from,0)*rms(out,from,0)+rms(out,from,1)*rms(out,from,1))/2)));
      if(L==-6) for(size_t i=0;i<out.size();++i) pk=std::max(pk,(double)std::fabs(out[i]));
    }
    std::printf("   %.3f\n",pk);
  }
  return 0;
}
static int voice_to_background(int argc,char**argv){
  bool reverb = argc>1 && argv[1][0]=='1';
  Cfg cfgs[]={{"game  hp ",Profile::Game,DeviceKind::Headphones},{"movie hp ",Profile::Movie,DeviceKind::Headphones},{"music hp ",Profile::Music,DeviceKind::Headphones},{"speech hp",Profile::Speech,DeviceKind::Headphones},
              {"game  spk",Profile::Game,DeviceKind::Speakers},{"movie spk",Profile::Movie,DeviceKind::Speakers},{"music spk",Profile::Music,DeviceKind::Speakers},{"speech spk",Profile::Speech,DeviceKind::Speakers},{"movie gl ",Profile::Movie,DeviceKind::Glasses}};
  const int N = int(8*SR)/kHop*kHop;
  // voice: speech-band noise with syllable envelope, centred. bg: uncorrelated stereo noise (wide music/effects)
  std::mt19937 rng(5); std::normal_distribution<float> nd(0,1);
  std::vector<float> v(N), bl(N), br(N);
  float a1=0,a2=0,b1=0,b2=0,c1=0,c2=0;
  for(int i=0;i<N;++i){ float x=nd(rng); a1+=0.25f*(x-a1); a2+=0.03f*(a1-a2); float env=0.6f+0.4f*std::sin(2*M_PI*4*i/SR); v[i]=(a1-a2)*env*3;
    float y=nd(rng),z=nd(rng); b1=0.97f*b1+0.2f*y; c1=0.97f*c1+0.2f*z; bl[i]=b1+0.3f*y; br[i]=c1+0.3f*z; }
  auto pw=[&](const std::vector<float>&x){double s=0;for(float f:x)s+=f*f;return s/x.size();};
  double gv=std::sqrt(pw(bl)/pw(v));
  for(auto&f:v) f*=gv; // equal power voice vs each bg channel
  float lvl = std::pow(10.f,-26/20.f)/std::sqrt((float)pw(bl)*2);
  std::printf("reverb=%d  voice/background change, dB (0 = unchanged; >0 = voice clearer)\n", reverb);
  for(auto&c:cfgs){
    std::vector<float> outs[2];
    for(int sgn=0;sgn<2;++sgn){ Renderer r; r.init(SR); r.set_profile(c.p); r.set_device(c.d); r.set_reverb(reverb);
      std::vector<float> in(2*N), out(2*N);
      for(int i=0;i<N;++i){ float s= sgn? -1.f:1.f; in[2*i]=lvl*(bl[i]+s*v[i]); in[2*i+1]=lvl*(br[i]+s*v[i]); }
      for(int b=0;b<N/kHop;++b) r.process_interleaved(&in[2*b*kHop], &out[2*b*kHop], kHop, 2);
      outs[sgn]=out; }
    double ev=0, eb=0; int from=int(3*SR)*2;
    for(int i=from;i<2*N;++i){ double vv=(outs[0][i]-outs[1][i])/2, bb=(outs[0][i]+outs[1][i])/2; ev+=vv*vv; eb+=bb*bb; }
    std::printf("%-10s %+6.1f\n", c.n, 10*std::log10(ev/eb) - 10*std::log10(1.0/1.0));
  }
  return 0;
}
int main(int argc,char**argv){ eq_and_dynamics(argc,argv); std::printf("\n"); return voice_to_background(argc,argv); }
