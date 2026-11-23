// Checks that the spatial renderer does what it claims, on real material:
//   A. each KU100 HRIR really sits at its 7.1 azimuth (ITD / ILD / rear cues),
//   B. how the stereo upmix spreads energy over the 7 virtual speakers,
//   C. spaciousness at the ears (IACC) and that a centred voice stays centred,
//   D. (NPU build) StemgenRT separation quality against known stems, live-paced.
// Build (A-C):  clang++ -std=c++17 -O2 -arch arm64 -I core tools/spatial_check.cpp core/spatial.cpp -o /tmp/spatial_check
// Build (A-D):  add -DALLY_ENABLE_NPU -I core/third_party/include core/vocals.cpp
// Run: /tmp/spatial_check voice.wav music.wav [singing.wav] [runtime_dir model.onnx]
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#define private public
#include "spatial.hpp"
#undef private
#include "hrir_ku100.hpp"

using namespace ally;
static const float SR = 48000.0f;

struct Audio { int ch = 0; std::vector<float> s; size_t frames() const { return ch ? s.size() / ch : 0; } };

static Audio read_wav(const char* path) {
    Audio a; FILE* f = std::fopen(path, "rb"); if (!f) { std::printf("cannot open %s\n", path); std::exit(1); }
    char id[4]; uint32_t size; std::fread(id, 1, 4, f); std::fread(&size, 4, 1, f); std::fread(id, 1, 4, f);
    uint16_t fmt = 0, ch = 0, bits = 0;
    while (std::fread(id, 1, 4, f) == 4 && std::fread(&size, 4, 1, f) == 1) {
        if (!std::memcmp(id, "fmt ", 4)) { std::vector<char> b(size); std::fread(b.data(), 1, size, f);
            std::memcpy(&fmt, &b[0], 2); std::memcpy(&ch, &b[2], 2); std::memcpy(&bits, &b[14], 2);
            if (fmt == 0xFFFE) std::memcpy(&fmt, &b[24], 2); }
        else if (!std::memcmp(id, "data", 4)) {
            a.ch = ch;
            if (fmt == 3 && bits == 32) { a.s.resize(size / 4); std::fread(a.s.data(), 4, a.s.size(), f); }
            else if (fmt == 1 && bits == 16) { std::vector<int16_t> t(size / 2); std::fread(t.data(), 2, t.size(), f); for (auto v : t) a.s.push_back(v / 32768.0f); }
            break;
        } else std::fseek(f, size + (size & 1), SEEK_CUR);
    }
    std::fclose(f); return a;
}

static double energy(const float* x, size_t n, size_t stride = 1) { double e = 0; for (size_t i = 0; i < n; ++i) e += double(x[i * stride]) * x[i * stride]; return e; }
static double db(double r) { return 10.0 * std::log10(r + 1e-20); }

// Max normalised interaural cross-correlation within +-1 ms and its lag (samples, + = left leads).
static void iacc(const std::vector<float>& l, const std::vector<float>& r, double& value, int& lag) {
    const int W = 48; double el = energy(l.data(), l.size()), er = energy(r.data(), r.size());
    value = 0; lag = 0;
    for (int k = -W; k <= W; ++k) {
        double c = 0;
        for (size_t i = W; i + W < l.size(); ++i) c += double(l[i]) * r[i + k];
        c /= std::sqrt(el * er) + 1e-20;
        if (std::fabs(c) > std::fabs(value)) { value = c; lag = k; }
    }
}

// Band energy of an IR above f0 via a naive DFT (256 taps, offline only).
static double band_energy(const float* h, int n, double f0, double f1) {
    double e = 0;
    for (int k = 0; k <= n / 2; ++k) { double f = k * SR / n; if (f < f0 || f >= f1) continue;
        std::complex<double> X = 0; for (int i = 0; i < n; ++i) X += double(h[i]) * std::polar(1.0, -2 * M_PI * k * i / n);
        e += std::norm(X); }
    return e;
}

static void part_a() {
    std::printf("A. HRIR placement (KU100 DFC). ITD expected from Woodworth, head radius 8.75 cm\n");
    std::printf("   spk  az   ITD meas   ITD exp    ILD all   ILD >3k   HF(4-12k) vs FL\n");
    const char* names[7] = {"FL", "FR", "FC", "RL", "RR", "SL", "SR"};
    const int az[7] = {30, -30, 0, 145, -145, 100, -100};
    const double fl_hf = band_energy(kHrirKu100[0][0], kHrirLength, 4000, 12000) + band_energy(kHrirKu100[0][1], kHrirLength, 4000, 12000);
    for (int s = 0; s < 7; ++s) {
        const float* L = kHrirKu100[s][0]; const float* R = kHrirKu100[s][1];
        int best = 0; double bv = -1;
        for (int k = -40; k <= 40; ++k) { double c = 0; for (int i = 0; i < kHrirLength; ++i) { int j = i + k; if (j >= 0 && j < kHrirLength) c += double(L[i]) * R[j]; } if (c > bv) { bv = c; best = k; } }
        const double lat = std::asin(std::sin(az[s] * M_PI / 180.0));
        const double itd_exp = 0.0875 / 343.0 * (std::fabs(lat) + std::sin(std::fabs(lat))) * 1e6 * (lat >= 0 ? 1 : -1);
        const double ild = db(energy(L, kHrirLength) / energy(R, kHrirLength));
        const double ild_hf = db(band_energy(L, kHrirLength, 3000, 20000) / band_energy(R, kHrirLength, 3000, 20000));
        const double hf = db((band_energy(L, kHrirLength, 4000, 12000) + band_energy(R, kHrirLength, 4000, 12000)) / fl_hf);
        std::printf("   %-3s %4d  %+6.0f us  %+6.0f us  %+6.1f dB  %+6.1f dB  %+6.1f dB\n", names[s], az[s], best / SR * 1e6, itd_exp, ild, ild_hf, hf);
    }
}

struct Case { const char* name; Profile p; bool reverb; };

static std::vector<float> make_stereo(const Audio& voice, const Audio& music, double voice_db, size_t frames) {
    std::vector<float> x(frames * 2, 0.0f);
    double ev = 0, em = 0;
    for (size_t i = 0; i < frames; ++i) {
        float v = voice.ch ? voice.s[(i % voice.frames()) * voice.ch] : 0.0f; ev += 2.0 * v * v;
        float ml = music.ch ? music.s[(i % music.frames()) * music.ch] : 0.0f, mr = music.ch ? music.s[(i % music.frames()) * music.ch + (music.ch > 1)] : 0.0f;
        em += double(ml) * ml + double(mr) * mr; }
    const double gv = (voice.ch && music.ch) ? std::sqrt(em / ev) * std::pow(10.0, voice_db / 20.0) : 1.0;
    for (size_t i = 0; i < frames; ++i) {
        float v = voice.ch ? float(gv) * voice.s[(i % voice.frames()) * voice.ch] : 0.0f;
        float ml = music.ch ? music.s[(i % music.frames()) * music.ch] : 0.0f, mr = music.ch ? music.s[(i % music.frames()) * music.ch + (music.ch > 1)] : 0.0f;
        x[2 * i] = 0.25f * (v + ml); x[2 * i + 1] = 0.25f * (v + mr); }
    return x;
}

static void part_bc(const Audio& voice, const Audio& music) {
    const Case cases[] = {{"game  ", Profile::Game, false}, {"movie ", Profile::Movie, false}, {"music ", Profile::Music, false}, {"speech", Profile::Speech, false},
                          {"game +room", Profile::Game, true}, {"movie+room", Profile::Movie, true}, {"music+room", Profile::Music, true}};
    const Audio none;
    struct Src { const char* name; const Audio* v; const Audio* m; };
    const Src srcs[] = {{"voice only (centred)", &voice, &none}, {"music only (stereo)", &none, &music}, {"voice -3 dB over music", &voice, &music}};
    const size_t N = size_t(12 * SR) / kHop * kHop;

    std::printf("\nB. Energy per virtual speaker, %% of all 7 feeds (headphones, stereo input; LFE shown separately)\n");
    std::printf("   %-24s %-10s   FL    FR    FC    RL    RR    SL    SR   | LFE\n", "source", "profile");
    std::printf("\nC. At the ears (headphones). IACC: 1 = in the head / mono, lower = wider and more external.\n");
    std::vector<std::string> c_lines;
    for (const auto& src : srcs) {
        auto in = make_stereo(*src.v, *src.m, -3.0, N);
        std::vector<float> il(N), ir(N); for (size_t i = 0; i < N; ++i) { il[i] = in[2 * i]; ir[i] = in[2 * i + 1]; }
        double ic0; int lag0; iacc(il, ir, ic0, lag0);
        char buf[256]; std::snprintf(buf, sizeof buf, "   %-24s input       IACC %.2f  lag %+3d us", src.name, ic0, int(lag0 / SR * 1e6)); c_lines.push_back(buf);
        for (const auto& c : cases) {
            Renderer r; r.init(SR); r.set_profile(c.p); r.set_device(DeviceKind::Headphones); r.set_reverb(c.reverb); r.set_neural(false);
            std::array<double, 8> e{}; std::vector<float> out(2 * N);
            for (size_t b = 0; b < N / kHop; ++b) {
                r.process_interleaved(&in[2 * b * kHop], &out[2 * b * kHop], kHop, 2);
                if (b * kHop > SR) { for (int s = 0; s < 7; ++s) e[s] += energy(&r.feeds_[s * kHop], kHop); e[7] += energy(r.lfe_.data(), kHop); }
            }
            double tot = 0; for (int s = 0; s < 7; ++s) tot += e[s];
            if (!c.reverb) {
                std::printf("   %-24s %-10s", src.name, c.name);
                for (int s = 0; s < 7; ++s) std::printf(" %5.1f", 100 * e[s] / tot);
                std::printf("  | %s\n", e[7] > 0 ? "yes" : "none");
            }
            std::vector<float> ol(N - size_t(SR)), orr(N - size_t(SR));
            for (size_t i = 0; i < ol.size(); ++i) { ol[i] = out[2 * (i + size_t(SR))]; orr[i] = out[2 * (i + size_t(SR)) + 1]; }
            double ic; int lag; iacc(ol, orr, ic, lag);
            const double ild = db(energy(ol.data(), ol.size()) / energy(orr.data(), orr.size()));
            std::snprintf(buf, sizeof buf, "   %-24s %-10s  IACC %.2f  lag %+3d us  ILD %+4.1f dB", src.name, c.name, ic, int(lag / SR * 1e6), ild);
            c_lines.push_back(buf);
        }
    }
    std::printf("\n"); for (auto& l : c_lines) std::printf("%s\n", l.c_str());
}

#ifdef ALLY_ENABLE_NPU
static void part_d(const Audio& voice, const Audio& music, const char* label, const std::string& runtime, const std::string& model) {
    const size_t N = std::min<size_t>(size_t(12 * SR), voice.frames()) / kHop * kHop;
    std::vector<float> v(N), ml(N), mr(N);
    double ev = 0, em = 0;
    for (size_t i = 0; i < N; ++i) { v[i] = voice.s[i * voice.ch]; ml[i] = music.s[(i % music.frames()) * music.ch]; mr[i] = music.s[(i % music.frames()) * music.ch + 1]; ev += 2.0 * v[i] * v[i]; em += double(ml[i]) * ml[i] + double(mr[i]) * mr[i]; }
    const float gv = float(std::sqrt(em / ev) * std::pow(10.0, -3.0 / 20.0));
    for (auto& x : v) x *= gv;
    PipelinedVocals sep;
    if (!sep.ensure_loaded(model.c_str(), false, (runtime + "/libonnxruntime.dylib").c_str())) { std::printf("D. model did not load\n"); return; }
    std::vector<float> dry(2 * N), rest(2 * N);
    std::array<float, 2 * kHop> hop{}, d{}, rr{};
    auto t0 = std::chrono::steady_clock::now();
    for (size_t b = 0; b < N / kHop; ++b) {
        for (int i = 0; i < kHop; ++i) { size_t n = b * kHop + i; hop[i] = 0.25f * (v[n] + ml[n]); hop[kHop + i] = 0.25f * (v[n] + mr[n]); }
        sep.process(hop.data(), d.data(), rr.data(), b);
        for (int i = 0; i < kHop; ++i) { dry[2 * (b * kHop + i)] = d[i]; dry[2 * (b * kHop + i) + 1] = d[kHop + i]; rest[2 * (b * kHop + i)] = rr[i]; rest[2 * (b * kHop + i) + 1] = rr[kHop + i]; }
        std::this_thread::sleep_until(t0 + std::chrono::microseconds(int64_t((b + 1) * kHop / SR * 1e6)));  // live pace
    }
    float ms; uint64_t drops; sep.stats(&ms, &drops);
    // Find the pipeline delay: dry is the input mix, delayed.
    int delay = 0;
    for (int dh = 0; dh <= 24 * kHop; ++dh) { double err = 0; for (size_t n = 4 * SR; n < 4 * SR + 4800; ++n) err += std::fabs(dry[2 * n] - 0.25f * (v[n - dh] + ml[n - dh])); if (err < 1e-3) { delay = dh; break; } }
    const size_t from = size_t(2 * SR);
    double sv = 0, nv = 0, sp = 0, np = 0, am = 0, sm = 0, nm = 0, vv = 0, cv = 0, cm = 0, mm = 0, vmix = 0;
    for (size_t n = from; n < N; ++n) for (int c = 0; c < 2; ++c) {
        const size_t k = n - delay; const float tv = 0.25f * v[k]; const float tm = 0.25f * (c ? mr[k] : ml[k]);
        const float est = dry[2 * n + c] - rest[2 * n + c];                    // model vocals
        const float pas = 0.25f * 0.5f * (0.5f * ((v[k] + ml[k]) + (v[k] + mr[k])));   // passive: voice = 0.5 * mid
        sv += tv * tv; nv += (est - tv) * (est - tv); np += (pas - tv) * (pas - tv);
        sm += tm * tm; nm += (rest[2 * n + c] - tm) * (rest[2 * n + c] - tm);
        vv += tv * tv; mm += tm * tm; cv += est * tv; cm += est * tm; vmix += tv * tm;
    }
    // Least squares est = a*v + b*m.
    const double det = vv * mm - vmix * vmix, a = (cv * mm - cm * vmix) / det, bm = (cm * vv - cv * vmix) / det;
    std::printf("   %-8s pipeline delay %4.1f ms, model %.2f ms/hop, drops %llu\n", label, delay / SR * 1e3, ms, (unsigned long long)drops);
    std::printf("            voice SDR: model %+5.1f dB   passive 0.5*mid %+5.1f dB   (higher = cleaner voice)\n", db(sv / nv), db(sv / np));
    std::printf("            in the model's voice stem: %.0f%% of the voice, %.0f%% of the music -> voice/music %+5.1f dB (input -3.0 dB)\n",
                100 * a, 100 * bm, db((a * a * vv) / (bm * bm * mm + 1e-20)));
    std::printf("            music stem (rest) SDR %+5.1f dB\n", db(sm / nm));
}
#endif

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: spatial_check voice.wav music.wav [singing.wav] [runtime_dir model.onnx]\n"); return 1; }
    const Audio voice = read_wav(argv[1]), music = read_wav(argv[2]);
    part_a();
    part_bc(voice, music);
#ifdef ALLY_ENABLE_NPU
    if (argc >= 6) {
        std::printf("\nD. StemgenRT separation, live-paced, voice -3 dB over music\n");
        part_d(voice, music, "speech", argv[4], argv[5]);
        const Audio sing = read_wav(argv[3]);
        part_d(sing, music, "singing", argv[4], argv[5]);
    }
#endif
    return 0;
}
