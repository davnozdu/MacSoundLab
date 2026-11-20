// Times the DSP core alone (no ONNX Runtime: built without ALLY_ENABLE_NPU).
// Run: clang++ -std=c++17 -O2 -arch arm64 -I core tools/bench_dsp.cpp core/spatial.cpp -o /tmp/bench_dsp
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "spatial.hpp"

using namespace ally;

int main() {
    constexpr int kBlocks = 20000;  // ~53 s of audio at 128 frames
    struct Case { const char* name; Profile profile; bool reverb; DeviceKind device; };
    const Case cases[] = {
        {"movie  hp  room=on ", Profile::Movie,    true,  DeviceKind::Headphones},
        {"movie  hp  room=off", Profile::Movie,    false, DeviceKind::Headphones},
        {"speech hp  room=on ", Profile::Speech,   true,  DeviceKind::Headphones},
        {"speech hp  room=off", Profile::Speech,   false, DeviceKind::Headphones},
        {"music  hp  room=on ", Profile::Music,    true,  DeviceKind::Headphones},
        {"movie  spk room=on ", Profile::Movie,    true,  DeviceKind::Speakers},
    };

    std::vector<float> in(2 * kHop), out(2 * kHop);
    for (int i = 0; i < kHop; ++i) {  // noise-ish, avoids denormal silence
        in[2 * i] = 0.15f * std::sin(0.031f * i) + 0.05f * std::sin(1.7f * i);
        in[2 * i + 1] = 0.14f * std::sin(0.029f * i + 0.6f) + 0.05f * std::sin(1.3f * i);
    }

    for (const auto& c : cases) {
        Renderer r;
        r.init(48000.0f);
        r.set_neural(false);
        r.set_profile(c.profile);
        r.set_device(c.device);
        r.set_reverb(c.reverb);
        for (int i = 0; i < 400; ++i) r.process_interleaved(in.data(), out.data(), kHop, 2);  // warm up
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kBlocks; ++i) r.process_interleaved(in.data(), out.data(), kHop, 2);
        auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / kBlocks;
        std::printf("%s  dsp=%7.3f ms/block   (%.1f%% of one core at 375 blocks/s)\n",
                    c.name, us / 1000.0, us / 1000.0 * 375.0 / 10.0);
    }
    return 0;
}
