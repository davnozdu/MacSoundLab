#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace ally {

// Asynchronous StemgenRT (vocals.onnx) vocal separator with NNAPI (NPU) support.
// Uses dlopen("libonnxruntime.so") at runtime so the base DSP has zero hard
// dependency on ONNX Runtime when the neural toggle is off.
class PipelinedVocals {
public:
    static constexpr int kHop = 128;
    static constexpr int kBufferHops = 10;

    PipelinedVocals() = default;
    ~PipelinedVocals();

    PipelinedVocals(const PipelinedVocals&) = delete;
    PipelinedVocals& operator=(const PipelinedVocals&) = delete;

    // Lazy init: loads libonnxruntime and vocals.onnx. Returns true if ready.
    // ort_library: optional explicit path to the runtime library.
    bool ensure_loaded(const char* model_path, bool use_npu = true, const char* ort_library = nullptr);
    bool is_ready() const { return ready_.load(); }
    bool using_npu() const { return npu_active_.load(); }

    // Non-blocking process: feeds one stereo hop (planar: 2 * kHop) and outputs
    // time-aligned dry (2 * kHop) and rest = dry - vocals (2 * kHop).
    // Falls back smoothly to passive mid-side matrix on underrun.
    void process(const float* stereo_in, float* dry_out, float* rest_out, uint64_t tick);

    // Rolling statistics for the UI since the previous call: average inference
    // time per hop in milliseconds, and the number of hops that fell back.
    void stats(float* average_ms, uint64_t* drops);

private:
    struct Impl;
    Impl* impl_ = nullptr;
    std::atomic<bool> ready_{false};
    std::atomic<bool> failed_{false};
    std::atomic<bool> npu_active_{false};
    std::atomic<uint64_t> runs_{0};
    std::atomic<uint64_t> drops_{0};
    std::atomic<uint64_t> total_us_{0};

    void worker_loop();
    static void passive_matrix(const float* dry, float* rest);
};

} // namespace ally
