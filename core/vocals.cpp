#include "vocals.hpp"

#include <dlfcn.h>
#include <onnxruntime_c_api.h>
#ifdef __ANDROID__
#include <nnapi_provider_factory.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#ifdef __ANDROID__
#include <android/log.h>
#define VLOGI(...) __android_log_print(ANDROID_LOG_INFO, "AllySoundNPU", __VA_ARGS__)
#define VLOGE(...) __android_log_print(ANDROID_LOG_ERROR, "AllySoundNPU", __VA_ARGS__)
#else
#define VLOGI(...) do { std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#define VLOGE(...) do { std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#endif

namespace ally {

namespace {

constexpr int kHopFrames = PipelinedVocals::kHop;
constexpr size_t kQueueSlots = 32;

struct InSlot {
    uint64_t gen = 0;
    uint64_t seq = 0;
    std::array<float, 2 * kHopFrames> audio{};
};

struct OutSlot {
    uint64_t gen = 0;
    uint64_t seq = 0;
    std::array<float, 2 * kHopFrames> dry{};
    std::array<float, 2 * kHopFrames> rest{};
};

// Fixed-capacity FIFO: the audio thread must not allocate, and std::deque of
// std::vector allocated twice per hop (375 times a second).
template <typename T>
struct Fifo {
    std::array<T, kQueueSlots> items{};
    size_t head = 0;
    size_t count = 0;

    bool empty() const { return count == 0; }
    size_t size() const { return count; }
    T& front() { return items[head]; }
    void pop() { head = (head + 1) % kQueueSlots; --count; }
    T* push() {
        if (count == kQueueSlots) return nullptr;
        T* slot = &items[(head + count) % kQueueSlots];
        ++count;
        return slot;
    }
    void clear() { head = 0; count = 0; }
};

} // namespace

struct PipelinedVocals::Impl {
    void* ort_lib = nullptr;
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    OrtSessionOptions* opts = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* mem_info = nullptr;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_ptrs;
    std::vector<const char*> output_ptrs;
    std::vector<std::vector<int64_t>> state_shapes;
    std::vector<std::vector<float>> state_bufs;
    std::vector<float> prev_in; // 2 * kHop

    // Input tensors wrap fixed buffers and are created once, not per run.
    std::array<float, 2 * kHopFrames> audio_buf{};
    std::vector<OrtValue*> in_vals;
    std::vector<OrtValue*> out_vals;

    std::mutex mutex;
    std::condition_variable cv;
    bool stop = false;
    std::thread worker;

    Fifo<InSlot> inbox;
    Fifo<OutSlot> outbox;
    uint64_t gen = 0;
    uint64_t seq = 0;
    uint64_t last_tick = UINT64_MAX;
    float wet = 1.0f;

    // Fallback delay line for (1 + kBufferHops) hops
    std::vector<float> fallback_buf;
    size_t fallback_pos = 0;

    // Releases a status; true when there was no error.
    bool ok(OrtStatus* status) const {
        if (status == nullptr) return true;
        VLOGE("onnxruntime: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return false;
    }

    void reset_states() {
        for (auto& b : state_bufs) {
            std::fill(b.begin(), b.end(), 0.0f);
        }
        std::fill(prev_in.begin(), prev_in.end(), 0.0f);
    }

    bool bind_inputs() {
        int64_t audio_shape[3] = {1, 2, kHopFrames};
        OrtValue* audio_val = nullptr;
        if (!ok(api->CreateTensorWithDataAsOrtValue(mem_info, audio_buf.data(), audio_buf.size() * sizeof(float),
                                                    audio_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &audio_val))) {
            return false;
        }
        in_vals.push_back(audio_val);
        for (size_t i = 0; i < state_bufs.size(); ++i) {
            OrtValue* sv = nullptr;
            if (!ok(api->CreateTensorWithDataAsOrtValue(mem_info, state_bufs[i].data(),
                                                        state_bufs[i].size() * sizeof(float),
                                                        state_shapes[i].data(), state_shapes[i].size(),
                                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &sv))) {
                return false;
            }
            in_vals.push_back(sv);
        }
        for (const auto& name : input_names) input_ptrs.push_back(name.c_str());
        for (const auto& name : output_names) output_ptrs.push_back(name.c_str());
        out_vals.assign(output_names.size(), nullptr);
        return in_vals.size() == input_names.size();
    }

    void release_values() {
        if (!api) return;
        for (auto* v : in_vals) if (v) api->ReleaseValue(v);
        in_vals.clear();
    }

    bool run_model(const float* in_audio, float* out_dry, float* out_rest) {
        if (!api || !session || in_vals.empty()) return false;
        std::memcpy(audio_buf.data(), in_audio, audio_buf.size() * sizeof(float));
        std::fill(out_vals.begin(), out_vals.end(), nullptr);

        if (!ok(api->Run(session, nullptr, input_ptrs.data(), in_vals.data(), in_vals.size(),
                         output_ptrs.data(), output_ptrs.size(), out_vals.data()))) {
            for (auto*& v : out_vals) if (v) { api->ReleaseValue(v); v = nullptr; }
            return false;
        }

        bool good = true;
        float* vocals_data = nullptr;
        good = ok(api->GetTensorMutableData(out_vals[0], reinterpret_cast<void**>(&vocals_data))) && vocals_data;
        if (good) {
            std::memcpy(out_dry, prev_in.data(), 2 * kHopFrames * sizeof(float));
            std::memcpy(prev_in.data(), in_audio, 2 * kHopFrames * sizeof(float));
            for (int i = 0; i < 2 * kHopFrames; ++i) {
                out_rest[i] = out_dry[i] - vocals_data[i];
            }
            // The state tensors wrap state_bufs, which this run read as input:
            // the new state is copied back only after the run has finished.
            for (size_t i = 0; i < state_bufs.size() && i + 1 < out_vals.size(); ++i) {
                float* sdata = nullptr;
                if (ok(api->GetTensorMutableData(out_vals[i + 1], reinterpret_cast<void**>(&sdata))) && sdata) {
                    std::memcpy(state_bufs[i].data(), sdata, state_bufs[i].size() * sizeof(float));
                }
            }
        }
        for (auto*& v : out_vals) if (v) { api->ReleaseValue(v); v = nullptr; }
        return good;
    }
};

PipelinedVocals::~PipelinedVocals() {
    if (loader_.joinable()) loader_.join();
    if (!impl_) return;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stop = true;
    }
    impl_->cv.notify_all();
    if (impl_->worker.joinable()) {
        impl_->worker.join();
    }
    if (impl_->api) {
        impl_->release_values();
        if (impl_->session) impl_->api->ReleaseSession(impl_->session);
        if (impl_->opts) impl_->api->ReleaseSessionOptions(impl_->opts);
        if (impl_->mem_info) impl_->api->ReleaseMemoryInfo(impl_->mem_info);
        if (impl_->env) impl_->api->ReleaseEnv(impl_->env);
    }
    if (impl_->ort_lib) dlclose(impl_->ort_lib);
    delete impl_;
    impl_ = nullptr;
}

void PipelinedVocals::load_async(const std::string& model_path, bool use_npu, const std::string& ort_library) {
    if (ready_.load() || failed_.load()) return;
    bool expected = false;
    if (!load_started_.compare_exchange_strong(expected, true)) return;
    loader_ = std::thread([this, model_path, use_npu, ort_library]() {
        ensure_loaded(model_path.c_str(), use_npu, ort_library.empty() ? nullptr : ort_library.c_str());
    });
}

bool PipelinedVocals::ensure_loaded(const char* model_path, bool use_npu, const char* ort_library) {
    std::lock_guard<std::mutex> guard(load_mutex_);
    if (ready_.load()) return true;
    if (failed_.load()) return false;

    auto* d = new Impl();
    auto fail = [&]() {
        failed_.store(true);
        npu_active_.store(false);
        if (d->api) {
            d->release_values();
            if (d->session) d->api->ReleaseSession(d->session);
            if (d->opts) d->api->ReleaseSessionOptions(d->opts);
            if (d->mem_info) d->api->ReleaseMemoryInfo(d->mem_info);
            if (d->env) d->api->ReleaseEnv(d->env);
        }
        if (d->ort_lib) dlclose(d->ort_lib);
        delete d;
        return false;
    };

    // The library ships with the app (Contents/Frameworks) or next to the
    // module binary; the plain name covers an install onto the dyld path.
    const char* kOrtCandidates[] = {
#ifdef __APPLE__
        ort_library,
        "@executable_path/../Frameworks/libonnxruntime.dylib",
        "@loader_path/../Frameworks/libonnxruntime.dylib",
        "libonnxruntime.dylib",
#else
        ort_library,
        "libonnxruntime.so",
        "/vendor/lib64/soundfx/libonnxruntime.so",
        "/vendor/lib64/libonnxruntime.so",
#endif
    };
    for (const char* cand : kOrtCandidates) {
        if (cand == nullptr) continue;
        d->ort_lib = dlopen(cand, RTLD_NOW | RTLD_LOCAL);
        if (d->ort_lib) {
            VLOGI("onnxruntime loaded: %s", cand);
            break;
        }
        VLOGE("dlopen %s failed: %s", cand, dlerror());
    }
    if (!d->ort_lib) return fail();

    using GetApiBaseFn = const OrtApiBase* (*)();
    auto get_base = reinterpret_cast<GetApiBaseFn>(dlsym(d->ort_lib, "OrtGetApiBase"));
    if (!get_base) {
        VLOGE("OrtGetApiBase not found");
        return fail();
    }
    d->api = get_base()->GetApi(ORT_API_VERSION);
    if (!d->api) {
        VLOGE("OrtApi version %d unavailable", ORT_API_VERSION);
        return fail();
    }
    if (!d->ok(d->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "AllySound", &d->env))) return fail();

    auto create_session = [&](bool try_accel) -> bool {
        if (d->opts) { d->api->ReleaseSessionOptions(d->opts); d->opts = nullptr; }
        if (!d->ok(d->api->CreateSessionOptions(&d->opts))) return false;
        // One thread: the model needs ~0.5 ms of a 2.67 ms block, and every
        // extra ORT thread spins between hops and keeps a core busy for nothing.
        // Spinning is off for the same reason - the wake-up cost is smaller
        // than the idle burn at this block rate.
        d->ok(d->api->SetIntraOpNumThreads(d->opts, 1));
        d->ok(d->api->SetInterOpNumThreads(d->opts, 1));
        d->ok(d->api->SetSessionGraphOptimizationLevel(d->opts, ORT_ENABLE_ALL));
        d->ok(d->api->AddSessionConfigEntry(d->opts, "session.intra_op.allow_spinning", "0"));
        d->ok(d->api->AddSessionConfigEntry(d->opts, "session.inter_op.allow_spinning", "0"));
        d->ok(d->api->AddSessionConfigEntry(d->opts, "session.use_spinning", "0"));
        npu_active_.store(false);
        if (try_accel) {
#ifdef __ANDROID__
            using AccelFn = OrtStatus* (*)(OrtSessionOptions*, uint32_t);
            auto append = reinterpret_cast<AccelFn>(
                dlsym(d->ort_lib, "OrtSessionOptionsAppendExecutionProvider_Nnapi"));
            const char* label = "NNAPI";
#else
            // Apple: CoreML runs the graph on the GPU / Neural Engine.
            using AccelFn = OrtStatus* (*)(OrtSessionOptions*, uint32_t);
            auto append = reinterpret_cast<AccelFn>(
                dlsym(d->ort_lib, "OrtSessionOptionsAppendExecutionProvider_CoreML"));
            const char* label = "CoreML";
#endif
            if (!append) return false;
            OrtStatus* ast = append(d->opts, 0);
            if (ast) {
                VLOGI("%s EP unavailable (%s), using CPU", label, d->api->GetErrorMessage(ast));
                d->api->ReleaseStatus(ast);
                return false;
            }
        }
        OrtStatus* sst = d->api->CreateSession(d->env, model_path, d->opts, &d->session);
        if (sst) {
            VLOGE("CreateSession failed (accel=%d): %s", try_accel ? 1 : 0, d->api->GetErrorMessage(sst));
            d->api->ReleaseStatus(sst);
            d->session = nullptr;
            return false;
        }
        npu_active_.store(try_accel);
        return true;
    };

    if (!(use_npu && create_session(true)) && !create_session(false)) return fail();

    if (!d->ok(d->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &d->mem_info))) return fail();

    OrtAllocator* alloc = nullptr;
    if (!d->ok(d->api->GetAllocatorWithDefaultOptions(&alloc))) return fail();
    size_t num_in = 0, num_out = 0;
    if (!d->ok(d->api->SessionGetInputCount(d->session, &num_in)) ||
        !d->ok(d->api->SessionGetOutputCount(d->session, &num_out)) || num_in == 0 || num_out == 0) {
        return fail();
    }

    for (size_t i = 0; i < num_in; ++i) {
        char* name = nullptr;
        if (!d->ok(d->api->SessionGetInputName(d->session, i, alloc, &name)) || !name) return fail();
        d->input_names.emplace_back(name);
        d->ok(d->api->AllocatorFree(alloc, name));

        if (i > 0) {
            OrtTypeInfo* ti = nullptr;
            if (!d->ok(d->api->SessionGetInputTypeInfo(d->session, i, &ti))) return fail();
            const OrtTensorTypeAndShapeInfo* si = nullptr;
            size_t dims = 0;
            bool good = d->ok(d->api->CastTypeInfoToTensorInfo(ti, &si)) && si &&
                        d->ok(d->api->GetDimensionsCount(si, &dims));
            std::vector<int64_t> shape(dims);
            good = good && d->ok(d->api->GetDimensions(si, shape.data(), dims));
            d->api->ReleaseTypeInfo(ti);
            if (!good) return fail();
            size_t total = 1;
            for (auto& dim : shape) {
                if (dim <= 0) dim = 1;
                total *= static_cast<size_t>(dim);
            }
            d->state_shapes.push_back(std::move(shape));
            d->state_bufs.emplace_back(total, 0.0f);
        }
    }
    for (size_t i = 0; i < num_out; ++i) {
        char* name = nullptr;
        if (!d->ok(d->api->SessionGetOutputName(d->session, i, alloc, &name)) || !name) return fail();
        d->output_names.emplace_back(name);
        d->ok(d->api->AllocatorFree(alloc, name));
    }

    d->prev_in.assign(2 * kHop, 0.0f);
    d->fallback_buf.assign((1 + kBufferHops) * 2 * kHop, 0.0f);
    if (!d->bind_inputs()) return fail();

    // Warm up
    std::array<float, 2 * kHop> zeros{}, d_tmp{}, r_tmp{};
    for (int i = 0; i < 4; ++i) {
        d->run_model(zeros.data(), d_tmp.data(), r_tmp.data());
    }
    d->reset_states();

    impl_ = d;
    impl_->worker = std::thread([this]() { worker_loop(); });
    ready_.store(true);
    VLOGI("StemgenRT loaded (npu=%d)", npu_active_.load() ? 1 : 0);
    return true;
}

void PipelinedVocals::worker_loop() {
    uint64_t worker_gen = UINT64_MAX;
    InSlot item;
    OutSlot out;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(impl_->mutex);
            impl_->cv.wait(lock, [&]() { return impl_->stop || !impl_->inbox.empty(); });
            if (impl_->stop) break;
            item = impl_->inbox.front();
            impl_->inbox.pop();
        }

        if (item.gen != worker_gen) {
            impl_->reset_states();
            worker_gen = item.gen;
        }

        out.gen = item.gen;
        const auto started = std::chrono::steady_clock::now();
        const bool ok = impl_->run_model(item.audio.data(), out.dry.data(), out.rest.data());
        const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - started)
                                    .count();
        total_us_.fetch_add(static_cast<uint64_t>(std::max<int64_t>(0, elapsed_us)));
        runs_.fetch_add(1);
        if (ok) {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (item.gen == impl_->gen && item.seq >= 1) {
                out.seq = item.seq - 1;
                if (OutSlot* slot = impl_->outbox.push()) *slot = out;
            }
        }
    }
}

void PipelinedVocals::stats(float* average_ms, uint64_t* drops) {
    const uint64_t runs = runs_.exchange(0);
    const uint64_t total_us = total_us_.exchange(0);
    if (average_ms) {
        *average_ms = runs ? static_cast<float>(static_cast<double>(total_us) / 1000.0 / static_cast<double>(runs)) : 0.0f;
    }
    if (drops) *drops = drops_.exchange(0);
}

void PipelinedVocals::passive_matrix(const float* dry, float* rest) {
    for (int i = 0; i < kHop; ++i) {
        float mid = 0.5f * (dry[i] + dry[kHop + i]);
        float v = 0.5f * mid;
        rest[i] = dry[i] - v;
        rest[kHop + i] = dry[kHop + i] - v;
    }
}

void PipelinedVocals::process(const float* stereo_in, float* dry_out, float* rest_out, uint64_t tick) {
    if (!ready_.load(std::memory_order_acquire) || !impl_) {
        std::memcpy(dry_out, stereo_in, 2 * kHop * sizeof(float));
        passive_matrix(dry_out, rest_out);
        return;
    }

    uint64_t cur_gen, cur_seq;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->last_tick != UINT64_MAX && tick != impl_->last_tick + 1) {
            impl_->gen++;
            impl_->seq = 0;
            impl_->inbox.clear();
            impl_->outbox.clear();
            std::fill(impl_->fallback_buf.begin(), impl_->fallback_buf.end(), 0.0f);
            impl_->fallback_pos = 0;
            impl_->wet = 1.0f;
        }
        impl_->last_tick = tick;
        cur_seq = impl_->seq++;
        if (impl_->inbox.size() > kBufferHops + 4) {
            impl_->gen++;
            impl_->seq = 1;
            cur_seq = 0;
            impl_->inbox.clear();
            impl_->outbox.clear();
            impl_->wet = 0.0f;
        }
        cur_gen = impl_->gen;
        if (InSlot* slot = impl_->inbox.push()) {
            slot->gen = cur_gen;
            slot->seq = cur_seq;
            std::memcpy(slot->audio.data(), stereo_in, 2 * kHop * sizeof(float));
        }
    }
    impl_->cv.notify_one();

    // Advance fallback delay line
    const size_t delay_hops = 1 + kBufferHops;
    float* slot = &impl_->fallback_buf[impl_->fallback_pos * 2 * kHop];
    std::array<float, 2 * kHop> dry_ref{};
    std::memcpy(dry_ref.data(), slot, 2 * kHop * sizeof(float));
    std::memcpy(slot, stereo_in, 2 * kHop * sizeof(float));
    impl_->fallback_pos = (impl_->fallback_pos + 1) % delay_hops;

    if (cur_seq < delay_hops) {
        std::memcpy(dry_out, dry_ref.data(), 2 * kHop * sizeof(float));
        passive_matrix(dry_out, rest_out);
        return;
    }

    const uint64_t target_seq = cur_seq - delay_hops;
    bool got = false;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto& box = impl_->outbox;
        while (!box.empty() && (box.front().gen != cur_gen || box.front().seq < target_seq)) {
            box.pop();
        }
        if (!box.empty() && box.front().gen == cur_gen && box.front().seq == target_seq) {
            const OutSlot& res = box.front();
            std::memcpy(dry_out, res.dry.data(), 2 * kHop * sizeof(float));
            if (impl_->wet < 1.0f) {
                std::array<float, 2 * kHop> p_rest{};
                passive_matrix(dry_out, p_rest.data());
                const float step = (1.0f - impl_->wet) / static_cast<float>(kHop - 1);
                for (int i = 0; i < kHop; ++i) {
                    const float w = impl_->wet + step * i;
                    rest_out[i] = p_rest[i] * (1.0f - w) + res.rest[i] * w;
                    rest_out[kHop + i] = p_rest[kHop + i] * (1.0f - w) + res.rest[kHop + i] * w;
                }
                impl_->wet = 1.0f;
            } else {
                std::memcpy(rest_out, res.rest.data(), 2 * kHop * sizeof(float));
            }
            box.pop();
            got = true;
        }
    }
    if (got) return;

    drops_.fetch_add(1);
    impl_->wet = 0.0f;
    std::memcpy(dry_out, dry_ref.data(), 2 * kHop * sizeof(float));
    passive_matrix(dry_out, rest_out);
}

} // namespace ally
