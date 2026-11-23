#include "soundlab_c.h"

#include "../spatial.hpp"

struct SLRenderer {
    ally::Renderer renderer;
};

extern "C" {

SLRenderer* sl_create(float sample_rate, int profile, int device, int neural, int reverb) {
    if (sample_rate <= 0.0f) return nullptr;
    auto* handle = new SLRenderer();
    handle->renderer.init(sample_rate);
    handle->renderer.set_profile(static_cast<ally::Profile>(profile));
    handle->renderer.set_device(static_cast<ally::DeviceKind>(device));
    handle->renderer.set_neural(neural != 0);
    handle->renderer.set_reverb(reverb != 0);
    return handle;
}

void sl_destroy(SLRenderer* r) {
    delete r;
}

void sl_process(SLRenderer* r, const float* in, float* out, int frames, int in_channels) {
    if (r == nullptr || frames <= 0) return;
    r->renderer.process_interleaved(in, out, frames, in_channels);
}

void sl_set_profile(SLRenderer* r, int profile) {
    if (r == nullptr) return;
    r->renderer.set_profile(static_cast<ally::Profile>(profile));
}

void sl_set_device(SLRenderer* r, int device) {
    if (r == nullptr) return;
    r->renderer.set_device(static_cast<ally::DeviceKind>(device));
}

void sl_set_neural(SLRenderer* r, int on) {
    if (r != nullptr) r->renderer.set_neural(on != 0);
}

void sl_set_reverb(SLRenderer* r, int on) {
    if (r != nullptr) r->renderer.set_reverb(on != 0);
}

void sl_set_bypass(SLRenderer* r, int on) {
    if (r != nullptr) r->renderer.set_bypass(on != 0);
}

int sl_prepare_neural(SLRenderer* r) {
    return (r != nullptr && r->renderer.prepare_neural()) ? 1 : 0;
}

int sl_latency_frames(SLRenderer* r) {
    return r == nullptr ? 0 : r->renderer.latency_frames();
}

void sl_reset(SLRenderer* r) {
    if (r != nullptr) r->renderer.reset();
}

void sl_set_runtime_dir(SLRenderer* r, const char* dir) {
    if (r != nullptr) r->renderer.set_runtime_dir(dir);
}

void sl_set_models(SLRenderer* r, const char* vocals, const char* speech) {
    if (r != nullptr) r->renderer.set_model_paths(vocals, speech);
}

int sl_profile(SLRenderer* r) {
    return r == nullptr ? SL_PROFILE_OFF : static_cast<int>(r->renderer.profile());
}

int sl_device(SLRenderer* r) {
    return r == nullptr ? SL_DEVICE_HEADPHONES : static_cast<int>(r->renderer.device());
}

int sl_is_surround(SLRenderer* r) {
    return (r != nullptr && r->renderer.source_is_surround()) ? 1 : 0;
}

int sl_accel_active(SLRenderer* r) {
    return (r != nullptr && r->renderer.npu_active()) ? 1 : 0;
}

int sl_neural_ready(SLRenderer* r) {
    return (r != nullptr && r->renderer.neural_ready()) ? 1 : 0;
}

void sl_stats(SLRenderer* r, float* model_ms, uint64_t* drops) {
    if (r == nullptr) {
        if (model_ms) *model_ms = 0.0f;
        if (drops) *drops = 0;
        return;
    }
    r->renderer.stats(model_ms, drops);
}

} // extern "C"
