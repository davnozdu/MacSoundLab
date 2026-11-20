/* Plain C surface of the Sound Lab DSP core, so the app can drive it from Swift. */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SLRenderer SLRenderer;

enum {
    SL_PROFILE_OFF = 0,
    SL_PROFILE_GAME = 1,
    SL_PROFILE_MOVIE = 2,
    SL_PROFILE_MUSIC = 3,
    SL_PROFILE_SPEECH = 4,
};

enum {
    SL_DEVICE_HEADPHONES = 0,
    SL_DEVICE_SPEAKERS = 1,
    SL_DEVICE_GLASSES = 2,
};

SLRenderer* sl_create(float sample_rate, int profile, int device, int neural, int reverb);
void sl_destroy(SLRenderer* r);

/* in/out are interleaved float32. in_channels: 2, 6 or 8; out is always stereo. */
void sl_process(SLRenderer* r, const float* in, float* out, int frames, int in_channels);

void sl_set_profile(SLRenderer* r, int profile);
void sl_set_device(SLRenderer* r, int device);
void sl_set_neural(SLRenderer* r, int on);
void sl_set_reverb(SLRenderer* r, int on);
void sl_set_bypass(SLRenderer* r, int on);
void sl_reset(SLRenderer* r);

/* Directory holding libonnxruntime; models: vocals.onnx (and optional speech.onnx). */
void sl_set_runtime_dir(SLRenderer* r, const char* dir);
void sl_set_models(SLRenderer* r, const char* vocals, const char* speech);

int sl_profile(SLRenderer* r);
int sl_device(SLRenderer* r);
int sl_is_surround(SLRenderer* r);
int sl_accel_active(SLRenderer* r);   /* CoreML / NNAPI provider took the graph */
int sl_neural_ready(SLRenderer* r);   /* separator session created */

/* Model timing since the previous call, for the UI. */
void sl_stats(SLRenderer* r, float* model_ms, uint64_t* drops);

#ifdef __cplusplus
}
#endif
