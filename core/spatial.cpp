#include "spatial.hpp"
#include "hrir_ku100.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unistd.h>

namespace ally {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr float kSilence = 1e-5f;
constexpr double kSurroundHold = 1.0;
constexpr float kAmbienceDelaySec = 0.015f;
constexpr float kRoomSeconds = 0.45f;
constexpr float kSurroundTrim = 0.65313055f;       // 10^(-3.7/20)
constexpr float kStereoTrim = 1.6595869f;          // 10^(+4.4/20)
constexpr float kSpeakerTrimStereo = 0.7762471f;   // 10^(-2.2/20)
constexpr float kSpeakerTrimSurround = 0.3981072f; // 10^(-8.0/20)

#ifdef ALLY_ENABLE_NPU
// The host application provides both paths through Renderer::set_model_paths.
#endif
constexpr float kGlassesBassGain = 0.2589254f;     // 10^(2.0/20) - 1

struct ProfileParams {
    float dialogue;
    float room;
    float rt60;
    bool compress;
    float side_gain;
    float rear_gain;
};

ProfileParams get_profile_params(Profile p) {
    switch (p) {
        case Profile::Movie:
            return {1.2589254f, 0.26f, 0.40f, true, 0.48f, 0.42f};
        case Profile::Music:
            return {1.0f, 0.28f, 0.25f, false, 0.38f, 0.32f};
        case Profile::Speech:
            // Podcast / audiobook: dialogue lifted and compressed, but the stereo
            // image of the source is kept - a YouTube talk still has music and
            // effects at the sides, and collapsing them to mono is not "cleaner",
            // it is just flat. The room follows the user's switch.
            return {1.20f, 0.26f, 0.22f, true, 0.45f, 0.30f};
        case Profile::Game:
        default:
            return {1.0f, 0.30f, 0.22f, false, 0.45f, 0.40f};
    }
}

std::vector<float> convolve_poly(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size() + b.size() - 1, 0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            out[i + j] += a[i] * b[j];
        }
    }
    return out;
}

// Deterministic xorshift32 PRNG + Box-Muller for synthetic room tail.
class Rng {
public:
    explicit Rng(uint32_t seed) : state_(seed ? seed : 1) {}
    uint32_t next_u32() {
        uint32_t x = state_;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state_ = x;
        return x;
    }
    float uniform(float lo, float hi) {
        float t = static_cast<float>(next_u32() & 0xFFFFFF) / static_cast<float>(0x1000000);
        return lo + t * (hi - lo);
    }
    float normal() {
        float u1 = std::max(1e-7f, uniform(0.0f, 1.0f));
        float u2 = uniform(0.0f, 1.0f);
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * static_cast<float>(kPi) * u2);
    }

private:
    uint32_t state_;
};

std::vector<float> generate_room_response(float rt60, float sample_rate, uint32_t seed) {
    const int length = static_cast<int>(kRoomSeconds * sample_rate);
    std::vector<float> resp(4 * length, 0.0f); // (2 sends, 2 ears, length)

    BiquadFilter low_f, air_f;
    low_f.configure_lowpass_2nd(2500.0f, sample_rate, 1);
    air_f.configure_lowpass_2nd(9000.0f, sample_rate, 1);

    Rng rng(seed);
    std::vector<float> noise(length);
    std::vector<float> low(length);
    std::vector<float> tail(length);

    double energy_sum = 0.0;
    for (int send = 0; send < 2; ++send) {
        for (int ear = 0; ear < 2; ++ear) {
            low_f.reset();
            air_f.reset();
            for (int i = 0; i < length; ++i) {
                noise[i] = rng.normal();
            }
            low_f.process(noise.data(), low.data(), 0, length);
            for (int i = 0; i < length; ++i) {
                float t = static_cast<float>(i) / sample_rate;
                float high = noise[i] - low[i];
                float d_low = std::exp(-6.91f * t / rt60);
                float d_high = std::exp(-6.91f * t / (rt60 * 0.45f));
                tail[i] = low[i] * d_low + high * d_high;
            }
            air_f.process_inplace(tail.data(), 0, length);
            for (int i = 0; i < length; ++i) {
                float t = static_cast<float>(i) / sample_rate;
                float onset = std::clamp((t - 0.008f) / 0.02f, 0.0f, 1.0f);
                tail[i] *= onset;
            }
            bool same_side = (send == ear);
            for (int r = 0; r < 10; ++r) {
                float delay = rng.uniform(0.003f, 0.03f);
                float gain = rng.uniform(0.3f, 0.7f) * (same_side ? 1.0f : 0.55f) * (1.0f - delay / 0.04f);
                float sign = (rng.next_u32() & 1) ? 1.0f : -1.0f;
                int idx = std::min(length - 1, static_cast<int>(delay * sample_rate));
                tail[idx] += gain * sign * 6.0f;
            }
            float* dst = &resp[(send * 2 + ear) * length];
            double path_e = 0.0;
            for (int i = 0; i < length; ++i) {
                dst[i] = tail[i];
                path_e += static_cast<double>(tail[i]) * tail[i];
            }
            energy_sum += path_e;
        }
    }
    float norm = static_cast<float>(std::sqrt(energy_sum / 4.0 + 1e-12));
    for (float& v : resp) {
        v /= norm;
    }
    return resp;
}

} // namespace

// --- BiquadFilter ---

void BiquadFilter::configure_lowpass_2nd(float cutoff_hz, float sample_rate, int channels) {
    double w0 = 2.0 * kPi * cutoff_hz / sample_rate;
    double c = std::cos(w0);
    double s = std::sin(w0);
    double alpha = s / std::sqrt(2.0);
    double a0 = 1.0 + alpha;
    b_ = {
        static_cast<float>(((1.0 - c) * 0.5) / a0),
        static_cast<float>((1.0 - c) / a0),
        static_cast<float>(((1.0 - c) * 0.5) / a0),
    };
    a_ = {1.0f, static_cast<float>((-2.0 * c) / a0), static_cast<float>((1.0 - alpha) / a0)};
    channels_ = channels;
    order_ = 2;
    state_.assign(channels_ * order_, 0.0f);
}

void BiquadFilter::configure_highpass_2nd(float cutoff_hz, float sample_rate, int channels) {
    double w0 = 2.0 * kPi * cutoff_hz / sample_rate;
    double c = std::cos(w0);
    double s = std::sin(w0);
    double alpha = s / std::sqrt(2.0);
    double a0 = 1.0 + alpha;
    b_ = {
        static_cast<float>(((1.0 + c) * 0.5) / a0),
        static_cast<float>(-(1.0 + c) / a0),
        static_cast<float>(((1.0 + c) * 0.5) / a0),
    };
    a_ = {1.0f, static_cast<float>((-2.0 * c) / a0), static_cast<float>((1.0 - alpha) / a0)};
    channels_ = channels;
    order_ = 2;
    state_.assign(channels_ * order_, 0.0f);
}

void BiquadFilter::configure_high_shelf(float freq_hz, float gain_db, float sample_rate,
                                       int channels) {
    const double A = std::pow(10.0, gain_db / 40.0);
    const double w0 = 2.0 * kPi * freq_hz / sample_rate;
    const double c = std::cos(w0);
    const double s = std::sin(w0);
    const double alpha = s / 2.0 * std::sqrt(2.0);
    const double two_sqrt_a_alpha = 2.0 * std::sqrt(A) * alpha;
    const double a0 = (A + 1.0) - (A - 1.0) * c + two_sqrt_a_alpha;
    b_ = {
        static_cast<float>((A * ((A + 1.0) + (A - 1.0) * c + two_sqrt_a_alpha)) / a0),
        static_cast<float>((-2.0 * A * ((A - 1.0) + (A + 1.0) * c)) / a0),
        static_cast<float>((A * ((A + 1.0) + (A - 1.0) * c - two_sqrt_a_alpha)) / a0),
    };
    a_ = {
        1.0f,
        static_cast<float>((2.0 * ((A - 1.0) - (A + 1.0) * c)) / a0),
        static_cast<float>(((A + 1.0) - (A - 1.0) * c - two_sqrt_a_alpha) / a0),
    };
    channels_ = channels;
    order_ = 2;
    state_.assign(channels_ * order_, 0.0f);
}

void BiquadFilter::configure_highpass_1st(float cutoff_hz, float sample_rate, int channels) {
    double k = std::tan(kPi * cutoff_hz / sample_rate);
    double a0 = 1.0 + k;
    b_ = {static_cast<float>(1.0 / a0), static_cast<float>(-1.0 / a0)};
    a_ = {1.0f, static_cast<float>((k - 1.0) / a0)};
    channels_ = channels;
    order_ = 1;
    state_.assign(channels_ * order_, 0.0f);
}

void BiquadFilter::reset() {
    std::fill(state_.begin(), state_.end(), 0.0f);
}

void BiquadFilter::process(const float* in, float* out, int ch, int frames) {
    float* z = &state_[ch * order_];
    const int nb = static_cast<int>(b_.size());
    const int na = static_cast<int>(a_.size());
    for (int i = 0; i < frames; ++i) {
        float x = in[i];
        float y = b_[0] * x + (order_ > 0 ? z[0] : 0.0f);
        for (int k = 1; k < order_; ++k) {
            float bk = (k < nb) ? b_[k] : 0.0f;
            float ak = (k < na) ? a_[k] : 0.0f;
            z[k - 1] = z[k] + bk * x - ak * y;
        }
        if (order_ > 0) {
            float blast = (order_ < nb) ? b_[order_] : 0.0f;
            float alast = (order_ < na) ? a_[order_] : 0.0f;
            z[order_ - 1] = blast * x - alast * y;
        }
        out[i] = y;
    }
}

void BiquadFilter::process_inplace(float* buf, int ch, int frames) {
    process(buf, buf, ch, frames);
}

// --- DelayLine ---

void DelayLine::init(int channels, int delay_samples) {
    channels_ = channels;
    delay_ = std::max(1, delay_samples);
    pos_ = 0;
    buffer_.assign(channels_ * delay_, 0.0f);
}

void DelayLine::reset() {
    pos_ = 0;
    std::fill(buffer_.begin(), buffer_.end(), 0.0f);
}

void DelayLine::process(const float* in, float* out, int frames) {
    for (int i = 0; i < frames; ++i) {
        for (int c = 0; c < channels_; ++c) {
            float& slot = buffer_[c * delay_ + pos_];
            float delayed = slot;
            slot = in[c * frames + i];
            out[c * frames + i] = delayed;
        }
        pos_ = (pos_ + 1) % delay_;
    }
}

// --- Convolver ---

void Convolver::init(int sources, int ears, int length, const float* responses, bool diagonal) {
    sources_ = sources;
    ears_ = ears;
    diagonal_ = diagonal;
    parts_ = (length + kHop - 1) / kHop;
    const size_t bins = fft_.bins();
    spectra_.assign(parts_ * sources_ * ears_ * bins, {0.0f, 0.0f});
    history_.assign(parts_ * sources_ * bins, {0.0f, 0.0f});
    previous_.assign(sources_ * kHop, 0.0f);
    accum_.assign(ears_ * bins, {0.0f, 0.0f});
    frame_buf_.assign(2 * kHop, 0.0f);
    set_responses(length, responses);
}

void Convolver::set_responses(int length, const float* responses) {
    const size_t bins = fft_.bins();
    std::vector<float> part_buf(2 * kHop, 0.0f);
    for (int s = 0; s < sources_; ++s) {
        for (int e = 0; e < ears_; ++e) {
            const float* src = &responses[(s * ears_ + e) * length];
            for (int p = 0; p < parts_; ++p) {
                std::fill(part_buf.begin(), part_buf.end(), 0.0f);
                int start = p * kHop;
                int count = std::min(kHop, length - start);
                if (count > 0) {
                    std::memcpy(part_buf.data(), src + start, count * sizeof(float));
                }
                std::complex<float>* dst = &spectra_[((p * sources_ + s) * ears_ + e) * bins];
                fft_.forward(part_buf.data(), dst);
            }
        }
    }
}

void Convolver::reset() {
    hist_pos_ = 0;
    std::fill(history_.begin(), history_.end(), std::complex<float>{0.0f, 0.0f});
    std::fill(previous_.begin(), previous_.end(), 0.0f);
}

// Complex multiply-accumulate over an interleaved spectrum.
// Written on raw floats instead of std::complex: with std::complex the compiler
// keeps the operands in separate scalars and refuses to vectorise, which costs
// ~2.8x on the partitioned convolution (it is 84% of the DSP budget).
static inline void complex_mac(std::complex<float>* acc, const std::complex<float>* h,
                               const std::complex<float>* sp, size_t bins) {
    float* a = reinterpret_cast<float*>(acc);
    const float* x = reinterpret_cast<const float*>(h);
    const float* y = reinterpret_cast<const float*>(sp);
    for (size_t b = 0; b < bins; ++b) {
        const float xr = x[2 * b], xi = x[2 * b + 1];
        const float yr = y[2 * b], yi = y[2 * b + 1];
        a[2 * b] += xr * yr - xi * yi;
        a[2 * b + 1] += xr * yi + xi * yr;
    }
}

void Convolver::process(const float* in, float* out) {
    const size_t bins = fft_.bins();
    // History lives in a ring: shifting it down a partition every hop moved
    // 348 KB per block for the room convolver and cost more than the multiply.
    const size_t slice = sources_ * bins;
    std::complex<float>* newest = &history_[static_cast<size_t>(hist_pos_) * slice];
    for (int s = 0; s < sources_; ++s) {
        std::memcpy(frame_buf_.data(), &previous_[s * kHop], kHop * sizeof(float));
        std::memcpy(frame_buf_.data() + kHop, &in[s * kHop], kHop * sizeof(float));
        std::memcpy(&previous_[s * kHop], &in[s * kHop], kHop * sizeof(float));
        fft_.forward(frame_buf_.data(), &newest[s * bins]);
    }

    std::fill(accum_.begin(), accum_.end(), std::complex<float>{0.0f, 0.0f});
    // Walk the ring backwards from the newest frame: partition p lives in slot
    // (hist_pos_ - p). One wrap-around, no per-iteration modulo.
    int slot = hist_pos_;
    if (diagonal_) {
        for (int p = 0; p < parts_; ++p) {
            for (int s = 0; s < sources_; ++s) {
                const auto* h = &history_[static_cast<size_t>(slot) * slice + s * bins];
                const auto* sp = &spectra_[((p * sources_ + s) * ears_ + 0) * bins];
                complex_mac(&accum_[s * bins], h, sp, bins);
            }
            if (--slot < 0) slot = parts_ - 1;
        }
    } else {
        for (int p = 0; p < parts_; ++p) {
            for (int s = 0; s < sources_; ++s) {
                const auto* h = &history_[static_cast<size_t>(slot) * slice + s * bins];
                for (int e = 0; e < ears_; ++e) {
                    const auto* sp = &spectra_[((p * sources_ + s) * ears_ + e) * bins];
                    complex_mac(&accum_[e * bins], h, sp, bins);
                }
            }
            if (--slot < 0) slot = parts_ - 1;
        }
    }
    hist_pos_ = (hist_pos_ + 1) % parts_;

    for (int e = 0; e < ears_; ++e) {
        fft_.inverse(&accum_[e * bins], frame_buf_.data());
        std::memcpy(&out[e * kHop], frame_buf_.data() + kHop, kHop * sizeof(float));
    }
}

// --- Dynamics ---

void Dynamics::init(float sample_rate) {
    sample_rate_ = sample_rate;
    k_high_.configure_highpass_2nd(100.0f, sample_rate, 2);
    k_shelf_.configure_highpass_1st(2000.0f, sample_rate, 2);
    lookahead_.init(2, kHop);
    momentary_ = std::exp(-static_cast<float>(kHop) / (sample_rate * 0.4f));
    grow_ = std::exp(-static_cast<float>(kHop) / (sample_rate * 10.0f));
    fall_ = std::exp(-static_cast<float>(kHop) / (sample_rate * 8.0f));
    relax_ = std::exp(-static_cast<float>(kHop) / (sample_rate * 0.5f));
    attack_ = std::exp(-static_cast<float>(kHop) / (sample_rate * 0.003f));
    release_ = std::exp(-static_cast<float>(kHop) / (sample_rate * 0.120f));
    reset();
}

void Dynamics::reset() {
    k_high_.reset();
    k_shelf_.reset();
    lookahead_.reset();
    mean_square_ = std::pow(10.0f, -20.0f / 10.0f);
    level_gain_db_ = 0.0f;
    envelope_db_ = -90.0f;
    pending_ = 1.0f;
    gain_ = 1.0f;
}

void Dynamics::update_leveler(const float* left, const float* right, bool enabled) {
    if (!enabled) {
        level_gain_db_ *= relax_;
        return;
    }
    k_high_.process(left, w_left_.data(), 0, kHop);
    k_high_.process(right, w_right_.data(), 1, kHop);
    k_shelf_.process(w_left_.data(), s_left_.data(), 0, kHop);
    k_shelf_.process(w_right_.data(), s_right_.data(), 1, kHop);

    double sum_l = 0.0, sum_r = 0.0;
    for (int i = 0; i < kHop; ++i) {
        float wl = w_left_[i] + 0.585f * s_left_[i];
        float wr = w_right_[i] + 0.585f * s_right_[i];
        sum_l += wl * wl;
        sum_r += wr * wr;
    }
    float power = static_cast<float>((sum_l + sum_r) / static_cast<double>(kHop));
    mean_square_ = momentary_ * mean_square_ + (1.0f - momentary_) * power;
    float loudness = -0.691f + 10.0f * std::log10(mean_square_ + 1e-12f);
    if (loudness <= -50.0f) {
        return;
    }
    float wanted = std::clamp(-20.0f - loudness, -12.0f, 12.0f);
    float coeff = (wanted > level_gain_db_) ? grow_ : fall_;
    level_gain_db_ = coeff * level_gain_db_ + (1.0f - coeff) * wanted;
}

float Dynamics::compute_compressor(const float* left, const float* right, float gain_db) {
    double sum = 0.0;
    for (int i = 0; i < kHop; ++i) {
        sum += left[i] * left[i] + right[i] * right[i];
    }
    float ms = static_cast<float>(sum / (2.0 * kHop));
    float level = 10.0f * std::log10(ms + 1e-12f) + gain_db;
    float coeff = (level > envelope_db_) ? attack_ : release_;
    envelope_db_ = coeff * envelope_db_ + (1.0f - coeff) * level;
    constexpr float kThresholdDb = -14.0f;
    constexpr float kRatio = 3.0f;
    constexpr float kKneeDb = 6.0f;
    float over = envelope_db_ - kThresholdDb;
    float half = kKneeDb * 0.5f;
    float slope = 1.0f - 1.0f / kRatio;
    if (over <= -half) return 0.0f;
    if (over < half) return slope * (over + half) * (over + half) / (2.0f * kKneeDb);
    return slope * over;
}

void Dynamics::process(float* left, float* right, bool leveler, bool compress) {
    update_leveler(left, right, leveler);
    float gain_db = level_gain_db_;
    if (compress) {
        gain_db -= compute_compressor(left, right, gain_db);
    }
    float incoming = std::pow(10.0f, gain_db / 20.0f);

    std::memcpy(del_in_.data(), left, kHop * sizeof(float));
    std::memcpy(del_in_.data() + kHop, right, kHop * sizeof(float));
    lookahead_.process(del_in_.data(), del_out_.data(), kHop);

    float own_peak = 1e-12f;
    float next_peak = 1e-12f;
    for (int i = 0; i < 2 * kHop; ++i) {
        own_peak = std::max(own_peak, std::abs(del_out_[i]));
        next_peak = std::max(next_peak, std::abs(del_in_[i]));
    }
    constexpr float kCeiling = 0.89f;
    float end = std::min({pending_, kCeiling / own_peak, kCeiling / next_peak});
    if (end > gain_) {
        end = gain_ + (end - gain_) * 0.05f;
    }
    float step = (end - gain_) / static_cast<float>(std::max(1, kHop - 1));
    for (int i = 0; i < kHop; ++i) {
        float g = gain_ + step * static_cast<float>(i);
        float l = del_out_[i] * g;
        float r = del_out_[kHop + i] * g;
        auto softclip = [](float x) {
            float ax = std::abs(x);
            if (ax > 0.95f) {
                float s = (x >= 0.0f) ? 1.0f : -1.0f;
                return s * (0.95f + 0.05f * std::tanh((ax - 0.95f) / 0.05f));
            }
            return x;
        };
        left[i] = softclip(l);
        right[i] = softclip(r);
    }
    gain_ = end;
    pending_ = incoming;
}

// --- CrosstalkWidener ---

void CrosstalkWidener::init(float sample_rate) {
    BiquadFilter hp, lp;
    hp.configure_highpass_2nd(250.0f, sample_rate, 1);
    lp.configure_lowpass_2nd(6000.0f, sample_rate, 1);
    auto b_bp = convolve_poly(hp.b_, lp.b_);
    auto a_bp = convolve_poly(hp.a_, lp.a_);

    constexpr float kDelay = 75e-6f;
    constexpr float kGain = 0.55f;
    std::vector<float> taps(16, 0.0f);
    for (int order = 1; order <= 2; ++order) {
        float weight = (order == 1) ? kGain : (kGain * kGain);
        float pos = order * kDelay * sample_rate;
        int whole = std::min(14, static_cast<int>(pos));
        float frac = pos - whole;
        taps[whole] += weight * (1.0f - frac);
        taps[whole + 1] += weight * frac;
    }
    filter_.b_ = convolve_poly(b_bp, taps);
    filter_.a_ = a_bp;
    filter_.channels_ = 1;
    filter_.order_ = static_cast<int>(std::max(filter_.a_.size(), filter_.b_.size())) - 1;
    filter_.state_.assign(filter_.order_, 0.0f);
}

void CrosstalkWidener::reset() {
    filter_.reset();
}

void CrosstalkWidener::process(float* left, float* right) {
    for (int i = 0; i < kHop; ++i) {
        side_[i] = 0.5f * (left[i] - right[i]);
    }
    filter_.process(side_.data(), extra_.data(), 0, kHop);
    for (int i = 0; i < kHop; ++i) {
        left[i] += extra_[i];
        right[i] -= extra_[i];
    }
}

// --- Renderer ---

void Renderer::init(float sample_rate) {
    sample_rate_ = sample_rate;

    // Flatten kHrirKu100 [7][2][256]
    direct_.init(kNumSpeakers, 2, kHrirLength, &kHrirKu100[0][0][0], false);

    room_game_ = generate_room_response(0.22f, sample_rate, 7);
    room_movie_ = generate_room_response(0.40f, sample_rate, 7);
    room_music_ = generate_room_response(0.25f, sample_rate, 7);

    const int room_len = static_cast<int>(kRoomSeconds * sample_rate);
    room_.init(2, 2, room_len, room_game_.data(), false);

    ambience_delay_.init(2, static_cast<int>(kAmbienceDelaySec * sample_rate));
    speaker_delay_.init(1, static_cast<int>(0.012f * sample_rate));
    lfe_filter_.configure_lowpass_2nd(120.0f, sample_rate, 1);
    speaker_lfe_.configure_lowpass_2nd(120.0f, sample_rate, 1);
    glasses_bass_.configure_lowpass_2nd(220.0f, sample_rate, 2);
    cinema_air_.configure_highpass_2nd(4500.0f, sample_rate, 2);
    // Speech chain: rumble out of the way, presence lifted so consonants cut through
    // on a phone speaker, nothing added on top.
    speech_hp_.configure_highpass_2nd(95.0f, sample_rate, 2);
    speech_presence_.configure_high_shelf(2800.0f, 4.5f, sample_rate, 2);
    widener_.init(sample_rate);
    dynamics_.init(sample_rate);

    set_profile(profile_);
    reset();
}

void Renderer::reset() {
    direct_.reset();
    room_.reset();
    ambience_delay_.reset();
    speaker_delay_.reset();
    lfe_filter_.reset();
    speaker_lfe_.reset();
    glasses_bass_.reset();
    cinema_air_.reset();
    widener_.reset();
    dynamics_.reset();
    clock_ = 0.0;
    surround_seen_ = -1e9;
    is_surround_ = false;
    fifo_fill_ = 0;
    out_read_pos_ = 0;
    out_avail_ = 0;
}

void Renderer::set_profile(Profile p) {
    profile_ = p;
    const int room_len = static_cast<int>(kRoomSeconds * sample_rate_);
    switch (p) {
        case Profile::Movie:
            room_.set_responses(room_len, room_movie_.data());
            break;
        case Profile::Music:
            room_.set_responses(room_len, room_music_.data());
            break;
        case Profile::Speech:
            // Speech borrows the short game room: a podcast should sound like a
            // small room, not a hall, but it must be audible when switched on.
            room_.set_responses(room_len, room_game_.data());
            break;
        case Profile::Game:
        default:
            room_.set_responses(room_len, room_game_.data());
            break;
    }
}

void Renderer::set_device(DeviceKind d) {
    device_ = d;
}

const char* Renderer::ort_library_path() const {
    if (runtime_dir_.empty()) return nullptr;
#ifdef __APPLE__
    ort_library_ = runtime_dir_ + "/libonnxruntime.dylib";
#else
    ort_library_ = runtime_dir_ + "/libonnxruntime.so";
#endif
    return ort_library_.c_str();
}

void Renderer::stats(float* model_ms, uint64_t* drops) {
#ifdef ALLY_ENABLE_NPU
    vocals_.stats(model_ms, drops);
#else
    if (model_ms) *model_ms = 0.0f;
    if (drops) *drops = 0;
#endif
}

bool Renderer::detect_surround() {
    clock_ += static_cast<double>(kHop) / static_cast<double>(sample_rate_);
    for (int ch = 2; ch < 8; ++ch) {
        double sum = 0.0;
        const float* p = &block_[ch * kHop];
        for (int i = 0; i < kHop; ++i) {
            sum += p[i] * p[i];
        }
        if (std::sqrt(sum / kHop) > kSilence) {
            surround_seen_ = clock_;
            break;
        }
    }
    is_surround_ = (clock_ - surround_seen_) < kSurroundHold;
    return is_surround_;
}

void Renderer::make_stereo_feeds(bool& has_lfe) {
    const auto cfg = get_profile_params(profile_);
    const float* fl = &block_[0 * kHop];
    const float* fr = &block_[1 * kHop];
    has_lfe = false;

    // Whether the source carries a stereo image is decided on the input, not on
    // the separated signal: a vocal model can centre the voice hard, and a
    // speech profile must not mistake that for a mono podcast.
    double source_difference = 0.0;
    double source_magnitude = 0.0;
    for (int i = 0; i < kHop; ++i) {
        source_difference += std::fabs(fl[i] - fr[i]);
        source_magnitude += std::fabs(fl[i]) + std::fabs(fr[i]);
    }
    const bool source_is_mono = source_difference < 0.02 * source_magnitude + 1e-9;

    std::array<float, 2 * kHop> amb{};
    std::array<float, 2 * kHop> rear{};

#ifdef ALLY_ENABLE_NPU
    if (neural_ && profile_ == Profile::Speech &&
        vocals_.ensure_loaded(speech_model_.c_str(), true, ort_library_path())) {
        // Speech model: keep the voice, tuck everything else away. No room, no air -
        // the stereo path below only has to place it.
        std::array<float, 2 * kHop> dry{};
        std::array<float, 2 * kHop> sep_rest{};
        vocals_.process(&block_[0], dry.data(), sep_rest.data(), tick_);
        for (int i = 0; i < 2 * kHop; ++i) {
            const float voice = dry[i] - sep_rest[i];
            block_[i] = voice * 1.15f + sep_rest[i] * 0.70f;
        }
    }
    if (neural_ && profile_ == Profile::Movie &&
        vocals_.ensure_loaded(vocals_model_.c_str(), true, ort_library_path())) {
        std::array<float, 2 * kHop> dry{};
        std::array<float, 2 * kHop> sep_rest{};
        vocals_.process(&block_[0], dry.data(), sep_rest.data(), tick_);

        std::array<float, kHop> air_l{}, air_r{};
        cinema_air_.process(&sep_rest[0], air_l.data(), 0, kHop);
        cinema_air_.process(&sep_rest[kHop], air_r.data(), 1, kHop);

        for (int i = 0; i < kHop; ++i) {
            float vl = (dry[i] - sep_rest[i]) * 0.75f;
            float vr = (dry[kHop + i] - sep_rest[kHop + i]) * 0.75f;
            float al = air_l[i] * 0.26f;
            float ar = air_r[i] * 0.26f;
            float rl = dry[i] - vl + al;
            float rr = dry[kHop + i] - vr + ar;
            float amb_l = sep_rest[i] + al;
            float amb_r = sep_rest[kHop + i] + ar;
            feeds_[0 * kHop + i] = rl;
            feeds_[1 * kHop + i] = rr;
            feeds_[2 * kHop + i] = 0.5f * (vl + vr) * cfg.dialogue;
            amb[0 * kHop + i] = amb_l - 0.7f * amb_r;
            amb[1 * kHop + i] = amb_r - 0.7f * amb_l;
            lfe_[i] = 0.5f * (sep_rest[i] + sep_rest[kHop + i]) * 0.25f;
        }
        has_lfe = true;
        ambience_delay_.process(amb.data(), rear.data(), kHop);
        for (int i = 0; i < kHop; ++i) {
            feeds_[3 * kHop + i] = rear[0 * kHop + i] * 0.52f;
            feeds_[4 * kHop + i] = rear[1 * kHop + i] * 0.52f;
            feeds_[5 * kHop + i] = amb[0 * kHop + i] * 0.58f;
            feeds_[6 * kHop + i] = amb[1 * kHop + i] * 0.58f;
        }
        return;
    }
#endif

    for (int i = 0; i < kHop; ++i) {
        float mid = 0.5f * (fl[i] + fr[i]);
        float voice = 0.5f * mid;
        float rest_l = fl[i] - voice;
        float rest_r = fr[i] - voice;
        feeds_[0 * kHop + i] = rest_l;
        feeds_[1 * kHop + i] = rest_r;
        feeds_[2 * kHop + i] = voice * cfg.dialogue;
        amb[0 * kHop + i] = rest_l - 0.7f * rest_r;
        amb[1 * kHop + i] = rest_r - 0.7f * rest_l;
    }

    // Podcasts, YouTube talks and audiobooks are usually mono: there is nothing
    // to place, so the voice goes dead centre and the ambience spread is
    // skipped instead of inventing a stereo image from nothing.
    if (profile_ == Profile::Speech && source_is_mono) {
        {
            for (int i = 0; i < kHop; ++i) {
                const float mid = 0.5f * (fl[i] + fr[i]);
                feeds_[0 * kHop + i] = 0.0f;
                feeds_[1 * kHop + i] = 0.0f;
                feeds_[2 * kHop + i] = mid * cfg.dialogue;
                feeds_[3 * kHop + i] = 0.0f;
                feeds_[4 * kHop + i] = 0.0f;
                feeds_[5 * kHop + i] = 0.0f;
                feeds_[6 * kHop + i] = 0.0f;
            }
            return;
        }
    }
    ambience_delay_.process(amb.data(), rear.data(), kHop);
    for (int i = 0; i < kHop; ++i) {
        feeds_[3 * kHop + i] = rear[0 * kHop + i] * cfg.rear_gain;
        feeds_[4 * kHop + i] = rear[1 * kHop + i] * cfg.rear_gain;
        feeds_[5 * kHop + i] = amb[0 * kHop + i] * cfg.side_gain;
        feeds_[6 * kHop + i] = amb[1 * kHop + i] * cfg.side_gain;
    }
}

void Renderer::make_surround_feeds(bool& has_lfe) {
    const auto cfg = get_profile_params(profile_);
    // block_ channels: 0=FL, 1=FR, 2=FC, 3=LFE, 4=RL, 5=RR, 6=SL, 7=SR
    // feeds_ channels: 0=FL, 1=FR, 2=FC, 3=RL, 4=RR, 5=SL, 6=SR
    std::memcpy(&feeds_[0 * kHop], &block_[0 * kHop], kHop * sizeof(float));
    std::memcpy(&feeds_[1 * kHop], &block_[1 * kHop], kHop * sizeof(float));
    for (int i = 0; i < kHop; ++i) {
        feeds_[2 * kHop + i] = block_[2 * kHop + i] * cfg.dialogue;
    }
    std::memcpy(&feeds_[3 * kHop], &block_[4 * kHop], kHop * sizeof(float));
    std::memcpy(&feeds_[4 * kHop], &block_[5 * kHop], kHop * sizeof(float));
    std::memcpy(&feeds_[5 * kHop], &block_[6 * kHop], kHop * sizeof(float));
    std::memcpy(&feeds_[6 * kHop], &block_[7 * kHop], kHop * sizeof(float));

    double side_e = 0.0;
    for (int i = 0; i < kHop; ++i) {
        side_e += block_[6 * kHop + i] * block_[6 * kHop + i] + block_[7 * kHop + i] * block_[7 * kHop + i];
    }
    if (std::sqrt(side_e / (2.0 * kHop)) <= kSilence) {
        // 5.1: move RL/RR into SL/SR (~110 deg) and zero RL/RR
        std::memcpy(&feeds_[5 * kHop], &feeds_[3 * kHop], 2 * kHop * sizeof(float));
        std::memset(&feeds_[3 * kHop], 0, 2 * kHop * sizeof(float));
    }
    std::memcpy(lfe_.data(), &block_[3 * kHop], kHop * sizeof(float));
    has_lfe = true;
}

void Renderer::apply_speech_eq(float* out_l, float* out_r) {
    if (profile_ != Profile::Speech) return;
    std::array<float, kHop> tmp{};
    speech_hp_.process(out_l, tmp.data(), 0, kHop);
    speech_presence_.process(tmp.data(), out_l, 0, kHop);
    std::fill(tmp.begin(), tmp.end(), 0.0f);
    speech_hp_.process(out_r, tmp.data(), 1, kHop);
    speech_presence_.process(tmp.data(), out_r, 1, kHop);
}

void Renderer::render_headphones(bool has_lfe, float* out_l, float* out_r) {
    const auto cfg = get_profile_params(profile_);
    direct_.process(feeds_.data(), ears_.data());

    if (reverb_) {
        for (int i = 0; i < kHop; ++i) {
            room_send_[0 * kHop + i] = 0.5f * (feeds_[0 * kHop + i] + 0.7f * feeds_[2 * kHop + i] + feeds_[5 * kHop + i] + feeds_[3 * kHop + i]);
            room_send_[1 * kHop + i] = 0.5f * (feeds_[1 * kHop + i] + 0.7f * feeds_[2 * kHop + i] + feeds_[6 * kHop + i] + feeds_[4 * kHop + i]);
        }
        room_.process(room_send_.data(), room_out_.data());
        for (int i = 0; i < 2 * kHop; ++i) {
            ears_[i] += room_out_[i] * cfg.room;
        }
    }
    if (has_lfe) {
        std::array<float, kHop> low{};
        lfe_filter_.process(lfe_.data(), low.data(), 0, kHop);
        for (int i = 0; i < kHop; ++i) {
            float l = low[i] * 1.4f;
            ears_[i] += l;
            ears_[kHop + i] += l;
        }
    }
    for (int i = 0; i < kHop; ++i) {
        out_l[i] = ears_[i] * 0.85f;
        out_r[i] = ears_[kHop + i] * 0.85f;
    }
    if (device_ == DeviceKind::Glasses) {
        std::array<float, kHop> bass_l{}, bass_r{};
        glasses_bass_.process(out_l, bass_l.data(), 0, kHop);
        glasses_bass_.process(out_r, bass_r.data(), 1, kHop);
        for (int i = 0; i < kHop; ++i) {
            out_l[i] += bass_l[i] * kGlassesBassGain;
            out_r[i] += bass_r[i] * kGlassesBassGain;
        }
    }
    apply_speech_eq(out_l, out_r);
}

void Renderer::render_speakers(bool has_lfe, float* out_l, float* out_r) {
    const auto cfg = get_profile_params(profile_);
    std::array<float, kHop> s_mid{}, late{};
    for (int i = 0; i < kHop; ++i) {
        float sl = feeds_[5 * kHop + i] + 0.8f * feeds_[3 * kHop + i];
        float sr = feeds_[6 * kHop + i] + 0.8f * feeds_[4 * kHop + i];
        s_mid[i] = 0.5f * (sl + sr);
    }
    speaker_delay_.process(s_mid.data(), late.data(), kHop);
    for (int i = 0; i < kHop; ++i) {
        float voice = 0.707f * feeds_[2 * kHop + i];
        float sl = feeds_[5 * kHop + i] + 0.8f * feeds_[3 * kHop + i];
        float sr = feeds_[6 * kHop + i] + 0.8f * feeds_[4 * kHop + i];
        float s_side = 0.75f * (sl - sr);
        float lt = late[i] * 0.6f;
        out_l[i] = feeds_[0 * kHop + i] + voice + lt + s_side;
        out_r[i] = feeds_[1 * kHop + i] + voice + lt - s_side;
    }
    widener_.process(out_l, out_r);
    if (reverb_) {
        for (int i = 0; i < kHop; ++i) {
            room_send_[i] = out_l[i] * 0.5f;
            room_send_[kHop + i] = out_r[i] * 0.5f;
        }
        room_.process(room_send_.data(), room_out_.data());
        float rg = cfg.room * 0.5f;
        for (int i = 0; i < kHop; ++i) {
            out_l[i] += room_out_[i] * rg;
            out_r[i] += room_out_[kHop + i] * rg;
        }
    }
    if (has_lfe) {
        std::array<float, kHop> low{};
        speaker_lfe_.process(lfe_.data(), low.data(), 0, kHop);
        for (int i = 0; i < kHop; ++i) {
            float l = low[i] * 0.7f;
            out_l[i] += l;
            out_r[i] += l;
        }
    }
    apply_speech_eq(out_l, out_r);
}

void Renderer::downmix(float* out_l, float* out_r) {
    if (!is_surround_) {
        std::memcpy(out_l, &block_[0 * kHop], kHop * sizeof(float));
        std::memcpy(out_r, &block_[1 * kHop], kHop * sizeof(float));
        return;
    }
    for (int i = 0; i < kHop; ++i) {
        float c = 0.707f * block_[2 * kHop + i];
        out_l[i] = (block_[0 * kHop + i] + c + 0.707f * (block_[6 * kHop + i] + block_[4 * kHop + i])) * 0.6f;
        out_r[i] = (block_[1 * kHop + i] + c + 0.707f * (block_[7 * kHop + i] + block_[5 * kHop + i])) * 0.6f;
    }
}

void Renderer::process_hop() {
    tick_++;
    if (profile_ == Profile::Off) {
        std::memcpy(out_fifo_[0].data(), &block_[0 * kHop], kHop * sizeof(float));
        std::memcpy(out_fifo_[1].data(), &block_[1 * kHop], kHop * sizeof(float));
        out_read_pos_ = 0;
        out_avail_ = kHop;
        return;
    }

    // Silence gate. The process tap keeps delivering zeroed buffers when no app
    // is playing, and running the separator plus seven HRTF convolutions on that
    // costs about a third of a core for nothing. The hangover is long enough for
    // the room tail to decay before the renderer goes quiet, so there is no cut.
    constexpr int kSilenceHangoverHops = 220;  // ~0.6 s at 48 kHz
    float block_peak = 0.0f;
    for (int c = 0; c < 2; ++c) {
        for (int i = 0; i < kHop; ++i) {
            const float v = std::fabs(block_[c * kHop + i]);
            if (v > block_peak) block_peak = v;
        }
    }
    if (block_peak < kSilence) {
        if (silent_hops_ < kSilenceHangoverHops) ++silent_hops_;
    } else if (silent_hops_ >= kSilenceHangoverHops) {
        // Coming back from a long gap: drop stale filter and model state so the
        // first block after silence does not carry the tail of the old one.
        reset();
        silent_hops_ = 0;
    } else {
        silent_hops_ = 0;
    }
    if (silent_hops_ >= kSilenceHangoverHops) {
        std::fill(out_fifo_[0].begin(), out_fifo_[0].end(), 0.0f);
        std::fill(out_fifo_[1].begin(), out_fifo_[1].end(), 0.0f);
        out_read_pos_ = 0;
        out_avail_ = kHop;
        return;
    }

    bool surround = detect_surround();
    float* out_l = out_fifo_[0].data();
    float* out_r = out_fifo_[1].data();

    if (bypass_) {
        downmix(out_l, out_r);
    } else {
        bool has_lfe = false;
        if (surround) {
            make_surround_feeds(has_lfe);
        } else {
            make_stereo_feeds(has_lfe);
        }
        float trim = 1.0f;
        if (device_ == DeviceKind::Speakers) {
            trim = surround ? kSpeakerTrimSurround : kSpeakerTrimStereo;
            render_speakers(has_lfe, out_l, out_r);
        } else {
            trim = surround ? kSurroundTrim : kStereoTrim;
            render_headphones(has_lfe, out_l, out_r);
        }
        for (int i = 0; i < kHop; ++i) {
            out_l[i] *= trim;
            out_r[i] *= trim;
        }
    }

    const auto cfg = get_profile_params(profile_);
    bool dynamic = cfg.compress || (device_ != DeviceKind::Headphones);
    dynamics_.process(out_l, out_r, dynamic, dynamic);
    out_read_pos_ = 0;
    out_avail_ = kHop;
}

void Renderer::process_interleaved(const float* in, float* out, int frames, int in_channels) {
    if (profile_ == Profile::Off && in_channels == 2) {
        std::memcpy(out, in, static_cast<size_t>(frames) * 2 * sizeof(float));
        return;
    }

    for (int i = 0; i < frames; ++i) {
        const float* src = in + i * in_channels;
        if (in_channels == 2) {
            in_fifo_[0][fifo_fill_] = src[0];
            in_fifo_[1][fifo_fill_] = src[1];
            for (int c = 2; c < 8; ++c) in_fifo_[c][fifo_fill_] = 0.0f;
        } else if (in_channels == 6) {
            // 5.1: FL FR FC LFE RL RR -> put RL/RR in 4/5, zero 6/7
            for (int c = 0; c < 6; ++c) in_fifo_[c][fifo_fill_] = src[c];
            in_fifo_[6][fifo_fill_] = 0.0f;
            in_fifo_[7][fifo_fill_] = 0.0f;
        } else {
            int copy_ch = std::min(8, in_channels);
            for (int c = 0; c < copy_ch; ++c) in_fifo_[c][fifo_fill_] = src[c];
            for (int c = copy_ch; c < 8; ++c) in_fifo_[c][fifo_fill_] = 0.0f;
        }
        fifo_fill_++;

        if (fifo_fill_ == kHop) {
            for (int c = 0; c < 8; ++c) {
                std::memcpy(&block_[c * kHop], in_fifo_[c].data(), kHop * sizeof(float));
            }
            process_hop();
            fifo_fill_ = 0;
        }

        if (out_avail_ > 0) {
            out[2 * i + 0] = out_fifo_[0][out_read_pos_];
            out[2 * i + 1] = out_fifo_[1][out_read_pos_];
            out_read_pos_++;
            out_avail_--;
        } else {
            // Initial kHop latency priming (2.67 ms at 48 kHz)
            out[2 * i + 0] = 0.0f;
            out[2 * i + 1] = 0.0f;
        }
    }
}

} // namespace ally
