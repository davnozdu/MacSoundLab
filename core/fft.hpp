#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace ally {

// Self-contained radix-2 real-to-complex FFT for power-of-two sizes (N = 256).
// Produces N/2 + 1 complex bins matching numpy.fft.rfft / irfft scaling.
//
// The butterflies work on interleaved float pairs rather than std::complex:
// std::complex keeps operands in separate scalars, so clang refuses to vectorise
// the inner loops, and this transform runs 9 times per 2.7 ms audio block.
class RealFFT {
public:
    explicit RealFFT(size_t n) : n_(n), half_(n / 2), rfft_twiddles_(2 * (n / 2)), scratch_(2 * (n / 2)),
                                 bitrev_(n / 2) {
        constexpr double kPi = 3.14159265358979323846;
        for (size_t k = 0; k < half_; ++k) {
            double angle = -2.0 * kPi * static_cast<double>(k) / static_cast<double>(n_);
            rfft_twiddles_[2 * k] = static_cast<float>(std::cos(angle));
            rfft_twiddles_[2 * k + 1] = static_cast<float>(std::sin(angle));
        }
        // Twiddles for the half-size complex FFT, indexed by butterfly offset.
        half_twiddles_.resize(2 * (half_ / 2 + 1));
        for (size_t k = 0; k <= half_ / 2; ++k) {
            double angle = -2.0 * kPi * static_cast<double>(k) / static_cast<double>(half_);
            half_twiddles_[2 * k] = static_cast<float>(std::cos(angle));
            half_twiddles_[2 * k + 1] = static_cast<float>(std::sin(angle));
        }
        // Bit-reversal table, built once instead of per transform.
        for (size_t i = 1, j = 0; i < half_; ++i) {
            size_t bit = half_ >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            bitrev_[i] = j;
        }
    }

    size_t size() const { return n_; }
    size_t bins() const { return half_ + 1; }

    // Forward real FFT: in[n] -> out[n/2 + 1]
    void forward(const float* in, std::complex<float>* out) const {
        float* buf = scratch_.data();
        for (size_t i = 0; i < half_; ++i) {
            buf[2 * i] = in[2 * i];
            buf[2 * i + 1] = in[2 * i + 1];
        }
        fft_complex(buf, half_, false);

        const float r0 = buf[0], i0 = buf[1];
        out[0] = {r0 + i0, 0.0f};
        out[half_] = {r0 - i0, 0.0f};

        for (size_t k = 1; k < half_; ++k) {
            // z[k] and conj(z[half-k]) give the even/odd (real/imag) split.
            const float zr = buf[2 * k], zi = buf[2 * k + 1];
            const float mr = buf[2 * (half_ - k)], mi = -buf[2 * (half_ - k) + 1];
            const float er = 0.5f * (zr + mr), ei = 0.5f * (zi + mi);
            const float dr = 0.5f * (zr - mr), di = 0.5f * (zi - mi);
            const float o_r = di, o_i = -dr;  // j * odd
            const float wr = rfft_twiddles_[2 * k], wi = rfft_twiddles_[2 * k + 1];
            out[k] = {er + (o_r * wr - o_i * wi), ei + (o_r * wi + o_i * wr)};
        }
    }

    // Inverse real FFT: in[n/2 + 1] -> out[n] (scaled by 1/n, matching numpy.fft.irfft)
    void inverse(const std::complex<float>* in, float* out) const {
        float* buf = scratch_.data();
        const float x0 = in[0].real();
        const float xn = in[half_].real();
        buf[0] = 0.5f * (x0 + xn);
        buf[1] = 0.5f * (x0 - xn);

        for (size_t k = 1; k < half_; ++k) {
            const float xr = in[k].real(), xi = in[k].imag();
            const float mr = in[half_ - k].real(), mi = -in[half_ - k].imag();
            const float er = 0.5f * (xr + mr), ei = 0.5f * (xi + mi);
            const float dr = 0.5f * (xr - mr), di = 0.5f * (xi - mi);
            // odd = conj(w) * odd_tw, then rotate by -j
            const float wr = rfft_twiddles_[2 * k], wi = -rfft_twiddles_[2 * k + 1];
            const float odr = dr * wr - di * wi;
            const float odi = dr * wi + di * wr;
            buf[2 * k] = er - odi;
            buf[2 * k + 1] = ei + odr;
        }

        fft_complex(buf, half_, true);
        const float inv = 1.0f / static_cast<float>(half_);
        for (size_t i = 0; i < half_; ++i) {
            out[2 * i] = buf[2 * i] * inv;
            out[2 * i + 1] = buf[2 * i + 1] * inv;
        }
    }

private:
    void fft_complex(float* data, size_t n, bool inv) const {
        for (size_t i = 1; i < n; ++i) {
            const size_t j = bitrev_[i];
            if (i < j) {
                std::swap(data[2 * i], data[2 * j]);
                std::swap(data[2 * i + 1], data[2 * j + 1]);
            }
        }
        for (size_t len = 2; len <= n; len <<= 1) {
            const size_t half = len >> 1;
            const size_t step = n / len;
            const float sign = inv ? -1.0f : 1.0f;  // twiddles carry the negative angle
            for (size_t i = 0; i < n; i += len) {
                for (size_t j = 0; j < half; ++j) {
                    const float wr = half_twiddles_[2 * (j * step)];
                    const float wi = sign * half_twiddles_[2 * (j * step) + 1];
                    float* a = data + 2 * (i + j);
                    float* b = data + 2 * (i + j + half);
                    const float vr = b[0] * wr - b[1] * wi;
                    const float vi = b[0] * wi + b[1] * wr;
                    const float ur = a[0], ui = a[1];
                    a[0] = ur + vr;
                    a[1] = ui + vi;
                    b[0] = ur - vr;
                    b[1] = ui - vi;
                }
            }
        }
    }

    size_t n_;
    size_t half_;
    std::vector<float> rfft_twiddles_;   // [half * 2] interleaved
    std::vector<float> half_twiddles_;   // [half/2 * 2] interleaved
    mutable std::vector<float> scratch_; // [half * 2] reused, no allocation per call
    std::vector<size_t> bitrev_;
};

} // namespace ally
