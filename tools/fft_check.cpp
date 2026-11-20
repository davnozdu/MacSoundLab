// NumPy parity for the FFT: writes forward bins and the round-trip error.
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

#include "fft.hpp"

int main() {
    constexpr size_t N = 256;
    ally::RealFFT fft(N);
    std::vector<float> in(N);
    for (size_t i = 0; i < N; ++i) in[i] = std::sin(0.07f * i) + 0.3f * std::cos(0.31f * i) - 0.012f * i;

    std::vector<std::complex<float>> spec(fft.bins());
    fft.forward(in.data(), spec.data());

    std::vector<float> back(N);
    fft.inverse(spec.data(), back.data());

    double err = 0.0, peak = 0.0;
    for (size_t i = 0; i < N; ++i) {
        err += std::pow(back[i] - in[i], 2);
        peak = std::max<double>(peak, std::fabs(in[i]));
    }
    std::printf("roundtrip rms error = %.3e (peak %.3f)\n", std::sqrt(err / N), peak);

    std::printf("bins (re,im) first 6:\n");
    for (size_t k = 0; k < 6; ++k) std::printf("  %zu % .6f % .6f\n", k, spec[k].real(), spec[k].imag());
    std::printf("bin 128 % .6f % .6f\n", spec[128].real(), spec[128].imag());

    // dump for numpy
    FILE* f = std::fopen("/tmp/fft_bins.txt", "w");
    for (size_t k = 0; k < fft.bins(); ++k) std::fprintf(f, "%.8f %.8f\n", spec[k].real(), spec[k].imag());
    std::fclose(f);
    FILE* g = std::fopen("/tmp/fft_in.txt", "w");
    for (size_t i = 0; i < N; ++i) std::fprintf(g, "%.8f\n", in[i]);
    std::fclose(g);
    return 0;
}
