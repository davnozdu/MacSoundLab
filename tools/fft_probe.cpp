#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>
#include "fft.hpp"

int main() {
    constexpr size_t N = 256;
    ally::RealFFT fft(N);
    std::vector<float> in(N, 0.0f);
    in[0] = 1.0f;  // impulse -> flat spectrum of 1s
    std::vector<std::complex<float>> spec(fft.bins());
    fft.forward(in.data(), spec.data());
    std::printf("impulse: ");
    for (int k = 0; k < 5; ++k) std::printf("(% .3f,% .3f) ", spec[k].real(), spec[k].imag());
    std::printf("\n");

    std::vector<float> dc(N, 1.0f);
    fft.forward(dc.data(), spec.data());
    std::printf("dc     : ");
    for (int k = 0; k < 5; ++k) std::printf("(% .3f,% .3f) ", spec[k].real(), spec[k].imag());
    std::printf("  bin128=(% .3f,% .3f)\n", spec[128].real(), spec[128].imag());

    // single sine at bin 10 -> spectrum should be N/2 at bin 10 with -j phase
    std::vector<float> s(N);
    for (size_t i = 0; i < N; ++i) s[i] = std::cos(2 * M_PI * 10 * i / N);
    fft.forward(s.data(), spec.data());
    std::printf("cos10  : bin10=(% .3f,% .3f) bin246=(% .3f,% .3f)\n", spec[10].real(), spec[10].imag(),
                spec[N - 10].real(), spec[N - 10].imag());
    return 0;
}
