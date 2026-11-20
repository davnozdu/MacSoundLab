#!/usr/bin/env python3
"""Compare the C++ core (via soundlab-cli) with the Python reference renderer.

The reference is the Decky plugin's spatial.py, run at 48 kHz with the room
and the neural separator disabled, so both paths are deterministic.
"""

import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, "/Users/davnozdu/Downloads/AllySoundNPU/decky-ally-sound")
import spatial  # noqa: E402

spatial.RATE = 48000


def read_wav(path):
    import scipy.io.wavfile as wavfile

    sample_rate, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim == 1:
        data = np.stack((data, data), axis=1)
    return data, sample_rate


def main():
    source = sys.argv[1]
    rendered = sys.argv[2]
    source_audio, source_rate = read_wav(source)
    rendered_audio, rendered_rate = read_wav(rendered)
    if abs(source_rate - 48000) > 1 or abs(rendered_rate - 48000) > 1:
        raise SystemExit(f"expected 48 kHz, got {source_rate} / {rendered_rate}")

    renderer = spatial.Renderer("/tmp/ku100_48000.sofa", make_vocals=None,
                                profile="movie", neural=False, reverb=False)
    renderer.set_device("headphones")

    frames = min(len(source_audio), len(rendered_audio))
    blocks = np.zeros((8, frames), dtype=np.float32)
    blocks[0:2] = source_audio[:frames, :2].T
    hopped = np.zeros((8, frames), dtype=np.float32)
    reference = np.zeros((2, frames), dtype=np.float32)
    for start in range(0, frames, spatial.HOP):
        end = min(start + spatial.HOP, frames)
        out = renderer.process(blocks[:, start:end])
        reference[:, start:end] = out[:, : end - start]

    # The core reports one kHop of priming latency, so align by the best lag
    # instead of assuming an exact offset.
    a = reference[:, :]
    b = rendered_audio[:frames, :2].T
    mono_a = a.mean(axis=0)
    mono_b = b.mean(axis=0)
    windows = 1 << 19
    spectrum_a = np.fft.rfft(mono_a, n=windows)
    spectrum_b = np.fft.rfft(mono_b, n=windows)
    correlation = np.fft.irfft(spectrum_a * np.conj(spectrum_b), n=windows)
    search = 8192
    lags = np.concatenate((correlation[-search:], correlation[:search]))
    lag = int(np.argmax(np.abs(lags))) - search
    print(f"best lag: {lag} samples ({lag / spatial.RATE * 1000:.2f} ms)")
    if lag > 0:
        a = a[:, lag:]
    elif lag < 0:
        b = b[:, -lag:]
    length = min(a.shape[1], b.shape[1])
    a = a[:, :length]
    b = b[:, :length]

    def db(value):
        return 20 * np.log10(max(float(value), 1e-12))

    rms_ref = np.sqrt(np.mean(a ** 2))
    rms_cpp = np.sqrt(np.mean(b ** 2))
    diff = a - b
    print(f"frames: {a.shape[1]}")
    print(f"RMS ref {db(rms_ref):.2f} dBFS | C++ {db(rms_cpp):.2f} dBFS | delta {db(rms_cpp) - db(rms_ref):+.3f} dB")
    print(f"max |diff| {np.max(np.abs(diff)):.5f} | RMS diff {db(np.sqrt(np.mean(diff ** 2))):.2f} dB")
    print(f"correlation {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.6f}")


if __name__ == "__main__":
    main()
