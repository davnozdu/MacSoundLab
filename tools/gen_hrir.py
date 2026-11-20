#!/usr/bin/env python3
"""Download SADIE II KU100 48kHz SOFA and emit dsp/hrir_ku100.hpp."""

import hashlib
import os
import sys
import urllib.request
import h5py
import numpy as np

URL = "https://zenodo.org/records/12542676/files/SADIE_KU100_DFC_256_order_fir_48000.sofa"
SHA256 = "6822b4318ce80ab6b8421a4b55b1edcedde1b276426aa5416526a57cccbe7e58"
SPEAKERS = (30, 330, 0, 145, 215, 100, 260)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "dsp/hrir_ku100.hpp"
    sofa_path = "/tmp/ku100_48000.sofa"
    if not os.path.exists(sofa_path):
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, sofa_path)
    digest = hashlib.sha256(open(sofa_path, "rb").read()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"SHA256 mismatch: {digest} != {SHA256}")

    with h5py.File(sofa_path, "r") as f:
        positions = f["SourcePosition"][:]
        ir = f["Data.IR"][:]
        chosen = []
        for az in SPEAKERS:
            dist = np.abs((positions[:, 0] - az + 180) % 360 - 180) + 2 * np.abs(positions[:, 1])
            chosen.append(int(np.argmin(dist)))
        responses = np.stack([ir[idx] for idx in chosen]).astype(np.float32)

    S, E, L = responses.shape
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as out:
        out.write("#pragma once\n\nnamespace ally {\n\n")
        out.write(f"constexpr int kHrirSpeakers = {S};\n")
        out.write(f"constexpr int kHrirEars = {E};\n")
        out.write(f"constexpr int kHrirLength = {L};\n\n")
        out.write(f"constexpr float kHrirKu100[{S}][{E}][{L}] = {{\n")
        for s in range(S):
            out.write("    {\n")
            for e in range(E):
                vals = ", ".join(f"{float(v):.8e}f" for v in responses[s, e])
                out.write(f"        {{ {vals} }},\n")
            out.write("    },\n")
        out.write("};\n\n} // namespace ally\n")
    print(f"Generated {out_path} ({S}x{E}x{L})")


if __name__ == "__main__":
    main()
