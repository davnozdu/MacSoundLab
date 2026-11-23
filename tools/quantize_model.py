#!/usr/bin/env python3
"""Builds assets/models/vocals_int8.onnx from the StemgenRT vocals model.

Dynamic int8 (weights and activations) on MatMul/Gemm, per-tensor: the 3-D
batched MatMul weights reject per-channel zero points in ONNX Runtime.
Measured against the fp32 model on this project's checks:
  - 0.337 vs 0.471 ms per 128-frame hop (one thread), 3.0 vs 11.0 MB;
  - separation unchanged within 0.1 dB (tools/spatial_check.cpp, part D);
  - int8 vs fp32 voice stem: 28-33 dB apart, no drift over time.
Needs: pip install onnx onnxruntime==1.32.0
Usage: tools/quantize_model.py models/vocals.onnx assets/models/vocals_int8.onnx
"""

import hashlib
import sys

from onnxruntime.quantization import QuantType, quantize_dynamic

FP32_SHA256 = "f29a3d8fa596ce4672dd4dc77ccf224f2538389940264aad46ab8168311c2148"


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    src, dst = sys.argv[1], sys.argv[2]
    digest = sha256(src)
    if digest != FP32_SHA256:
        raise SystemExit(f"unexpected source model {digest}")
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=False,
                     op_types_to_quantize=["MatMul", "Gemm"])
    print(f"{dst}: sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
