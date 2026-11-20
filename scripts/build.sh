#!/usr/bin/env bash
# Builds MacSoundLab.app (arm64) and ad-hoc signs it.
# The ONNX Runtime library is fetched from NuGet once: the GitHub release
# assets of microsoft/onnxruntime have been unavailable since 17 Nov 2026.
set -euo pipefail
cd "$(dirname "$0")/.."

ORT_VERSION=1.32.0
ORT_NUGET="https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/${ORT_VERSION}"
APP="build/MacSoundLab.app"

mkdir -p core/third_party/include build

if [ ! -f core/third_party/libonnxruntime.dylib ] || [ ! -f core/third_party/include/onnxruntime_c_api.h ]; then
    echo "==> ONNX Runtime ${ORT_VERSION} (NuGet)"
    tmp="$(mktemp -d)"
    curl -fL --retry 3 -o "${tmp}/ort.nupkg" "${ORT_NUGET}"
    unzip -o -q "${tmp}/ort.nupkg" "runtimes/osx-arm64/native/libonnxruntime.dylib" "build/native/include/*" -d "${tmp}"
    cp "${tmp}/runtimes/osx-arm64/native/libonnxruntime.dylib" core/third_party/
    cp "${tmp}"/build/native/include/*.h core/third_party/include/
    rm -rf "${tmp}"
fi

echo "==> Building (arm64)"
# On Apple Silicon this is a plain build; on an Intel CI runner it is a
# cross-build into arm64, which the same Xcode SDK supports.
if [ "$(uname -m)" = "arm64" ]; then
    TARGET=(--arch arm64)
else
    TARGET=(--triple arm64-apple-macosx14.2)
fi
swift build -c release --product MacSoundLab "${TARGET[@]}"
swift build -c release --product soundlab-cli "${TARGET[@]}"

echo "==> Bundling"
rm -rf "${APP}"
mkdir -p "${APP}/Contents/MacOS" "${APP}/Contents/Frameworks" "${APP}/Contents/Resources"
cp .build/release/MacSoundLab "${APP}/Contents/MacOS/"
cp .build/release/soundlab-cli "${APP}/Contents/MacOS/"
cp scripts/Info.plist "${APP}/Contents/Info.plist"
cp assets/AppIcon.icns "${APP}/Contents/Resources/"
cp core/third_party/libonnxruntime.dylib "${APP}/Contents/Frameworks/"

echo "==> Signing (ad-hoc)"
# --timestamp=none: without it a re-signed binary whose embedded signature
# carries a timestamp newer than the clock (fresh CI checkout, clock skew)
# makes codesign fail with "timestamps differ by N seconds".
codesign --force --sign - --timestamp=none "${APP}/Contents/Frameworks/libonnxruntime.dylib"
codesign --force --sign - --timestamp=none --entitlements scripts/MacSoundLab.entitlements \
    --options runtime "${APP}/Contents/MacOS/soundlab-cli"
codesign --force --sign - --timestamp=none --entitlements scripts/MacSoundLab.entitlements \
    --options runtime "${APP}"

echo "==> Done: ${APP}"
echo "    open ${APP}"
