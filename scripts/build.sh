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
swift build -c release --product MacSoundLab --arch arm64
swift build -c release --product soundlab-cli --arch arm64

echo "==> Bundling"
rm -rf "${APP}"
mkdir -p "${APP}/Contents/MacOS" "${APP}/Contents/Frameworks" "${APP}/Contents/Resources"
cp .build/release/MacSoundLab "${APP}/Contents/MacOS/"
cp .build/release/soundlab-cli "${APP}/Contents/MacOS/"
cp scripts/Info.plist "${APP}/Contents/Info.plist"
cp assets/AppIcon.icns "${APP}/Contents/Resources/"
cp core/third_party/libonnxruntime.dylib "${APP}/Contents/Frameworks/"

echo "==> Signing (ad-hoc)"
codesign --force --sign - "${APP}/Contents/Frameworks/libonnxruntime.dylib"
codesign --force --sign - --entitlements scripts/MacSoundLab.entitlements \
    --options runtime "${APP}/Contents/MacOS/soundlab-cli"
codesign --force --sign - --entitlements scripts/MacSoundLab.entitlements \
    --options runtime "${APP}"

echo "==> Done: ${APP}"
echo "    open ${APP}"
