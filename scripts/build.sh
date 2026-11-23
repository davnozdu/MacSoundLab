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
# int8 StemgenRT (tools/quantize_model.py): 3 MB, so it ships instead of being downloaded.
cp assets/models/vocals_int8.onnx assets/models/LICENSE-StemgenRT.md "${APP}/Contents/Resources/"
cp core/third_party/libonnxruntime.dylib "${APP}/Contents/Frameworks/"

# Signing. A stable certificate keeps the designated requirement the same
# across builds ("identifier ... and certificate root = H..."), so macOS keeps
# the "System Audio Recording" permission after an update, and the updater can
# refuse a bundle signed by anyone else. Ad-hoc signing (a new cdhash every
# build) is only the fallback for builds without the certificate.
#   local: ~/.config/macsoundlab-signing/{signing.keychain-db,signing.env}
#   CI:    SIGN_KEYCHAIN / SIGN_KEYCHAIN_PASSWORD set by the workflow
LOCAL_SIGNING="${HOME}/.config/macsoundlab-signing"
SIGN_KEYCHAIN="${SIGN_KEYCHAIN:-}"
if [ -z "${SIGN_KEYCHAIN}" ] && [ -f "${LOCAL_SIGNING}/signing.env" ]; then
    # shellcheck disable=SC1091
    . "${LOCAL_SIGNING}/signing.env"
    SIGN_KEYCHAIN="${LOCAL_SIGNING}/signing.keychain-db"
fi
SIGN_ARGS=(--sign -)
if [ -n "${SIGN_KEYCHAIN}" ] && [ -f "${SIGN_KEYCHAIN}" ]; then
    security unlock-keychain -p "${SIGN_KEYCHAIN_PASSWORD}" "${SIGN_KEYCHAIN}"
    SIGN_IDENTITY="$(security find-identity -p codesigning "${SIGN_KEYCHAIN}" \
        | awk '/"Mac Sound Lab Signing"/ { print $2; exit }')"
    if [ -n "${SIGN_IDENTITY}" ]; then
        SIGN_ARGS=(--sign "${SIGN_IDENTITY}" --keychain "${SIGN_KEYCHAIN}")
    fi
fi
if [ "${SIGN_ARGS[1]}" = "-" ]; then
    if [ "${REQUIRE_STABLE_SIGNATURE:-0}" = "1" ]; then
        echo "error: no signing certificate, refusing to build a release ad-hoc" >&2
        exit 1
    fi
    echo "==> Signing (ad-hoc: no certificate, permissions will reset on update)"
else
    echo "==> Signing (Mac Sound Lab Signing, ${SIGN_IDENTITY})"
fi

# --timestamp=none: a timestamp newer than the clock (fresh CI checkout, clock
# skew) makes codesign fail with "timestamps differ by N seconds". The dylib
# ships with a signature from the machine that built it; drop it first.
codesign --remove-signature "${APP}/Contents/Frameworks/libonnxruntime.dylib" 2>/dev/null || true
codesign --force "${SIGN_ARGS[@]}" --timestamp=none "${APP}/Contents/Frameworks/libonnxruntime.dylib"
codesign --force "${SIGN_ARGS[@]}" --timestamp=none --entitlements scripts/MacSoundLab.entitlements \
    --options runtime "${APP}/Contents/MacOS/soundlab-cli"
codesign --force "${SIGN_ARGS[@]}" --timestamp=none --entitlements scripts/MacSoundLab.entitlements \
    --options runtime "${APP}"
codesign --verify --deep --strict "${APP}"
codesign -d -r- "${APP}" 2>&1 | sed -n 's/^#* *designated => /    designated: /p'

echo "==> Done: ${APP}"
echo "    open ${APP}"
