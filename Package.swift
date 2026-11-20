// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MacSoundLab",
    platforms: [.macOS("14.2")],
    products: [
        .executable(name: "MacSoundLab", targets: ["MacSoundLab"]),
        .executable(name: "soundlab-cli", targets: ["SoundLabCLI"]),
    ],
    targets: [
        // The Ally / Android DSP, unchanged apart from host paths.
        .target(
            name: "CSoundLab",
            path: "core",
            exclude: ["third_party/libonnxruntime.dylib"],
            sources: ["spatial.cpp", "vocals.cpp", "soundlab_c.cpp"],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("third_party/include"),
                .define("ALLY_ENABLE_NPU"),
                // The convolver and the FFT are reassociation-bound: with strict
                // FP semantics clang keeps them scalar. Measured on the DSP
                // bench: 0.052 -> 0.034 ms per block. Output stays within
                // -100 dB of the strict build (checked by rendering both).
                .unsafeFlags(["-O3", "-ffast-math"], .when(configuration: .release)),
            ]
        ),
        .executableTarget(
            name: "MacSoundLab",
            dependencies: ["CSoundLab"],
            path: "Sources/MacSoundLab"
        ),
        .executableTarget(
            name: "SoundLabCLI",
            dependencies: ["CSoundLab"],
            path: "Sources/SoundLabCLI"
        ),
    ],
    cxxLanguageStandard: .cxx17
)
