import Foundation
import CryptoKit

/// Owns the app's support folder and finds the separator model: the int8 model
/// that ships inside the bundle (3 MB, built by tools/quantize_model.py), or,
/// in a build without it, the fp32 model downloaded on first use.
final class ModelStore: ObservableObject {
    static let modelURL = URL(string: "https://media.githubusercontent.com/media/sweetspotsoundsystem/stemgen-rt/main/model/parallel/vocals.onnx")!
    static let modelSHA256 = "f29a3d8fa596ce4672dd4dc77ccf224f2538389940264aad46ab8168311c2148"

    @Published private(set) var isDownloading = false
    @Published private(set) var progress: Double = 0
    @Published private(set) var errorText: String?

    let supportDirectory: URL
    let modelsDirectory: URL

    init() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        supportDirectory = base.appendingPathComponent("Mac Sound Lab", isDirectory: true)
        modelsDirectory = supportDirectory.appendingPathComponent("models", isDirectory: true)
        try? FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        // The bundled int8 model replaces the 11 MB download of older versions.
        if bundledModelPath != nil, FileManager.default.fileExists(atPath: downloadedModelPath) {
            try? FileManager.default.removeItem(atPath: downloadedModelPath)
        }
    }

    /// int8 model inside the app bundle; next to the sources when run from a build tree.
    var bundledModelPath: String? {
        if let path = Bundle.main.path(forResource: "vocals_int8", ofType: "onnx") { return path }
        let tree = FileManager.default.currentDirectoryPath + "/assets/models/vocals_int8.onnx"
        return FileManager.default.fileExists(atPath: tree) ? tree : nil
    }

    var modelIsBundled: Bool { bundledModelPath != nil }

    private var downloadedModelPath: String { modelsDirectory.appendingPathComponent("vocals.onnx").path }

    var vocalsModelPath: String { bundledModelPath ?? downloadedModelPath }

    var modelInstalled: Bool { FileManager.default.fileExists(atPath: vocalsModelPath) }

    /// Directory that holds libonnxruntime.dylib: the bundle's Frameworks folder,
    /// or the support folder when the app runs from a Swift build tree.
    var runtimeDirectory: String {
        if let frameworks = Bundle.main.privateFrameworksPath,
           FileManager.default.fileExists(atPath: frameworks + "/libonnxruntime.dylib") {
            return frameworks
        }
        for candidate in [
            Bundle.main.bundleURL.appendingPathComponent("Contents/Frameworks").path,
            supportDirectory.path,
            FileManager.default.currentDirectoryPath + "/core/third_party",
        ] where FileManager.default.fileExists(atPath: candidate + "/libonnxruntime.dylib") {
            return candidate
        }
        return Bundle.main.privateFrameworksPath ?? supportDirectory.path
    }

    var runtimeInstalled: Bool {
        FileManager.default.fileExists(atPath: runtimeDirectory + "/libonnxruntime.dylib")
    }

    func downloadModel() {
        guard !isDownloading else { return }
        isDownloading = true
        progress = 0
        errorText = nil
        Task { @MainActor in
            do {
                let (temporary, response) = try await URLSession.shared.download(from: Self.modelURL)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    throw NSError(domain: "MacSoundLab", code: 1,
                                  userInfo: [NSLocalizedDescriptionKey: "Сервер вернул \( (response as? HTTPURLResponse)?.statusCode ?? 0)"])
                }
                let digest = try Self.sha256(of: temporary)
                guard digest == Self.modelSHA256 else {
                    try? FileManager.default.removeItem(at: temporary)
                    throw NSError(domain: "MacSoundLab", code: 2,
                                  userInfo: [NSLocalizedDescriptionKey: "Контрольная сумма модели не совпала."])
                }
                let target = URL(fileURLWithPath: downloadedModelPath)
                try? FileManager.default.removeItem(at: target)
                try FileManager.default.moveItem(at: temporary, to: target)
                progress = 1
            } catch {
                errorText = error.localizedDescription
            }
            isDownloading = false
            objectWillChange.send()
        }
    }

    private static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while autoreleasepool(invoking: {
            let chunk = try? handle.read(upToCount: 1 << 20)
            guard let chunk, !chunk.isEmpty else { return false }
            hasher.update(data: chunk)
            return true
        }) {}
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
