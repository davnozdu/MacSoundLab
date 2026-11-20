import AppKit
import Foundation

/// Updates from GitHub Releases: check the latest tag, download the zip, verify
/// what is inside, swap the bundle and relaunch. No Sparkle, no appcast key —
/// the release itself is the source of truth.
final class Updater: ObservableObject {
    static let repository = "davnozdu/MacSoundLab"

    struct Release: Decodable {
        struct Asset: Decodable {
            let name: String
            let browserDownloadURL: URL
            let size: Int

            enum CodingKeys: String, CodingKey {
                case name
                case browserDownloadURL = "browser_download_url"
                case size
            }
        }

        let tagName: String
        let body: String?
        let htmlURL: URL
        let assets: [Asset]

        enum CodingKeys: String, CodingKey {
            case tagName = "tag_name"
            case body
            case htmlURL = "html_url"
            case assets
        }
    }

    @Published private(set) var latestVersion: String?
    @Published private(set) var latestNotes: String?
    @Published private(set) var isChecking = false
    @Published private(set) var isInstalling = false
    @Published private(set) var status: String?
    @Published private(set) var errorText: String?

    var currentVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
    }

    var updateAvailable: Bool {
        guard let latestVersion else { return false }
        return Self.isNewer(latestVersion, than: currentVersion)
    }

    // MARK: - checks

    func check(silent: Bool = false) {
        guard !isChecking else { return }
        isChecking = true
        if !silent { status = "Проверяю обновления…" }
        errorText = nil

        let request = Self.releaseRequest()

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                guard let self else { return }
                self.isChecking = false
                if let error {
                    if !silent { self.errorText = "Не удалось проверить обновления: \(error.localizedDescription)" }
                    self.status = nil
                    return
                }
                guard let http = response as? HTTPURLResponse, http.statusCode == 200, let data else {
                    if !silent {
                        self.errorText = "GitHub вернул \((response as? HTTPURLResponse)?.statusCode ?? 0)."
                    }
                    self.status = nil
                    return
                }
                do {
                    let release = try JSONDecoder().decode(Release.self, from: data)
                    self.latestVersion = Self.normalise(release.tagName)
                    self.latestNotes = release.body
                    self.status = self.updateAvailable
                        ? "Доступна версия \(self.latestVersion ?? "")"
                        : "Установлена последняя версия"
                    NSLog("MacSoundLab: update current=\(self.currentVersion) latest=\(self.latestVersion ?? "-") available=\(self.updateAvailable)")
                } catch {
                    if !silent { self.errorText = "Не разобрал ответ GitHub: \(error.localizedDescription)" }
                    self.status = nil
                }
            }
        }.resume()
    }

    // MARK: - install

    func install() {
        guard !isInstalling, let version = latestVersion, updateAvailable else { return }
        isInstalling = true
        errorText = nil
        status = "Скачиваю \(version)…"

        let request = Self.releaseRequest()

        URLSession.shared.dataTask(with: request) { [weak self] data, _, _ in
            guard let self, let data,
                  let release = try? JSONDecoder().decode(Release.self, from: data) else {
                DispatchQueue.main.async {
                    self?.isInstalling = false
                    self?.errorText = "Не удалось получить список файлов релиза."
                }
                return
            }
            guard let asset = release.assets.first(where: { $0.name.lowercased().hasSuffix(".zip") }) else {
                DispatchQueue.main.async {
                    self.isInstalling = false
                    self.errorText = "В релизе нет zip-архива приложения."
                }
                return
            }
            URLSession.shared.downloadTask(with: asset.browserDownloadURL) { location, _, error in
                DispatchQueue.main.async {
                    self.handleDownload(location: location, error: error, version: version)
                }
            }.resume()
        }.resume()
    }

    private func handleDownload(location: URL?, error: Error?, version: String) {
        NSLog("MacSoundLab: update download finished for \(version), error=\(error?.localizedDescription ?? "-")")
        guard let location, error == nil else {
            isInstalling = false
            errorText = "Скачивание не удалось: \(error?.localizedDescription ?? "неизвестная ошибка")"
            return
        }
        status = "Распаковываю…"
        do {
            let work = FileManager.default.temporaryDirectory
                .appendingPathComponent("MacSoundLabUpdate-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
            let zip = work.appendingPathComponent("update.zip")
            try FileManager.default.moveItem(at: location, to: zip)
            try run("/usr/bin/ditto", ["-x", "-k", zip.path, work.path])

            let contents = try FileManager.default.contentsOfDirectory(at: work, includingPropertiesForKeys: nil)
            guard let app = contents.first(where: { $0.pathExtension == "app" }) else {
                throw UpdaterError.noAppInArchive
            }
            let identifier = Bundle(url: app)?.bundleIdentifier
            let bundledVersion = Bundle(url: app)?
                .object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
            guard identifier == Bundle.main.bundleIdentifier else {
                throw UpdaterError.wrongBundle(identifier ?? "nil")
            }
            guard bundledVersion == version else {
                throw UpdaterError.versionMismatch(bundledVersion ?? "nil", version)
            }

            let target = Bundle.main.bundleURL
            status = "Устанавливаю…"
            try scheduleSwap(target: target, replacement: app)
            NSLog("MacSoundLab: update swap scheduled, replacing \(target.path)")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                NSApp.terminate(nil)
            }
        } catch {
            isInstalling = false
            errorText = error.localizedDescription
            status = nil
        }
    }

    /// The running bundle cannot replace itself, so a tiny shell script waits for
    /// the app to exit, swaps the bundles and opens the new one.
    private func scheduleSwap(target: URL, replacement: URL) throws {
        let scriptURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("MacSoundLabUpdate-\(UUID().uuidString).sh")
        let script = """
        #!/bin/bash
        sleep 1.5
        APP="\(target.path)"
        NEW="\(replacement.path)"
        OLD="$APP.old"
        # Never touch the installed app unless the replacement is a real,
        # signed bundle: a truncated zip would otherwise be copied over a
        # working install and ditto can still exit 0 on garbage.
        [ -f "$NEW/Contents/Info.plist" ] || exit 2
        [ -x "$NEW/Contents/MacOS/\(Bundle.main.executableURL?.lastPathComponent ?? "MacSoundLab")" ] || exit 2
        /usr/bin/codesign --verify --deep --strict "$NEW" >/dev/null 2>&1 || exit 2
        rm -rf "$OLD"
        mv "$APP" "$OLD" || exit 1
        if /usr/bin/ditto "$NEW" "$APP" && /usr/bin/codesign --verify --deep --strict "$APP" >/dev/null 2>&1; then
            rm -rf "$OLD"
            /usr/bin/xattr -dr com.apple.quarantine "$APP" 2>/dev/null
            /usr/bin/open "$APP"
        else
            rm -rf "$APP"
            mv "$OLD" "$APP"
            /usr/bin/open "$APP"
        fi
        """
        try script.write(to: scriptURL, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: scriptURL.path)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [scriptURL.path]
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
    }

    private func run(_ executable: String, _ arguments: [String]) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let pipe = Pipe()
        process.standardError = pipe
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let message = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            throw UpdaterError.toolFailed(executable, message.trimmingCharacters(in: .whitespacesAndNewlines))
        }
    }

    // MARK: - networking

    /// GitHub serves the release list with `Cache-Control: max-age=60`; without
    /// bypassing the cache a release published a minute ago looks like "no
    /// update" and the app keeps saying it is current.
    private static func releaseRequest() -> URLRequest {
        var request = URLRequest(url: URL(string: "https://api.github.com/repos/\(repository)/releases/latest")!,
                                 cachePolicy: .reloadIgnoringLocalCacheData,
                                 timeoutInterval: 20)
        request.setValue("MacSoundLab", forHTTPHeaderField: "User-Agent")
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        return request
    }

    // MARK: - versions

    private static func normalise(_ tag: String) -> String {
        tag.hasPrefix("v") ? String(tag.dropFirst()) : tag
    }

    private static func isNewer(_ candidate: String, than current: String) -> Bool {
        let left = normalise(candidate).split(separator: ".").map { Int($0) ?? 0 }
        let right = normalise(current).split(separator: ".").map { Int($0) ?? 0 }
        for index in 0..<max(left.count, right.count) {
            let a = index < left.count ? left[index] : 0
            let b = index < right.count ? right[index] : 0
            if a != b { return a > b }
        }
        return false
    }
}

enum UpdaterError: LocalizedError {
    case noAppInArchive
    case wrongBundle(String)
    case versionMismatch(String, String)
    case toolFailed(String, String)

    var errorDescription: String? {
        switch self {
        case .noAppInArchive:
            return "В архиве нет приложения .app."
        case .wrongBundle(let identifier):
            return "В архиве чужое приложение (\(identifier))."
        case .versionMismatch(let found, let expected):
            return "В архиве версия \(found), ждали \(expected)."
        case .toolFailed(let tool, let message):
            return "\(tool) завершился с ошибкой: \(message)"
        }
    }
}
