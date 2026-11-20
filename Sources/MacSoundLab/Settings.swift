import Combine
import CoreAudio
import Foundation
import ServiceManagement

enum Profile: Int, CaseIterable, Identifiable {
    case game = 1
    case movie = 2
    case music = 3
    case speech = 4

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .game: return "Игра"
        case .movie: return "Кино"
        case .music: return "Музыка"
        case .speech: return "Подкаст"
        }
    }

    var hint: String {
        switch self {
        case .game: return "Точная локализация, сухая комната, динамика без компрессии"
        case .movie: return "Диалоги +2 дБ, большая комната, нейросеть и компрессор"
        case .music: return "Широкая сцена, без акцента на центр"
        case .speech: return "Подкасты, YouTube, аудиокниги: голос по центру, без комнаты, ровная громкость"
        }
    }
}

final class SettingsStore: ObservableObject {
    private let defaults = UserDefaults.standard

    @Published var enabled: Bool { didSet { save() } }
    @Published var profile: Profile { didSet { save() } }
    @Published var neural: Bool { didSet { save() } }
    @Published var reverb: Bool { didSet { save() } }
    @Published var bypass: Bool { didSet { save() } }
    @Published var deviceUID: String? { didSet { save() } }
    @Published var kindOverrides: [String: Int] { didSet { save() } }
    @Published var launchAtLogin: Bool { didSet { save(); applyLoginItem() } }

    init() {
        enabled = defaults.bool(forKey: "enabled")
        profile = Profile(rawValue: defaults.integer(forKey: "profile")) ?? .movie
        neural = defaults.object(forKey: "neural") as? Bool ?? true
        reverb = defaults.object(forKey: "reverb") as? Bool ?? true
        bypass = defaults.bool(forKey: "bypass")
        deviceUID = defaults.string(forKey: "deviceUID")
        kindOverrides = defaults.dictionary(forKey: "kindOverrides") as? [String: Int] ?? [:]
        launchAtLogin = defaults.bool(forKey: "launchAtLogin")
    }

    private func save() {
        defaults.set(enabled, forKey: "enabled")
        defaults.set(profile.rawValue, forKey: "profile")
        defaults.set(neural, forKey: "neural")
        defaults.set(reverb, forKey: "reverb")
        defaults.set(bypass, forKey: "bypass")
        if let deviceUID { defaults.set(deviceUID, forKey: "deviceUID") } else { defaults.removeObject(forKey: "deviceUID") }
        defaults.set(kindOverrides, forKey: "kindOverrides")
        defaults.set(launchAtLogin, forKey: "launchAtLogin")
    }

    private func applyLoginItem() {
        do {
            if launchAtLogin {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
        } catch {
            // Sandboxed or unsigned builds cannot register; the toggle is best-effort.
        }
    }
}

/// Glue between the controls, the engine and the model store.
final class AppModel: ObservableObject {
    let engine = AudioEngine()
    let settings = SettingsStore()
    let models = ModelStore()

    @Published private(set) var devices: [AudioOutput] = []
    @Published var errorText: String?
    @Published private(set) var currentOutputName = "—"
    @Published private(set) var currentKind: OutputKind = .headphones

    private var bag = Set<AnyCancellable>()
    private var deviceTimer: Timer?
    private var defaultDeviceListener: AudioObjectPropertyListenerBlock?

    init() {
        refreshDevices()

        settings.$enabled.sink { [weak self] _ in self?.apply() }.store(in: &bag)
        settings.$profile.sink { [weak self] value in self?.engine.setProfile(value.rawValue) }.store(in: &bag)
        settings.$neural.sink { [weak self] value in self?.engine.setNeural(value) }.store(in: &bag)
        settings.$reverb.sink { [weak self] value in self?.engine.setReverb(value) }.store(in: &bag)
        settings.$bypass.sink { [weak self] value in self?.engine.setBypass(value) }.store(in: &bag)
        settings.$deviceUID.sink { [weak self] _ in self?.apply() }.store(in: &bag)
        settings.$kindOverrides.sink { [weak self] _ in self?.apply() }.store(in: &bag)

        engine.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &bag)

        watchDefaultDevice()
        deviceTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] _ in
            self?.refreshDevices()
        }

        if settings.enabled {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { [weak self] in self?.apply() }
        }
    }

    func refreshDevices() {
        let found = AudioDevices.outputs()
        if found != devices { devices = found }
        updateCurrent()
    }

    private func currentDevice() -> AudioOutput? {
        if let uid = settings.deviceUID, let match = devices.first(where: { $0.uid == uid }) {
            return match
        }
        guard let id = AudioDevices.defaultOutput() else { return nil }
        return devices.first { $0.id == id }
    }

    private func kind(for device: AudioOutput) -> OutputKind {
        if let raw = settings.kindOverrides[device.uid], let kind = OutputKind(rawValue: raw) {
            return kind
        }
        return device.kindGuess
    }

    private func updateCurrent() {
        guard let device = currentDevice() else {
            currentOutputName = "нет выхода"
            return
        }
        currentOutputName = device.name
        currentKind = kind(for: device)
    }

    func apply() {
        guard settings.enabled else {
            engine.stop()
            errorText = nil
            updateCurrent()
            return
        }
        guard let device = currentDevice() else {
            errorText = "Не найдено устройство вывода звука."
            engine.stop()
            return
        }
        guard let defaultID = AudioDevices.defaultOutput() else {
            errorText = "Система не сообщает устройство вывода."
            return
        }

        let output = devices.first { $0.id == defaultID } ?? device
        let wanted = kind(for: output)
        let neuralReady = settings.neural && models.modelInstalled
        currentOutputName = output.name
        currentKind = wanted
        if settings.neural && !models.modelInstalled {
            errorText = "Нейросеть включена, но модель не скачана — работает пассивная матрица."
        } else {
            errorText = nil
        }

        if engine.isRunning, engine.outputName == output.name {
            engine.setProfile(settings.profile.rawValue)
            engine.setDeviceKind(wanted)
            engine.setNeural(neuralReady)
            engine.setReverb(settings.reverb)
            engine.setBypass(settings.bypass)
            return
        }

        do {
            try engine.start(output: output,
                             kind: wanted,
                             profile: settings.profile.rawValue,
                             neural: neuralReady,
                             reverb: settings.reverb,
                             bypass: settings.bypass,
                             runtimeDir: models.runtimeDirectory,
                             vocalsModel: models.vocalsModelPath)
        } catch {
            engine.stop()
            errorText = error.localizedDescription
        }
    }

    private func watchDefaultDevice() {
        var address = AudioDevices.propertyAddress(kAudioHardwarePropertyDefaultOutputDevice)
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            DispatchQueue.main.async {
                self?.refreshDevices()
                if self?.settings.enabled == true { self?.apply() }
            }
        }
        defaultDeviceListener = listener
        AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, DispatchQueue.main, listener)
    }
}
