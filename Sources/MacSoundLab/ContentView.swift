import AppKit
import SwiftUI

@main
struct MacSoundLabApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup("Mac Sound Lab") {
            ContentView()
                .environmentObject(model)
                .environmentObject(model.settings)
        }
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Звук") {
                Toggle("Обработка", isOn: Binding(
                    get: { model.settings.enabled },
                    set: { model.settings.enabled = $0 }
                ))
                .keyboardShortcut("e")
                Toggle("Сравнить с оригиналом", isOn: Binding(
                    get: { model.settings.bypass },
                    set: { model.settings.bypass = $0 }
                ))
                .keyboardShortcut("b")
            }
        }

        MenuBarExtra {
            MenuBarContent()
                .environmentObject(model)
                .environmentObject(model.settings)
        } label: {
            Image(systemName: model.settings.enabled ? "waveform.circle.fill" : "waveform.circle")
        }
    }
}

struct MenuBarContent: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var settings: SettingsStore

    var body: some View {
        Toggle("Включить обработку", isOn: $settings.enabled)
        Divider()
        ForEach(Profile.allCases) { profile in
            Button {
                settings.profile = profile
            } label: {
                if settings.profile == profile {
                    Label("\(profile.title) — \(profile.hint)", systemImage: "checkmark")
                } else {
                    Text("\(profile.title) — \(profile.hint)")
                }
            }
            .disabled(!settings.enabled)
        }
        Divider()
        Toggle("Нейросеть", isOn: $settings.neural)
            .disabled(!settings.enabled || !model.models.modelInstalled)
        Toggle("Реверберация (комната)", isOn: $settings.reverb)
            .disabled(!settings.enabled)
        Toggle("Сравнить с оригиналом", isOn: $settings.bypass)
            .disabled(!settings.enabled)
        Divider()
        if let version = model.updater.latestVersion, model.updater.updateAvailable {
            Button("Обновить до \(version)") { model.updater.install() }
            Divider()
        }
        if model.engine.isRunning {
            Text("Выход: \(model.currentOutputName)")
            Text("Задержка: \(Int(model.engine.estimatedLatencyMs)) мс")
        } else {
            Text("Выключено")
        }
        Divider()
        Button("Открыть Mac Sound Lab") {
            NSApp.activate(ignoringOtherApps: true)
            for window in NSApp.windows where window.canBecomeMain {
                window.makeKeyAndOrderFront(nil)
            }
        }
        Button("Выйти") { NSApp.terminate(nil) }
    }
}

struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var settings: SettingsStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                profilePicker
                outputPicker
                switches
                statistics
                modelSection
                updatesSection
                if let message = model.engine.errorMessage ?? model.errorText ?? model.models.errorText {
                    banner(message)
                }
                footer
            }
            .padding(20)
        }
        .frame(width: 470, height: 720)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 34))
                .foregroundStyle(settings.enabled ? Color.accentColor : Color.secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text("Mac Sound Lab").font(.title2.weight(.semibold))
                Text(settings.enabled
                     ? (model.engine.isRunning ? "Обрабатывается: \(model.currentOutputName)" : "Запуск…")
                     : "Выключено — звук идёт напрямую")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Toggle("", isOn: $settings.enabled)
                .toggleStyle(.switch)
                .labelsHidden()
                .controlSize(.large)
        }
    }

    private var profilePicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Профиль").font(.headline)
            Picker("", selection: $settings.profile) {
                ForEach(Profile.allCases) { profile in
                    Text(profile.title).tag(profile)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .disabled(!settings.enabled)
            Text(settings.profile.hint)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var outputPicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Выход").font(.headline)
            Picker("", selection: Binding(
                get: { settings.deviceUID ?? "" },
                set: { settings.deviceUID = $0.isEmpty ? nil : $0 }
            )) {
                Text("Системный по умолчанию").tag("")
                ForEach(model.devices) { device in
                    Text(device.name).tag(device.uid)
                }
            }
            .labelsHidden()
            .disabled(!settings.enabled)
            HStack {
                Text("Тип устройства").font(.callout).foregroundStyle(.secondary)
                Picker("", selection: Binding(
                    get: { settings.kindOverrides[model.deviceUIDSnapshot] ?? -1 },
                    set: { value in
                        if value < 0 {
                            settings.kindOverrides.removeValue(forKey: model.deviceUIDSnapshot)
                        } else {
                            settings.kindOverrides[model.deviceUIDSnapshot] = value
                        }
                    }
                )) {
                    Text("Авто").tag(-1)
                    ForEach(OutputKind.allCases) { kind in
                        Text(kind.title).tag(kind.rawValue)
                    }
                }
                .labelsHidden()
                .frame(width: 130)
                Spacer()
                Text("сейчас: \(model.currentKind.title)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .disabled(!settings.enabled)
        }
    }

    private var switches: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Обработка").font(.headline)
            Toggle("Нейросеть для стерео (StemgenRT)", isOn: $settings.neural)
                .disabled(!settings.enabled || !model.models.modelInstalled)
            VStack(alignment: .leading, spacing: 2) {
                Toggle("Реверберация (комната)", isOn: $settings.reverb)
                    .disabled(!settings.enabled)
                Text(settings.reverb
                     ? "Включена: добавлен синтетический хвост комнаты — объём и «воздух»."
                     : "Выключена: сухой звук без комнаты, только прямые HRTF-отражения.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if settings.profile == .speech {
                Text("В профиле «Подкаст» комната отключена всегда: голос должен быть сухим.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            Toggle("Сравнить с оригиналом", isOn: $settings.bypass)
                .disabled(!settings.enabled)
        }
        .toggleStyle(.switch)
        .controlSize(.small)
    }

    private var statistics: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Состояние").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 4) {
                GridRow {
                    Text("Источник").foregroundStyle(.secondary)
                    Text(model.engine.sourceIsSurround ? "5.1 / 7.1" : "стерео")
                }
                GridRow {
                    Text("Задержка").foregroundStyle(.secondary)
                    Text(model.engine.isRunning ? "\(Int(model.engine.estimatedLatencyMs)) мс" : "—")
                }
                GridRow {
                    Text("DSP на блок").foregroundStyle(.secondary)
                    Text(model.engine.isRunning ? String(format: "%.2f мс", model.engine.dspMs) : "—")
                }
                GridRow {
                    Text("Нейросеть").foregroundStyle(.secondary)
                    Text(neuralLine)
                }
                GridRow {
                    Text("Приоритет").foregroundStyle(.secondary)
                    Text(model.engine.neuralReady && model.engine.accelActive ? "GPU / Neural Engine" : "CPU")
                }
            }
            .font(.callout)
            meters
        }
    }

    private var neuralLine: String {
        if !settings.neural { return "выключена" }
        if !model.models.modelInstalled { return "модель не скачана" }
        if !model.engine.neuralReady { return "—" }
        return String(format: "%.2f мс/блок, сбоев %llu", model.engine.modelMs, model.engine.modelDrops)
    }

    private var meters: some View {
        HStack(spacing: 10) {
            meter(model.engine.levelLeft)
            meter(model.engine.levelRight)
        }
        .frame(height: 8)
        .padding(.top, 4)
    }

    private func meter(_ level: Float) -> some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 3).fill(Color.secondary.opacity(0.18))
                RoundedRectangle(cornerRadius: 3)
                    .fill(level > 0.98 ? Color.red : Color.accentColor)
                    .frame(width: geometry.size.width * CGFloat(min(1, level)))
            }
        }
    }

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Компоненты").font(.headline)
            HStack(spacing: 8) {
                Circle()
                    .fill(model.models.runtimeInstalled ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)
                Text(model.models.runtimeInstalled ? "ONNX Runtime на месте" : "ONNX Runtime не найден в бандле")
                    .font(.callout)
            }
            HStack(spacing: 8) {
                Circle()
                    .fill(model.models.modelInstalled ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)
                Text(model.models.modelInstalled ? "Модель StemgenRT на месте" : "Модель StemgenRT не скачана")
                    .font(.callout)
                Spacer()
                if !model.models.modelInstalled {
                    if model.models.isDownloading {
                        ProgressView(value: model.models.progress)
                            .frame(width: 90)
                    } else {
                        Button("Скачать") { model.models.downloadModel() }
                            .controlSize(.small)
                    }
                }
            }
        }
    }

    private var updatesSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Версия").font(.headline)
            HStack(spacing: 8) {
                Text("Установлена \(model.updater.currentVersion)")
                    .font(.callout)
                Spacer()
                if model.updater.isInstalling {
                    ProgressView().controlSize(.small)
                } else if model.updater.isChecking {
                    ProgressView().controlSize(.small)
                } else {
                    Button("Проверить обновления") { model.updater.check() }
                        .controlSize(.small)
                }
            }
            if model.updater.updateAvailable, let version = model.updater.latestVersion {
                HStack(spacing: 8) {
                    Image(systemName: "arrow.down.circle.fill").foregroundStyle(Color.accentColor)
                    Text("Вышла версия \(version)")
                        .font(.callout)
                    Spacer()
                    Button("Обновить") { model.updater.install() }
                        .controlSize(.small)
                        .disabled(model.updater.isInstalling)
                }
            }
            if let status = model.updater.status {
                Text(status).font(.caption).foregroundStyle(.secondary)
            }
            if let error = model.updater.errorText {
                Text(error).font(.caption).foregroundStyle(.orange)
            }
            Toggle("Обновляться автоматически", isOn: $settings.autoUpdate)
                .toggleStyle(.checkbox)
                .controlSize(.small)
            Text("Обновления берутся из GitHub Releases: приложение скачивает архив, "
                 + "проверяет подпись бандла и перезапускается.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func banner(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            Text(message).font(.callout).fixedSize(horizontal: false, vertical: true)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
    }

    private var footer: some View {
        HStack {
            Toggle("Запускать при входе", isOn: $settings.launchAtLogin)
                .toggleStyle(.checkbox)
                .controlSize(.small)
            Spacer()
            Text("DSP: Ally / Android, порт для macOS")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }
}

extension AppModel {
    /// UID of the device the "type" override applies to.
    var deviceUIDSnapshot: String {
        if let uid = settings.deviceUID, !uid.isEmpty { return uid }
        if let id = AudioDevices.defaultOutput(), let match = devices.first(where: { $0.id == id }) {
            return match.uid
        }
        return ""
    }
}
