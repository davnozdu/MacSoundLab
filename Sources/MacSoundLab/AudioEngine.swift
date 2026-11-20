import AppKit
import AudioToolbox
import CoreAudio
import Foundation
import CSoundLab

/// Stereo float ring between the tap IOProc and the output unit, with a
/// slowly adapting resample ratio so the two clocks never drift apart.
final class AudioRing {
    private var samples: [Float]
    private let capacity: Int
    private var writeIndex: Int64 = 0
    private var readPosition: Double = 0
    private var lock = os_unfair_lock_s()

    init(capacity: Int = 1 << 15) {
        self.capacity = capacity
        samples = [Float](repeating: 0, count: capacity * 2)
    }

    private func sample(_ index: Int64, _ channel: Int) -> Float {
        let slot = Int(index % Int64(capacity))
        return samples[slot * 2 + channel]
    }

    private func store(_ index: Int64, _ left: Float, _ right: Float) {
        let slot = Int(index % Int64(capacity))
        samples[slot * 2] = left
        samples[slot * 2 + 1] = right
    }

    func write(_ input: UnsafePointer<Float>, frames: Int) {
        os_unfair_lock_lock(&lock)
        for i in 0..<frames {
            store(writeIndex, input[2 * i], input[2 * i + 1])
            writeIndex += 1
        }
        os_unfair_lock_unlock(&lock)
    }

    /// Seconds of audio sitting in the ring, from the reader's point of view.
    func fillFrames() -> Double {
        os_unfair_lock_lock(&lock)
        let fill = Double(writeIndex) - readPosition
        os_unfair_lock_unlock(&lock)
        return fill
    }

    /// Pulls `frames` frames with linear interpolation at `step`.
    /// Returns how many were produced; the caller silences the remainder.
    @discardableResult
    func read(into output: UnsafeMutablePointer<Float>, frames: Int, step: Double) -> Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        let available = Double(writeIndex) - readPosition
        if available <= 0 {
            readPosition = Double(writeIndex)
            return 0
        }
        let needed = Double(frames) * step
        if available < needed {
            return 0
        }
        for i in 0..<frames {
            let base = Int64(readPosition.rounded(.down))
            let fraction = Float(readPosition - Double(base))
            for channel in 0..<2 {
                let a = sample(base, channel)
                let b = sample(base + 1, channel)
                output[2 * i + channel] = a + (b - a) * fraction
            }
            readPosition += step
        }
        return frames
    }

    /// After an underrun or overflow: drop the backlog and keep a known headroom.
    func resync(targetFrames: Double) {
        os_unfair_lock_lock(&lock)
        readPosition = max(0, Double(writeIndex) - targetFrames)
        os_unfair_lock_unlock(&lock)
    }

    func reset() {
        os_unfair_lock_lock(&lock)
        writeIndex = 0
        readPosition = 0
        os_unfair_lock_unlock(&lock)
    }
}

/// Live statistics for the UI, written from the audio threads.
struct EngineStats {
    var dspMs: Double = 0
    var blocks: Int = 0
    var underruns: Int = 0
    var ringFillMs: Double = 0
    var levelLeft: Float = 0
    var levelRight: Float = 0
}

enum EngineError: LocalizedError {
    case tapUnavailable
    case aggregateFailed
    case noOutputUnit
    case rendererFailed

    var errorDescription: String? {
        switch self {
        case .tapUnavailable:
            return "Не удалось создать аудиотап. Разреши приложению «Запись системного звука» в Настройках → Конфиденциальность."
        case .aggregateFailed:
            return "Не удалось создать агрегатное устройство для тапа."
        case .noOutputUnit:
            return "Не удалось открыть выбранное устройство вывода."
        case .rendererFailed:
            return "Не удалось запустить DSP."
        }
    }
}

final class AudioEngine: ObservableObject {
    static let processingRate: Double = 48000

    @Published private(set) var isRunning = false
    @Published private(set) var errorMessage: String?
    @Published private(set) var outputName = ""
    @Published private(set) var tapRate: Double = 0
    @Published private(set) var deviceKind: OutputKind = .headphones
    @Published private(set) var sourceIsSurround = false
    @Published private(set) var neuralReady = false
    private var lastReverbLog: Bool?
    @Published private(set) var accelActive = false
    @Published private(set) var modelMs: Float = 0
    @Published private(set) var modelDrops: UInt64 = 0
    @Published private(set) var dspMs: Double = 0
    @Published private(set) var ringFillMs: Double = 0
    @Published private(set) var levelLeft: Float = 0
    @Published private(set) var levelRight: Float = 0

    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var inputProcID: AudioDeviceIOProcID?
    private var outputUnit: AudioUnit?
    private var renderer: OpaquePointer?
    private var outputDevice: AudioOutput?

    private let ring = AudioRing()
    private let ioQueue = DispatchQueue(label: "cz.davnozdu.macsoundlab.tap", qos: .userInteractive)
    private let statsLock = NSLock()
    private var stats = EngineStats()
    private var inBuffer = [Float]()          // tap thread only
    private var tapOutBuffer = [Float]()      // tap thread only
    private var renderBuffer = [Float]()      // output thread only
    private var savedBufferSize: UInt32?
    private var uiTimer: Timer?
    private var logTicks = 0
    private var loggedFirstBlock = false

    /// Device rate / tap rate, plus the slow drift correction.
    private var baseStep: Double = 1.0
    private var primed = false
    private let targetFillFrames: Double = 1024   // ~21 ms of headroom

    // MARK: - lifecycle

    func start(output: AudioOutput, kind: OutputKind, profile: Int, neural: Bool,
               reverb: Bool, bypass: Bool, runtimeDir: String, vocalsModel: String) throws {
        stop()
        errorMessage = nil

        guard let created = sl_create(Float(Self.processingRate), Int32(profile), Int32(kind.rawValue),
                                      neural ? 1 : 0, reverb ? 1 : 0) else {
            throw EngineError.rendererFailed
        }
        renderer = created
        runtimeDir.withCString { sl_set_runtime_dir(created, $0) }
        vocalsModel.withCString { vocals in
            "".withCString { speech in sl_set_models(created, vocals, speech) }
        }
        sl_set_bypass(created, bypass ? 1 : 0)

        do {
            try startTap()
            try startOutput(device: output)
        } catch {
            stop()
            throw error
        }

        outputDevice = output
        outputName = output.name
        deviceKind = kind
        isRunning = true
        startUITimer()
    }

    func stop() {
        uiTimer?.invalidate()
        uiTimer = nil
        if aggregateID != AudioObjectID(kAudioObjectUnknown) {
            if let proc = inputProcID {
                AudioDeviceStop(aggregateID, proc)
                AudioDeviceDestroyIOProcID(aggregateID, proc)
            }
            inputProcID = nil
            AudioHardwareDestroyAggregateDevice(aggregateID)
            aggregateID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != AudioObjectID(kAudioObjectUnknown) {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
        if let unit = outputUnit {
            AudioOutputUnitStop(unit)
            AudioUnitUninitialize(unit)
            AudioComponentInstanceDispose(unit)
            outputUnit = nil
        }
        if let device = outputDevice, let size = savedBufferSize {
            var address = AudioDevices.propertyAddress(kAudioDevicePropertyBufferFrameSize)
            var value = size
            AudioObjectSetPropertyData(device.id, &address, 0, nil, UInt32(MemoryLayout<UInt32>.size), &value)
        }
        savedBufferSize = nil
        if let created = renderer {
            sl_destroy(created)
            renderer = nil
        }
        ring.reset()
        primed = false
        loggedFirstBlock = false
        outputDevice = nil
        isRunning = false
        stats = EngineStats()
        levelLeft = 0
        levelRight = 0
    }

    // MARK: - live parameters

    func setProfile(_ profile: Int) {
        if let renderer { sl_set_profile(renderer, Int32(profile)) }
    }

    func setDeviceKind(_ kind: OutputKind) {
        deviceKind = kind
        if let renderer { sl_set_device(renderer, Int32(kind.rawValue)) }
    }

    func setNeural(_ on: Bool) {
        if let renderer { sl_set_neural(renderer, on ? 1 : 0) }
    }

    func setReverb(_ on: Bool) {
        if let renderer {
            sl_set_reverb(renderer, on ? 1 : 0)
            // Only on a real change: apply() sets the same values on start and
            // used to print this four times in a row.
            if lastReverbLog != on {
                lastReverbLog = on
                NSLog("MacSoundLab: reverb=\(on ? 1 : 0)")
            }
        }
    }

    func setBypass(_ on: Bool) {
        if let renderer { sl_set_bypass(renderer, on ? 1 : 0) }
    }

    // MARK: - tap

    private func startTap() throws {
        let ownProcess = AudioDevices.ownProcessObject()
        let excluded: [AudioObjectID] = ownProcess.map { [$0] } ?? []
        let description = CATapDescription(stereoGlobalTapButExcludeProcesses: excluded)
        description.name = "Mac Sound Lab"
        description.uuid = UUID()
        description.isPrivate = true
        description.muteBehavior = .muted   // we play the processed mix instead

        var tap = AudioObjectID(kAudioObjectUnknown)
        let created = AudioHardwareCreateProcessTap(description, &tap)
        guard created == noErr, tap != AudioObjectID(kAudioObjectUnknown) else {
            NSLog("MacSoundLab: AudioHardwareCreateProcessTap failed: %d", created)
            throw EngineError.tapUnavailable
        }
        tapID = tap
        NSLog("MacSoundLab: tap created id=%u", tap)

        var address = AudioDevices.propertyAddress(kAudioTapPropertyUID)
        guard let tapUID = AudioDevices.readString(tap, address) else {
            throw EngineError.aggregateFailed
        }

        let aggregateUID = UUID().uuidString
        let tapList: [[String: Any]] = [[kAudioSubTapUIDKey as String: tapUID]]
        let aggregateDescription: [String: Any] = [
            kAudioAggregateDeviceNameKey as String: "Mac Sound Lab",
            kAudioAggregateDeviceUIDKey as String: aggregateUID,
            kAudioAggregateDeviceIsPrivateKey as String: 1,
            kAudioAggregateDeviceIsStackedKey as String: 0,
            kAudioAggregateDeviceTapListKey as String: tapList,
            kAudioAggregateDeviceTapAutoStartKey as String: 1,
        ]
        var aggregate = AudioObjectID(kAudioObjectUnknown)
        let aggregateStatus = AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &aggregate)
        guard aggregateStatus == noErr, aggregate != AudioObjectID(kAudioObjectUnknown) else {
            NSLog("MacSoundLab: AudioHardwareCreateAggregateDevice failed: %d", aggregateStatus)
            throw EngineError.aggregateFailed
        }
        aggregateID = aggregate
        NSLog("MacSoundLab: aggregate created id=%u", aggregate)

        // The HRIRs are 48 kHz, so the tap side always runs there.
        var deviceAddress = AudioDevices.propertyAddress(kAudioDevicePropertyNominalSampleRate)
        var rate = Self.processingRate
        AudioObjectSetPropertyData(aggregate, &deviceAddress, 0, nil, UInt32(MemoryLayout<Float64>.size), &rate)

        var formatAddress = AudioDevices.propertyAddress(kAudioDevicePropertyStreamFormat,
                                                        scope: kAudioObjectPropertyScopeInput)
        var format = AudioStreamBasicDescription()
        var formatSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        guard AudioObjectGetPropertyData(aggregate, &formatAddress, 0, nil, &formatSize, &format) == noErr else {
            throw EngineError.aggregateFailed
        }
        tapRate = format.mSampleRate > 0 ? format.mSampleRate : Self.processingRate
        NSLog("MacSoundLab: tap format rate=%.0f ch=%u bytesPerFrame=%u flags=0x%x",
              format.mSampleRate, format.mChannelsPerFrame, format.mBytesPerFrame, format.mFormatFlags)
        address = AudioDevices.propertyAddress(kAudioDevicePropertyBufferFrameSize)
        var bufferFrames: UInt32 = 256
        AudioObjectSetPropertyData(aggregate, &address, 0, nil, UInt32(MemoryLayout<UInt32>.size), &bufferFrames)

        var procID: AudioDeviceIOProcID?
        let status = AudioDeviceCreateIOProcIDWithBlock(&procID, aggregate, ioQueue) { [weak self] _, input, _, _, _ in
            self?.consume(input)
        }
        guard status == noErr, let procID else { throw EngineError.aggregateFailed }
        inputProcID = procID

        let startStatus = AudioDeviceStart(aggregate, procID)
        guard startStatus == noErr else { throw EngineError.tapUnavailable }
    }

    private func consume(_ input: UnsafePointer<AudioBufferList>?) {
        guard let input, let renderer else { return }
        let list = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
        guard list.count > 0 else { return }

        let frames = list.map { Int($0.mDataByteSize) / (max(1, Int($0.mNumberChannels)) * 4) }.max() ?? 0
        guard frames > 0 else { return }
        if logTicks == 0 && !loggedFirstBlock {
            loggedFirstBlock = true
            NSLog("MacSoundLab: first tap block %d frames, %d buffers, %u ch",
                  frames, list.count, list[0].mNumberChannels)
        }

        if inBuffer.count < frames * 2 { inBuffer = [Float](repeating: 0, count: frames * 2) }
        if tapOutBuffer.count < frames * 2 { tapOutBuffer = [Float](repeating: 0, count: frames * 2) }

        inBuffer.withUnsafeMutableBufferPointer { destination in
            guard let base = destination.baseAddress else { return }
            if list.count >= 2, Int(list[0].mNumberChannels) == 1, Int(list[1].mNumberChannels) == 1,
               let leftRaw = list[0].mData, let rightRaw = list[1].mData {
                // Planar: one buffer per channel.
                let left = leftRaw.assumingMemoryBound(to: Float.self)
                let right = rightRaw.assumingMemoryBound(to: Float.self)
                for i in 0..<frames {
                    base[2 * i] = left[i]
                    base[2 * i + 1] = right[i]
                }
            } else if Int(list[0].mNumberChannels) >= 2, let raw = list[0].mData {
                let interleaved = raw.assumingMemoryBound(to: Float.self)
                let stride = Int(list[0].mNumberChannels)
                for i in 0..<frames {
                    base[2 * i] = interleaved[i * stride]
                    base[2 * i + 1] = interleaved[i * stride + 1]
                }
            } else if let raw = list[0].mData {
                let mono = raw.assumingMemoryBound(to: Float.self)
                for i in 0..<frames {
                    base[2 * i] = mono[i]
                    base[2 * i + 1] = mono[i]
                }
            }
        }

        let started = DispatchTime.now().uptimeNanoseconds
        tapOutBuffer.withUnsafeMutableBufferPointer { destination in
            inBuffer.withUnsafeBufferPointer { source in
                guard let inBase = source.baseAddress, let outBase = destination.baseAddress else { return }
                sl_process(renderer, inBase, outBase, Int32(frames), 2)
                ring.write(outBase, frames: frames)
            }
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000

        statsLock.lock()
        stats.dspMs += elapsed
        stats.blocks += 1
        statsLock.unlock()
    }

    // MARK: - output

    private func startOutput(device: AudioOutput) throws {
        var component = AudioComponentDescription(componentType: kAudioUnitType_Output,
                                                  componentSubType: kAudioUnitSubType_HALOutput,
                                                  componentManufacturer: kAudioUnitManufacturer_Apple,
                                                  componentFlags: 0,
                                                  componentFlagsMask: 0)
        guard let found = AudioComponentFindNext(nil, &component) else { throw EngineError.noOutputUnit }
        var unit: AudioUnit?
        guard AudioComponentInstanceNew(found, &unit) == noErr, let unit else { throw EngineError.noOutputUnit }

        var enable: UInt32 = 0
        AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &enable, UInt32(MemoryLayout<UInt32>.size))
        enable = 1
        AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &enable, UInt32(MemoryLayout<UInt32>.size))

        var deviceID = device.id
        AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &deviceID, UInt32(MemoryLayout<AudioDeviceID>.size))

        let outputRate = device.sampleRate > 0 ? device.sampleRate : Self.processingRate
        baseStep = Self.processingRate / outputRate
        var format = AudioStreamBasicDescription(mSampleRate: outputRate,
                                                 mFormatID: kAudioFormatLinearPCM,
                                                 mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
                                                 mBytesPerPacket: 8,
                                                 mFramesPerPacket: 1,
                                                 mBytesPerFrame: 8,
                                                 mChannelsPerFrame: 2,
                                                 mBitsPerChannel: 32,
                                                 mReserved: 0)
        let formatStatus = AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &format,
                                                UInt32(MemoryLayout<AudioStreamBasicDescription>.size))
        guard formatStatus == noErr else {
            AudioComponentInstanceDispose(unit)
            throw EngineError.noOutputUnit
        }

        var callback = AURenderCallbackStruct(inputProc: { reference, _, _, _, frameCount, ioData in
            let engine = Unmanaged<AudioEngine>.fromOpaque(reference).takeUnretainedValue()
            return engine.renderOutput(frameCount: frameCount, ioData: ioData)
        }, inputProcRefCon: Unmanaged.passUnretained(self).toOpaque())
        AudioUnitSetProperty(unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &callback,
                             UInt32(MemoryLayout<AURenderCallbackStruct>.size))

        guard AudioUnitInitialize(unit) == noErr else {
            AudioComponentInstanceDispose(unit)
            throw EngineError.noOutputUnit
        }
        outputUnit = unit

        // Small buffer on the hardware side: 128 frames is ~2.7 ms at 48 kHz.
        var address = AudioDevices.propertyAddress(kAudioDevicePropertyBufferFrameSize)
        var current: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        if AudioObjectGetPropertyData(device.id, &address, 0, nil, &size, &current) == noErr {
            savedBufferSize = current
            var wanted: UInt32 = 128
            AudioObjectSetPropertyData(device.id, &address, 0, nil, UInt32(MemoryLayout<UInt32>.size), &wanted)
        }

        guard AudioOutputUnitStart(unit) == noErr else {
            AudioUnitUninitialize(unit)
            ComponentDispose(unit)
            throw EngineError.noOutputUnit
        }
    }

    private func ComponentDispose(_ unit: AudioUnit) {
        AudioComponentInstanceDispose(unit)
    }

    private func renderOutput(frameCount: UInt32, ioData: UnsafeMutablePointer<AudioBufferList>?) -> OSStatus {
        guard let ioData else { return noErr }
        let list = UnsafeMutableAudioBufferListPointer(ioData)
        let frames = Int(frameCount)
        guard frames > 0, list.count > 0 else { return noErr }

        let fill = ring.fillFrames()

        // The tap delivers whole blocks (often ~10 ms), the output unit pulls
        // 2.7 ms at a time: hold output until the ring is primed, otherwise the
        // first burst is consumed frame by frame and never builds headroom.
        if !primed {
            if fill >= targetFillFrames {
                primed = true
            } else {
                silence(list, frames: frames)
                return noErr
            }
        }

        // Gentle proportional correction, capped at +-0.2% pitch.
        var correction = (fill - targetFillFrames) / targetFillFrames * 0.0008
        correction = min(max(correction, -0.002), 0.002)
        let step = baseStep * (1 + correction)

        if renderBuffer.count < frames * 2 { renderBuffer = [Float](repeating: 0, count: frames * 2) }
        var produced = 0
        renderBuffer.withUnsafeMutableBufferPointer { destination in
            guard let base = destination.baseAddress else { return }
            for i in 0..<frames * 2 { base[i] = 0 }
            produced = ring.read(into: base, frames: frames, step: step)
            if produced == 0 {
                // Starved: re-prime instead of trickling the next burst through.
                primed = false
                statsLock.lock()
                stats.underruns += 1
                statsLock.unlock()
            }
        }

        var peakLeft: Float = 0
        var peakRight: Float = 0
        renderBuffer.withUnsafeBufferPointer {
            guard let base = $0.baseAddress else { return }
            for i in 0..<frames {
                peakLeft = max(peakLeft, abs(base[2 * i]))
                peakRight = max(peakRight, abs(base[2 * i + 1]))
            }
            write(list, from: base, frames: frames)
        }

        statsLock.lock()
        stats.ringFillMs = fill / Self.processingRate * 1000
        stats.levelLeft = max(stats.levelLeft * 0.82, peakLeft)
        stats.levelRight = max(stats.levelRight * 0.82, peakRight)
        statsLock.unlock()
        return noErr
    }

    private func silence(_ list: UnsafeMutableAudioBufferListPointer, frames: Int) {
        for buffer in list {
            if let raw = buffer.mData {
                memset(raw, 0, Int(buffer.mDataByteSize))
            }
        }
    }

    private func write(_ list: UnsafeMutableAudioBufferListPointer,
                       from base: UnsafePointer<Float>, frames: Int) {
        if list.count >= 2, Int(list[0].mNumberChannels) == 1, Int(list[1].mNumberChannels) == 1,
           let leftRaw = list[0].mData, let rightRaw = list[1].mData {
            let left = leftRaw.assumingMemoryBound(to: Float.self)
            let right = rightRaw.assumingMemoryBound(to: Float.self)
            for i in 0..<frames {
                left[i] = base[2 * i]
                right[i] = base[2 * i + 1]
            }
        } else if let raw = list[0].mData {
            let interleaved = raw.assumingMemoryBound(to: Float.self)
            let stride = max(1, Int(list[0].mNumberChannels))
            for i in 0..<frames {
                interleaved[i * stride] = base[2 * i]
                if stride > 1 { interleaved[i * stride + 1] = base[2 * i + 1] }
            }
        }
    }

    // MARK: - UI refresh

    private func startUITimer() {
        uiTimer?.invalidate()
        uiTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            self?.publishStats()
        }
    }

    private func publishStats() {
        guard let renderer else { return }
        var modelMs: Float = 0
        var drops: UInt64 = 0
        sl_stats(renderer, &modelMs, &drops)

        statsLock.lock()
        let blockCount = stats.blocks
        let totalMs = stats.dspMs
        stats.blocks = 0
        stats.dspMs = 0
        let fill = stats.ringFillMs
        let left = stats.levelLeft
        let right = stats.levelRight
        statsLock.unlock()

        self.modelMs = modelMs
        modelDrops += drops
        dspMs = blockCount > 0 ? totalMs / Double(blockCount) : 0
        ringFillMs = fill
        levelLeft = left
        levelRight = right
        sourceIsSurround = sl_is_surround(renderer) == 1
        neuralReady = sl_neural_ready(renderer) == 1
        accelActive = sl_accel_active(renderer) == 1

        logTicks += 1
        if logTicks % 8 == 0 {
            NSLog("MacSoundLab: live fill=%.0fms dsp=%.2fms peak=%.3f/%.3f model=%.2fms",
                  fill, dspMs, left, right, modelMs)
        }
    }

    /// Rough mouth-to-ear estimate shown in the UI.
    var estimatedLatencyMs: Double {
        let fill = ring.fillFrames() / Self.processingRate * 1000
        return fill + 256.0 / Self.processingRate * 1000 + 128.0 / Self.processingRate * 1000
    }
}