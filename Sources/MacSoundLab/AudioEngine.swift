import AppKit
import AudioToolbox
import CoreAudio
import Foundation
import CSoundLab

/// os_unfair_lock at a stable heap address. A lock stored as a plain Swift
/// property and passed as `&lock` is not guaranteed to keep its address, which
/// os_unfair_lock requires; NSLock is a pthread mutex and is heavier on the
/// audio threads.
final class UnfairLock {
    private let pointer: UnsafeMutablePointer<os_unfair_lock>

    init() {
        pointer = .allocate(capacity: 1)
        pointer.initialize(to: os_unfair_lock())
    }

    deinit {
        pointer.deinitialize(count: 1)
        pointer.deallocate()
    }

    @inline(__always) func lock() { os_unfair_lock_lock(pointer) }
    @inline(__always) func unlock() { os_unfair_lock_unlock(pointer) }
}

/// Stereo float ring between the tap IOProc and the output unit, with a
/// slowly adapting resample ratio so the two clocks never drift apart.
/// Raw storage with a power-of-two mask: no array bounds or uniqueness checks
/// and no 64-bit division per sample on the audio threads.
final class AudioRing {
    private let storage: UnsafeMutablePointer<Float>
    private let capacity: Int
    private let mask: Int
    private var writeIndex: Int64 = 0
    private var readPosition: Double = 0
    private let lock = UnfairLock()

    init(capacityPowerOfTwo: Int = 15) {
        capacity = 1 << capacityPowerOfTwo
        mask = capacity - 1
        storage = .allocate(capacity: capacity * 2)
        storage.initialize(repeating: 0, count: capacity * 2)
    }

    deinit {
        storage.deallocate()
    }

    /// Frames the ring can hold before the writer overwrites unread audio.
    var capacityFrames: Int { capacity }

    func write(_ input: UnsafePointer<Float>, frames: Int) {
        guard frames > 0 else { return }
        lock.lock()
        var done = 0
        while done < frames {
            let slot = Int(writeIndex + Int64(done)) & mask
            let chunk = min(frames - done, capacity - slot)
            (storage + slot * 2).update(from: input + done * 2, count: chunk * 2)
            done += chunk
        }
        writeIndex += Int64(frames)
        lock.unlock()
    }

    /// Frames of audio sitting in the ring, from the reader's point of view.
    func fillFrames() -> Double {
        lock.lock()
        let fill = Double(writeIndex) - readPosition
        lock.unlock()
        return fill
    }

    /// Pulls `frames` frames with linear interpolation at `step`.
    /// Returns how many were produced; the caller silences the remainder.
    @discardableResult
    func read(into output: UnsafeMutablePointer<Float>, frames: Int, step: Double) -> Int {
        lock.lock()
        defer { lock.unlock() }

        let available = Double(writeIndex) - readPosition
        if available <= 0 {
            readPosition = Double(writeIndex)
            return 0
        }
        // One extra frame: interpolation reads base + 1.
        if available < Double(frames) * step + 1 {
            return 0
        }
        var position = readPosition
        for i in 0..<frames {
            let whole = position.rounded(.down)
            let fraction = Float(position - whole)
            let base = Int(Int64(whole)) & mask
            let next = (base + 1) & mask
            let al = storage[base * 2], ar = storage[base * 2 + 1]
            let bl = storage[next * 2], br = storage[next * 2 + 1]
            output[2 * i] = al + (bl - al) * fraction
            output[2 * i + 1] = ar + (br - ar) * fraction
            position += step
        }
        readPosition = position
        return frames
    }

    /// After an overflow (the output stalled while the tap kept writing): drop
    /// the backlog and keep a known headroom. Without this the stale audio
    /// played late and the 0.2% drift correction took minutes to catch up.
    func resync(targetFrames: Double) {
        lock.lock()
        readPosition = max(0, Double(writeIndex) - targetFrames)
        lock.unlock()
    }

    func reset() {
        lock.lock()
        writeIndex = 0
        readPosition = 0
        lock.unlock()
    }
}

/// Live numbers for the statistics panel. Kept apart from AudioEngine so a
/// meter tick redraws only the panel that shows it, not the whole window and
/// the menu bar item; values are published only when they visibly change.
final class EngineMeters: ObservableObject {
    @Published fileprivate(set) var dspMs: Double = 0
    @Published fileprivate(set) var latencyMs: Int = 0
    @Published fileprivate(set) var levelLeft: Float = 0
    @Published fileprivate(set) var levelRight: Float = 0
    @Published fileprivate(set) var modelMs: Float = 0
    @Published fileprivate(set) var modelDrops: UInt64 = 0
    @Published fileprivate(set) var sourceIsSurround = false
    @Published fileprivate(set) var neuralReady = false
    @Published fileprivate(set) var accelActive = false

    fileprivate func update<T: Equatable>(_ keyPath: ReferenceWritableKeyPath<EngineMeters, T>, _ value: T) {
        if self[keyPath: keyPath] != value { self[keyPath: keyPath] = value }
    }

    fileprivate func clear() {
        update(\.dspMs, 0)
        update(\.latencyMs, 0)
        update(\.levelLeft, 0)
        update(\.levelRight, 0)
        update(\.modelMs, 0)
        update(\.sourceIsSurround, false)
        update(\.neuralReady, false)
        update(\.accelActive, false)
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
    var outputFrames: UInt64 = 0
    var outputPeak: Float = 0
    var inFrames: UInt64 = 0
    var callbacks: Int = 0
    var lastFrames: Int = 0
    var silentReturns: Int = 0
    var resyncs: Int = 0
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
    private var lastReverbLog: Bool?
    private var dumpInHandle: FileHandle?
    private var dumpOutHandle: FileHandle?
    private var dumpFrames = 0
    private let muteOutput = ProcessInfo.processInfo.environment["SL_MUTE"] != nil
    private var lastPublishAt = CACurrentMediaTime()
    /// Live statistics for the UI; observe this, not the engine, for meters.
    let meters = EngineMeters()
    private var modelDropsTotal: UInt64 = 0

    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var inputProcID: AudioDeviceIOProcID?
    private var outputUnit: AudioUnit?
    private var renderer: OpaquePointer?
    private var outputDevice: AudioOutput?

    private let ring = AudioRing()
    private let ioQueue = DispatchQueue(label: "cz.davnozdu.macsoundlab.tap", qos: .userInteractive)
    private let statsLock = UnfairLock()
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
    private var resumeFade = 0
    private static let resumeFadeFrames = 480
    private static let maxBlockFrames = 4096
    private let targetFillFrames: Double = 1024   // ~21 ms of headroom
    /// Past this the output has stalled (sleep, a Bluetooth reconnect): jump
    /// back to the target instead of playing the backlog late.
    private var maxFillFrames: Double { targetFillFrames * 4 }

    func matchesOutput(_ output: AudioOutput) -> Bool {
        guard let current = outputDevice else { return false }
        return current.uid == output.uid && current.id == output.id && current.sampleRate == output.sampleRate
    }

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
        resumeFade = 0
        lastReverbLog = nil
        dumpFrames = 0
        loggedFirstBlock = false
        outputDevice = nil
        isRunning = false
        stats = EngineStats()
        modelDropsTotal = 0
        meters.update(\.modelDrops, 0)
        meters.clear()
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

    private func startDumpsIfRequested() {
        let env = ProcessInfo.processInfo.environment
        func open(_ key: String) -> FileHandle? {
            guard let path = env[key] else { return nil }
            FileManager.default.createFile(atPath: path, contents: nil)
            return FileHandle(forWritingAtPath: path)
        }
        dumpInHandle = open("SL_DUMP_INPUT")
        dumpOutHandle = open("SL_DUMP_OUTPUT")
        if dumpInHandle != nil || dumpOutHandle != nil {
            NSLog("MacSoundLab: dumping audio to %@ / %@", env["SL_DUMP_INPUT"] ?? "-", env["SL_DUMP_OUTPUT"] ?? "-")
        }
    }

    private func dump(_ handle: FileHandle?, _ base: UnsafePointer<Float>, frames: Int) {
        guard let handle, dumpFrames < 6 * 48000 else { return }
        handle.write(Data(bytes: base, count: frames * 2 * MemoryLayout<Float>.size))
    }

    private func startTap() throws {
        startDumpsIfRequested()
        // Sized for the largest IO block the HAL hands out, so the IOProc never
        // allocates on the audio thread.
        if inBuffer.count < Self.maxBlockFrames * 2 { inBuffer = [Float](repeating: 0, count: Self.maxBlockFrames * 2) }
        if tapOutBuffer.count < Self.maxBlockFrames * 2 { tapOutBuffer = [Float](repeating: 0, count: Self.maxBlockFrames * 2) }
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

        // Read the description back: the mute we asked for is not necessarily
        // the mute we got, and that is the difference between "processed audio
        // replaces the original" and "processed audio plays on top of it".
        var descAddress = AudioDevices.propertyAddress(kAudioTapPropertyDescription)
        var returned: CATapDescription?
        var descSize = UInt32(MemoryLayout<CATapDescription?>.size)
        let readStatus = withUnsafeMutablePointer(to: &returned) { pointer in
            AudioObjectGetPropertyData(tap, &descAddress, 0, nil, &descSize, pointer)
        }
        if readStatus == noErr, let returned {
            NSLog("MacSoundLab: tap mute behavior = %d (asked for %d), name=\(returned.name)",
                  returned.muteBehavior.rawValue, CATapMuteBehavior.muted.rawValue)
        } else {
            NSLog("MacSoundLab: tap description read failed: %d", readStatus)
        }

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

        // A plain loop: list.map allocated an array on every IO callback.
        var frames = 0
        for buffer in list {
            frames = max(frames, Int(buffer.mDataByteSize) / (max(1, Int(buffer.mNumberChannels)) * 4))
        }
        guard frames > 0 else { return }
        statsLock.lock()
        stats.inFrames += UInt64(frames)
        statsLock.unlock()


        if logTicks == 0 && !loggedFirstBlock {
            loggedFirstBlock = true
            NSLog("MacSoundLab: first tap block %d frames, %d buffers, %u ch",
                  frames, list.count, list[0].mNumberChannels)
        }

        // Preallocated in startTap; growing here is a fallback for an unusually
        // large block and allocates once.
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
            dump(dumpInHandle, base, frames: frames)
        }

        let started = DispatchTime.now().uptimeNanoseconds
        tapOutBuffer.withUnsafeMutableBufferPointer { destination in
            inBuffer.withUnsafeBufferPointer { source in
                guard let inBase = source.baseAddress, let outBase = destination.baseAddress else { return }
                sl_process(renderer, inBase, outBase, Int32(frames), 2)
                if muteOutput {
                    for i in 0..<frames * 2 { outBase[i] = 0 }
                }
                dump(dumpOutHandle, outBase, frames: frames)
                if dumpInHandle != nil || dumpOutHandle != nil { dumpFrames += frames }
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
        if renderBuffer.count < Self.maxBlockFrames * 2 {
            renderBuffer = [Float](repeating: 0, count: Self.maxBlockFrames * 2)
        }

        // Small buffer on the hardware side: 128 frames is ~2.7 ms at 48 kHz.
        var address = AudioDevices.propertyAddress(kAudioDevicePropertyBufferFrameSize)
        var current: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        if AudioObjectGetPropertyData(device.id, &address, 0, nil, &size, &current) == noErr {
            savedBufferSize = current
            var wanted: UInt32 = 128
            AudioObjectSetPropertyData(device.id, &address, 0, nil, UInt32(MemoryLayout<UInt32>.size), &wanted)
        }

        let startStatus = AudioOutputUnitStart(unit)
        NSLog("MacSoundLab: output device=%u \"%@\" uid=%@ rate=%.0f start=%d",
              device.id, device.name, device.uid, outputRate, startStatus)
        guard startStatus == noErr else {
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

        var fill = ring.fillFrames()
        if fill > maxFillFrames {
            ring.resync(targetFrames: targetFillFrames)
            fill = targetFillFrames
            resumeFade = Self.resumeFadeFrames
            statsLock.lock()
            stats.resyncs += 1
            statsLock.unlock()
        }

        // The tap delivers whole blocks (often ~10 ms), the output unit pulls
        // 2.7 ms at a time: hold output until the ring is primed, otherwise the
        // first burst is consumed frame by frame and never builds headroom.
        if !primed {
            if fill >= targetFillFrames {
                primed = true
            } else {
                silence(list, frames: frames)
                statsLock.lock()
                stats.silentReturns += 1
                statsLock.unlock()
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
                // Starved. Re-priming here used to cost a full 21 ms of silence,
                // which is exactly the click the ear hears as a dropout; keep the
                // stream alive and fade back in when the ring recovers.
                resumeFade = Self.resumeFadeFrames
                statsLock.lock()
                stats.underruns += 1
                statsLock.unlock()
            } else if resumeFade > 0 {
                let ramp = min(resumeFade, produced)
                let total = Self.resumeFadeFrames
                let done = total - resumeFade
                for i in 0..<ramp {
                    let gain = Float(done + i) / Float(total)
                    base[2 * i] *= gain
                    base[2 * i + 1] *= gain
                }
                resumeFade -= ramp
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
            statsLock.lock()
            stats.outputFrames += UInt64(frames)
            stats.outputPeak = max(stats.outputPeak, max(peakLeft, peakRight))
            stats.callbacks += 1
            stats.lastFrames = frames
            statsLock.unlock()
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
        let timer = Timer(timeInterval: 0.25, repeats: true) { [weak self] _ in
            self?.publishStats()
        }
        timer.tolerance = 0.1   // lets the system coalesce the wake-ups
        RunLoop.main.add(timer, forMode: .common)
        uiTimer = timer
    }

    /// True when some app window is on screen. Without one there is nobody to
    /// look at the meters, so they are not published at all.
    private var statsVisible: Bool {
        NSApp?.windows.contains { $0.canBecomeMain && $0.isVisible && $0.occlusionState.contains(.visible) } ?? false
    }

    private func publishStats() {
        guard let renderer else { return }
        let now = CACurrentMediaTime()
        let elapsed = max(1e-6, now - lastPublishAt)
        lastPublishAt = now
        var modelMs: Float = 0
        var drops: UInt64 = 0
        sl_stats(renderer, &modelMs, &drops)
        modelDropsTotal += drops

        statsLock.lock()
        let blockCount = stats.blocks
        let totalMs = stats.dspMs
        stats.blocks = 0
        stats.dspMs = 0
        let fill = stats.ringFillMs
        let left = stats.levelLeft
        let right = stats.levelRight
        let outFrames = stats.outputFrames
        let inFr = stats.inFrames
        let underruns = stats.underruns
        let resyncs = stats.resyncs
        stats.inFrames = 0
        stats.outputFrames = 0
        stats.outputPeak = 0
        stats.callbacks = 0
        stats.silentReturns = 0
        statsLock.unlock()

        let dsp = blockCount > 0 ? totalMs / Double(blockCount) : 0

        if statsVisible {
            // Rounded to what the panel shows, so an unchanged number does not
            // trigger a redraw.
            meters.update(\.dspMs, (dsp * 100).rounded() / 100)
            meters.update(\.latencyMs, Int(estimatedLatencyMs))
            meters.update(\.levelLeft, (left * 50).rounded() / 50)
            meters.update(\.levelRight, (right * 50).rounded() / 50)
            meters.update(\.modelMs, (modelMs * 100).rounded() / 100)
            meters.update(\.modelDrops, modelDropsTotal)
            meters.update(\.sourceIsSurround, sl_is_surround(renderer) == 1)
            meters.update(\.neuralReady, sl_neural_ready(renderer) == 1)
            meters.update(\.accelActive, sl_accel_active(renderer) == 1)
        }

        logTicks += 1
        if logTicks % 40 == 0 {   // every 10 s
            NSLog("MacSoundLab: live fill=%.0fms dsp=%.2fms peak=%.3f/%.3f model=%.2fms drops=%llu in=%.0f/s out=%.0f/s underruns=%d resyncs=%d",
                  fill, dsp, left, right, modelMs, modelDropsTotal, Double(inFr) / elapsed,
                  Double(outFrames) / elapsed, underruns, resyncs)
        }
    }

    /// Rough mouth-to-ear estimate shown in the UI: the ring, the tap block,
    /// the output buffer and the renderer's own delay (which includes the
    /// separator pipeline, ~29 ms, while Movie runs it).
    var estimatedLatencyMs: Double {
        let fill = ring.fillFrames() / Self.processingRate * 1000
        let dsp = renderer.map { Double(sl_latency_frames($0)) } ?? 0
        return fill + (256.0 + 128.0 + dsp) / Self.processingRate * 1000
    }
}