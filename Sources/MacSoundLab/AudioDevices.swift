import AppKit
import CoreAudio
import Foundation

/// One hardware output we can render into.
struct AudioOutput: Identifiable, Hashable {
    let id: AudioDeviceID
    let uid: String
    let name: String
    let transport: UInt32
    let sampleRate: Double
    let channels: Int

    /// What the device most likely is, from the same rules the Ally plugin uses:
    /// vendor names first, then transport, then a name heuristic.
    var kindGuess: OutputKind {
        let lower = name.lowercased()
        if ["xreal", "nreal", "viture", "rokid", "rayneo", "quest", "pico", "air 2"].contains(where: lower.contains) {
            return .glasses
        }
        if transport == kAudioDeviceTransportTypeBluetooth || transport == kAudioDeviceTransportTypeBluetoothLE {
            return .headphones
        }
        if transport == kAudioDeviceTransportTypeHDMI || transport == kAudioDeviceTransportTypeDisplayPort {
            return .speakers
        }
        if transport == kAudioDeviceTransportTypeBuiltIn {
            // A Mac's built-in jack still reports "BuiltIn"; the name gives it away.
            if lower.contains("headphone") || lower.contains("external") { return .headphones }
            return .speakers
        }
        if lower.contains("headphone") || lower.contains("headset") || lower.contains("earbud") || lower.contains("airpods") {
            return .headphones
        }
        if lower.contains("speaker") || lower.contains("display") || lower.contains("tv") { return .speakers }
        return .headphones
    }
}

enum OutputKind: Int, CaseIterable, Identifiable {
    case headphones = 0
    case speakers = 1
    case glasses = 2

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .headphones: return "Наушники"
        case .speakers: return "Динамики"
        case .glasses: return "Очки"
        }
    }
}

enum AudioDevices {
    static func propertyAddress(_ selector: AudioObjectPropertySelector,
                                scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal,
                                element: AudioObjectPropertyElement = kAudioObjectPropertyElementMain) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: element)
    }

    static func readUInt32(_ object: AudioObjectID, _ address: AudioObjectPropertyAddress) -> UInt32? {
        var address = address
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        let status = AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value)
        return status == noErr ? value : nil
    }

    static func readString(_ object: AudioObjectID, _ address: AudioObjectPropertyAddress) -> String? {
        var address = address
        var value: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.size)
        let status = withUnsafeMutablePointer(to: &value) { pointer in
            AudioObjectGetPropertyData(object, &address, 0, nil, &size, pointer)
        }
        return status == noErr ? (value as String) : nil
    }

    static func readDouble(_ object: AudioObjectID, _ address: AudioObjectPropertyAddress) -> Double? {
        var address = address
        var value: Float64 = 0
        var size = UInt32(MemoryLayout<Float64>.size)
        let status = AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value)
        return status == noErr ? value : nil
    }

    static func channelCount(_ object: AudioObjectID, scope: AudioObjectPropertyScope) -> Int {
        var address = propertyAddress(kAudioDevicePropertyStreamConfiguration, scope: scope)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(object, &address, 0, nil, &size) == noErr, size > 0 else { return 0 }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, raw) == noErr else { return 0 }
        let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        return list.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    /// Physical outputs only: aggregates, loopbacks and taps are not choices here.
    static func outputs() -> [AudioOutput] {
        var address = propertyAddress(kAudioHardwarePropertyDevices)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr else {
            return []
        }
        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        var ids = [AudioDeviceID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids) == noErr else {
            return []
        }

        var result: [AudioOutput] = []
        for id in ids {
            let outputChannels = channelCount(id, scope: kAudioObjectPropertyScopeOutput)
            guard outputChannels > 0 else { continue }
            let transport = readUInt32(id, propertyAddress(kAudioDevicePropertyTransportType)) ?? 0
            if transport == kAudioDeviceTransportTypeAggregate || transport == kAudioDeviceTransportTypeVirtual {
                continue
            }
            guard let uid = readString(id, propertyAddress(kAudioDevicePropertyDeviceUID)) else { continue }
            let name = readString(id, propertyAddress(kAudioObjectPropertyName)) ?? uid
            let rate = readDouble(id, propertyAddress(kAudioDevicePropertyNominalSampleRate)) ?? 48000
            result.append(AudioOutput(id: id, uid: uid, name: name, transport: transport,
                                      sampleRate: rate, channels: outputChannels))
        }
        return result.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    static func defaultOutput() -> AudioDeviceID? {
        var address = propertyAddress(kAudioHardwarePropertyDefaultOutputDevice)
        var deviceID = AudioDeviceID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID)
        return status == noErr ? deviceID : nil
    }

    static func device(withUID uid: String) -> AudioOutput? {
        outputs().first { $0.uid == uid }
    }

    /// The audio object that represents this process in the tap list.
    static func ownProcessObject() -> AudioObjectID? {
        var address = propertyAddress(kAudioHardwarePropertyProcessObjectList)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr,
              size > 0 else { return nil }
        let count = Int(size) / MemoryLayout<AudioObjectID>.size
        var ids = [AudioObjectID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids) == noErr else {
            return nil
        }
        let ownPID = getpid()
        for id in ids {
            if let pid = readUInt32(id, propertyAddress(kAudioProcessPropertyPID)), pid_t(pid) == ownPID {
                return id
            }
        }
        return nil
    }
}
