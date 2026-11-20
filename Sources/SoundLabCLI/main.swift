import AVFoundation
import Foundation
import CSoundLab

// Offline renderer: same DSP path as the live app, no audio permissions needed.
//
//   soundlab-cli in.wav out.wav [--profile game|movie|music|speech]
//                [--device headphones|speakers|glasses]
//                [--no-neural] [--no-reverb] [--bypass]
//                [--model vocals.onnx] [--runtime /path/with/libonnxruntime.dylib]

let rate: Double = 48000
let chunk = 512

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

var positional: [String] = []
var options: [String: String] = [:]
var flags = Set<String>()
var index = 1
let arguments = CommandLine.arguments
while index < arguments.count {
    let arg = arguments[index]
    if arg.hasPrefix("--") {
        let name = String(arg.dropFirst(2))
        if ["no-neural", "no-reverb", "bypass", "help"].contains(name) {
            flags.insert(name)
        } else if index + 1 < arguments.count {
            options[name] = arguments[index + 1]
            index += 1
        }
    } else {
        positional.append(arg)
    }
    index += 1
}

if flags.contains("help") || positional.count < 2 {
    print("""
    soundlab-cli in.wav out.wav [--profile game|movie|music|speech] [--device headphones|speakers|glasses]
                 [--no-neural] [--no-reverb] [--bypass] [--model vocals.onnx] [--runtime dir]
    """)
    exit(positional.count < 2 ? 1 : 0)
}

let profiles: [String: Int32] = ["off": 0, "game": 1, "movie": 2, "music": 3, "speech": 4]
let devices: [String: Int32] = ["headphones": 0, "speakers": 1, "glasses": 2]

let profileName = options["profile"] ?? "movie"
guard let profile = profiles[profileName] else { fail("Неизвестный профиль: \(profileName)") }
let deviceName = options["device"] ?? "headphones"
guard let device = devices[deviceName] else { fail("Неизвестный тип выхода: \(deviceName)") }

// --- input ---------------------------------------------------------------

let inputURL = URL(fileURLWithPath: positional[0])
let outputURL = URL(fileURLWithPath: positional[1])
let input: AVAudioFile
do {
    input = try AVAudioFile(forReading: inputURL)
} catch {
    fail("Не открывается \(inputURL.path): \(error.localizedDescription)")
}

let sourceFormat = input.processingFormat
guard let sourceBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat,
                                          frameCapacity: AVAudioFrameCount(max(1, input.length))) else {
    fail("Не хватило памяти на вход")
}
do {
    try input.read(into: sourceBuffer)
} catch {
    fail("Ошибка чтения: \(error.localizedDescription)")
}

var stereo = [Float](repeating: 0, count: Int(sourceBuffer.frameLength) * 2)
if let channels = sourceBuffer.floatChannelData {
    let count = Int(sourceBuffer.frameLength)
    if sourceFormat.channelCount >= 2 {
        for i in 0..<count {
            stereo[2 * i] = channels[0][i]
            stereo[2 * i + 1] = channels[1][i]
        }
    } else {
        for i in 0..<count {
            stereo[2 * i] = channels[0][i]
            stereo[2 * i + 1] = channels[0][i]
        }
    }
} else {
    fail("Неподдерживаемый формат входа (нужен PCM float)")
}

var sourceRate = sourceFormat.sampleRate
if abs(sourceRate - rate) > 0.5 {
    sourceRate = sourceFormat.sampleRate
    print("Ресемплинг \(Int(sourceRate)) → \(Int(rate)) Гц")
    let frames = Int(Double(stereo.count / 2) * rate / sourceRate)
    var resampled = [Float](repeating: 0, count: frames * 2)
    let ratio = sourceRate / rate
    let sourceFrames = stereo.count / 2
    for i in 0..<frames {
        let position = Double(i) * ratio
        let base = min(sourceFrames - 2, Int(position))
        let fraction = Float(position - Double(base))
        for channel in 0..<2 {
            let a = stereo[base * 2 + channel]
            let b = stereo[(base + 1) * 2 + channel]
            resampled[i * 2 + channel] = a + (b - a) * fraction
        }
    }
    stereo = resampled
}

// --- render --------------------------------------------------------------

guard let renderer = sl_create(Float(rate), profile, device,
                               flags.contains("no-neural") ? 0 : 1,
                               flags.contains("no-reverb") ? 0 : 1) else {
    fail("Не удалось создать DSP")
}
defer { sl_destroy(renderer) }

if let runtime = options["runtime"] {
    runtime.withCString { sl_set_runtime_dir(renderer, $0) }
}
if let model = options["model"] {
    model.withCString { path in
        "".withCString { speech in sl_set_models(renderer, path, speech) }
    }
}
sl_set_bypass(renderer, flags.contains("bypass") ? 1 : 0)

let totalFrames = stereo.count / 2
var output = [Float](repeating: 0, count: totalFrames * 2)
let started = Date()

stereo.withUnsafeBufferPointer { source in
    output.withUnsafeMutableBufferPointer { destination in
        guard let inBase = source.baseAddress, let outBase = destination.baseAddress else { return }
        var offset = 0
        while offset < totalFrames {
            let frames = min(chunk, totalFrames - offset)
            sl_process(renderer, inBase + offset * 2, outBase + offset * 2, Int32(frames), 2)
            offset += frames
        }
    }
}

let elapsed = Date().timeIntervalSince(started)
let audioSeconds = Double(totalFrames) / rate

func rms(_ samples: [Float]) -> Double {
    guard !samples.isEmpty else { return 0 }
    var sum = 0.0
    for value in samples { sum += Double(value) * Double(value) }
    return (sum / Double(samples.count)).squareRoot()
}

let peak = output.reduce(Float(0)) { max($0, abs($1)) }
print(String(format: "Вход: %.1f с, RMS %.4f | Выход: RMS %.4f, пик %.3f", audioSeconds, rms(stereo), rms(output), peak))
var modelMs: Float = 0
var modelDrops: UInt64 = 0
sl_stats(renderer, &modelMs, &modelDrops)
print("Нейросеть: \(sl_neural_ready(renderer) == 1 ? "загружена" : "не загружена"), "
      + String(format: "%.2f мс/блок, сбоев %llu", modelMs, modelDrops)
      + " | ускорение: \(sl_accel_active(renderer) == 1 ? "CoreML" : "CPU")")
print(String(format: "Рендер: %.2f с на %.1f с звука (%.2f× реального времени)",
             elapsed, audioSeconds, elapsed / max(audioSeconds, 0.001)))

// --- output --------------------------------------------------------------

let settings: [String: Any] = [
    AVFormatIDKey: kAudioFormatLinearPCM,
    AVSampleRateKey: rate,
    AVNumberOfChannelsKey: 2,
    AVLinearPCMBitDepthKey: 32,
    AVLinearPCMIsFloatKey: true,
    AVLinearPCMIsNonInterleaved: false,
]
var writer: AVAudioFile? = try? AVAudioFile(forWriting: outputURL, settings: settings,
                                            commonFormat: .pcmFormatFloat32, interleaved: false)
guard let file = writer else {
    fail("Не открывается на запись \(outputURL.path)")
}
guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                    frameCapacity: AVAudioFrameCount(totalFrames)) else {
    fail("Не хватило памяти на выход")
}
buffer.frameLength = AVAudioFrameCount(totalFrames)
if let channels = buffer.floatChannelData {
    for i in 0..<totalFrames {
        channels[0][i] = output[2 * i]
        channels[1][i] = output[2 * i + 1]
    }
}
do {
    try file.write(from: buffer)
} catch {
    fail("Ошибка записи: \(error.localizedDescription)")
}
// AVAudioFile finalises the header on deinit, so drop the last reference here.
writer = nil
print("Записано: \(outputURL.path)")
