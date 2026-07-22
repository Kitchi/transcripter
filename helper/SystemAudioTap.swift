// SystemAudioTap — non-invasive system-audio capture for macOS 14.4+.
//
// Creates a Core Audio *process tap* over global system output and wraps it in
// a private aggregate device. The tap is a read-only observer: your real output
// device stays selected and the hardware volume keys keep working. No BlackHole,
// no Multi-Output.
//
// Output contract:
//   - stderr: one line of JSON describing the stream format, e.g.
//       {"sampleRate":48000,"channels":2,"format":"float32"}
//   - stdout: raw interleaved float32 PCM frames, streamed until killed.
//
// The Python side reads the stderr header once, then drains stdout into the
// existing chunker. Kill with SIGINT/SIGTERM to stop cleanly.

import CoreAudio
import Foundation

// MARK: - Helpers

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data((msg + "\n").utf8))
    exit(1)
}

func check(_ status: OSStatus, _ what: String) {
    if status != noErr { fail("\(what) failed: OSStatus \(status)") }
}

// MARK: - 1. Create a global process tap (taps everything, excludes nothing)

let tapDescription = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
tapDescription.isPrivate = true          // don't surface in other apps
tapDescription.muteBehavior = .unmuted   // observe only; don't mute the output

var tapID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateProcessTap(tapDescription, &tapID),
      "AudioHardwareCreateProcessTap")

// Read the tap's UID so we can reference it from the aggregate device.
var uidAddr = AudioObjectPropertyAddress(
    mSelector: kAudioTapPropertyUID,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
var tapUID: CFString = "" as CFString
var uidSize = UInt32(MemoryLayout<CFString>.size)
check(AudioObjectGetPropertyData(tapID, &uidAddr, 0, nil, &uidSize, &tapUID),
      "read tap UID")

// MARK: - 2. Wrap the tap in a private aggregate device

let aggUID = "transcripter-tap-\(ProcessInfo.processInfo.processIdentifier)"
let aggDescription: [String: Any] = [
    kAudioAggregateDeviceNameKey as String: "Transcripter Tap",
    kAudioAggregateDeviceUIDKey as String: aggUID,
    kAudioAggregateDeviceIsPrivateKey as String: true,
    kAudioAggregateDeviceIsStackedKey as String: false,
    kAudioAggregateDeviceTapAutoStartKey as String: true,
    kAudioAggregateDeviceTapListKey as String: [
        [kAudioSubTapUIDKey as String: tapUID],
    ],
]

var aggID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateAggregateDevice(aggDescription as CFDictionary, &aggID),
      "AudioHardwareCreateAggregateDevice")

// MARK: - 3. Discover the tap stream format

var fmtAddr = AudioObjectPropertyAddress(
    mSelector: kAudioTapPropertyFormat,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
var asbd = AudioStreamBasicDescription()
var asbdSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
check(AudioObjectGetPropertyData(tapID, &fmtAddr, 0, nil, &asbdSize, &asbd),
      "read tap format")

let channels = Int(asbd.mChannelsPerFrame)
let header = "{\"sampleRate\":\(Int(asbd.mSampleRate))," +
             "\"channels\":\(channels),\"format\":\"float32\"}\n"
FileHandle.standardError.write(Data(header.utf8))

// MARK: - 4. Stream PCM from an IO proc to stdout

let stdout = FileHandle.standardOutput
let ioQueue = DispatchQueue(label: "transcripter.tap.io")

var ioProcID: AudioDeviceIOProcID?
let status = AudioDeviceCreateIOProcIDWithBlock(
    &ioProcID, aggID, ioQueue
) { _, inInputData, _, _, _ in
    let bufferList = inInputData.pointee
    // Global tap presents a single interleaved buffer.
    guard bufferList.mNumberBuffers > 0 else { return }
    let buf = bufferList.mBuffers  // first AudioBuffer
    if let data = buf.mData, buf.mDataByteSize > 0 {
        stdout.write(Data(bytes: data, count: Int(buf.mDataByteSize)))
    }
}
check(status, "AudioDeviceCreateIOProcIDWithBlock")
check(AudioDeviceStart(aggID, ioProcID), "AudioDeviceStart")

// MARK: - Cleanup on signal

func teardown() {
    AudioDeviceStop(aggID, ioProcID)
    if let id = ioProcID { AudioDeviceDestroyIOProcID(aggID, id) }
    AudioHardwareDestroyAggregateDevice(aggID)
    AudioHardwareDestroyProcessTap(tapID)
}

let sigHandler: @convention(c) (Int32) -> Void = { _ in
    // Best-effort cleanup; the OS reclaims tap/aggregate on exit regardless.
    exit(0)
}
signal(SIGINT, sigHandler)
signal(SIGTERM, sigHandler)
atexit { teardown() }

// Run until signalled.
dispatchMain()
