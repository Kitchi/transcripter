// SystemAudioTap — non-invasive system-audio capture for macOS 14.4+.
//
// Creates a Core Audio *process tap* over global system output and wraps it in
// a private aggregate device. The tap is a read-only observer: the user's real
// output device stays selected and the hardware volume keys keep working. No
// BlackHole, no Multi-Output.
//
// Launch & transport: this helper MUST be launched via LaunchServices (`open`)
// so macOS treats it as its own TCC-responsible process and attributes the
// system-audio-recording permission to *this bundle* (via Info.plist) rather
// than the spawning terminal. LaunchServices detaches stdio, so instead of
// stdout we connect back to a unix-domain socket the caller is listening on,
// passed as argv[1].
//
// Wire format on the socket:
//   - one line of JSON + '\n' describing the stream, e.g.
//       {"sampleRate":48000,"channels":2,"format":"float32"}
//   - then raw interleaved float32 PCM frames until the peer closes (we exit on
//     the resulting write error) or we receive SIGINT/SIGTERM.

import CoreAudio
import Darwin
import Foundation

// MARK: - Helpers

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data((msg + "\n").utf8))
    exit(1)
}

func check(_ status: OSStatus, _ what: String) {
    if status != noErr { fail("\(what) failed: OSStatus \(status)") }
}

// MARK: - Connect back to the caller's unix-domain socket

guard CommandLine.arguments.count >= 2 else {
    fail("usage: system-audio-tap <unix-socket-path>")
}
let socketPath = CommandLine.arguments[1]

let sockFD = socket(AF_UNIX, SOCK_STREAM, 0)
guard sockFD >= 0 else { fail("socket() failed: errno \(errno)") }

var addr = sockaddr_un()
addr.sun_family = sa_family_t(AF_UNIX)
socketPath.withCString { src in
    withUnsafeMutablePointer(to: &addr.sun_path) { rawTuple in
        rawTuple.withMemoryRebound(to: CChar.self, capacity: MemoryLayout.size(ofValue: addr.sun_path)) {
            _ = strncpy($0, src, MemoryLayout.size(ofValue: addr.sun_path) - 1)
        }
    }
}
let connectRC = withUnsafePointer(to: &addr) { rawAddr in
    rawAddr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
        connect(sockFD, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
    }
}
guard connectRC == 0 else { fail("connect(\(socketPath)) failed: errno \(errno)") }

// Peer close should surface as a send() error, not a process-killing SIGPIPE.
signal(SIGPIPE, SIG_IGN)

@discardableResult
func writeAll(_ ptr: UnsafeRawPointer, _ count: Int) -> Bool {
    var offset = 0
    while offset < count {
        let n = send(sockFD, ptr + offset, count - offset, 0)
        if n <= 0 { return false }
        offset += n
    }
    return true
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

// MARK: - 3. Discover the tap stream format and send the header

var fmtAddr = AudioObjectPropertyAddress(
    mSelector: kAudioTapPropertyFormat,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
var asbd = AudioStreamBasicDescription()
var asbdSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
check(AudioObjectGetPropertyData(tapID, &fmtAddr, 0, nil, &asbdSize, &asbd),
      "read tap format")

let header = "{\"sampleRate\":\(Int(asbd.mSampleRate))," +
             "\"channels\":\(Int(asbd.mChannelsPerFrame)),\"format\":\"float32\"}\n"
_ = Array(header.utf8).withUnsafeBytes { writeAll($0.baseAddress!, $0.count) }

// MARK: - 4. Stream PCM from an IO proc to the socket

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
        if !writeAll(data, Int(buf.mDataByteSize)) {
            exit(0)  // peer closed; our work is done.
        }
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

let sigHandler: @convention(c) (Int32) -> Void = { _ in exit(0) }
signal(SIGINT, sigHandler)
signal(SIGTERM, sigHandler)
atexit { teardown() }

dispatchMain()
