#!/usr/bin/env bash
# Build and sign the SystemAudioTap helper as a minimal .app bundle.
#
# The bundle (Info.plist + code signature) is what lets macOS request and
# remember the *system-audio-recording* permission and list the helper under
# System Settings -> Privacy & Security -> Screen & System Audio Recording.
# A bare CLI binary cannot hold that grant and silently receives zeroed audio.
#
# Produces a universal (arm64 + x86_64) binary, signed with a stable self-signed
# identity so the TCC grant survives rebuilds.
#
# One-time setup -- create the free self-signed code-signing certificate:
#   1. Open Keychain Access.
#   2. Menu: Keychain Access -> Certificate Assistant -> Create a Certificate...
#   3. Name: "Transcripter Dev"  (must match $IDENTITY below)
#      Identity Type: Self Signed Root
#      Certificate Type: Code Signing
#   4. Create. No Apple Developer account or payment required.
#
# Then:  ./build.sh
#
# For a quick local test without making a cert, ad-hoc sign instead:
#   TRANSCRIPTER_SIGN_IDENTITY=- ./build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/SystemAudioTap.swift"
PLIST="$HERE/Info.plist"
ENTITLEMENTS="$HERE/SystemAudioTap.entitlements"

BIN_DIR="$HERE/../src/transcripter/_bin"
APP="$BIN_DIR/SystemAudioTap.app"
MACOS_DIR="$APP/Contents/MacOS"
EXE="$MACOS_DIR/system-audio-tap"
IDENTITY="${TRANSCRIPTER_SIGN_IDENTITY:-Transcripter Dev}"

echo "assembling bundle -> $APP"
rm -rf "$APP"
mkdir -p "$MACOS_DIR"
cp "$PLIST" "$APP/Contents/Info.plist"

echo "compiling universal binary"
swiftc -O -target arm64-apple-macos14.4  -o "$EXE.arm64"  "$SRC"
swiftc -O -target x86_64-apple-macos14.4 -o "$EXE.x86_64" "$SRC"
lipo -create -output "$EXE" "$EXE.arm64" "$EXE.x86_64"
rm -f "$EXE.arm64" "$EXE.x86_64"

echo "signing bundle with identity: $IDENTITY"
codesign --force --sign "$IDENTITY" \
    --entitlements "$ENTITLEMENTS" \
    --options runtime --timestamp=none \
    "$APP"
codesign --verify --verbose "$APP"

echo "done: $APP"
lipo -info "$EXE"
