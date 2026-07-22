#!/usr/bin/env bash
# Build and sign the SystemAudioTap helper.
#
# Produces a universal (arm64 + x86_64) binary and code-signs it with a stable
# self-signed identity so its macOS audio-capture (TCC) grant survives rebuilds.
#
# One-time setup — create the free self-signed code-signing certificate:
#   1. Open Keychain Access.
#   2. Menu: Keychain Access -> Certificate Assistant -> Create a Certificate...
#   3. Name: "Transcripter Dev"  (must match $IDENTITY below)
#      Identity Type: Self Signed Root
#      Certificate Type: Code Signing
#   4. Create. No Apple Developer account or payment required.
#
# Then run:  ./build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/SystemAudioTap.swift"
OUT_DIR="$HERE/../src/transcripter/_bin"
OUT="$OUT_DIR/system-audio-tap"
IDENTITY="${TRANSCRIPTER_SIGN_IDENTITY:-Transcripter Dev}"

mkdir -p "$OUT_DIR"

echo "compiling universal binary -> $OUT"
swiftc -O \
    -target arm64-apple-macos14.4 \
    -o "$OUT.arm64" "$SRC"
swiftc -O \
    -target x86_64-apple-macos14.4 \
    -o "$OUT.x86_64" "$SRC"
lipo -create -output "$OUT" "$OUT.arm64" "$OUT.x86_64"
rm -f "$OUT.arm64" "$OUT.x86_64"

echo "signing with identity: $IDENTITY"
codesign --force --sign "$IDENTITY" --timestamp=none "$OUT"
codesign --verify --verbose "$OUT"

echo "done: $OUT"
lipo -info "$OUT"
