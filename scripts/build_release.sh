#!/usr/bin/env bash
# Full release build:
#   1. PyInstaller → src-tauri/binaries/keiba-ai-backend-<triple>[.exe]
#   2. pnpm install + tauri build
#   3. Copy the resulting exe to games/keiba-ai/keiba-ai.exe
#
# NOTE: This script produces a Windows EXE only when run on Windows or with
# a Windows cross-compilation toolchain. On Linux/WSL it is expected to fail
# at step 2 when targeting Windows; run it on Windows to produce keiba-ai.exe.
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
GAME_DIR="$SCRIPT_DIR/.."

# ── Step 1: Build the Python sidecar ───────────────────────────────────────
echo "[build_release] Step 1/3 — Building FastAPI sidecar with PyInstaller..."
bash "$SCRIPT_DIR/build_backend.sh"

# ── Step 2: Install frontend deps + run tauri build ────────────────────────
echo "[build_release] Step 2/3 — Installing frontend dependencies..."
( cd "$GAME_DIR/frontend" && pnpm install )

echo "[build_release] Step 2/3 — Building Tauri application..."
( cd "$GAME_DIR/frontend" && pnpm tauri:build )

# ── Step 3: Copy the bundled exe to the game root ──────────────────────────
echo "[build_release] Step 3/3 — Copying executable..."

TAURI_RELEASE_DIR="$GAME_DIR/src-tauri/target/release"

if [[ "$(uname -s)" == *MINGW* ]] || [[ "$(uname -s)" == *CYGWIN* ]] || \
   [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win"* ]]; then
    # Windows
    SRC_EXE="$TAURI_RELEASE_DIR/keiba-ai.exe"
    DEST_EXE="$GAME_DIR/keiba-ai.exe"
else
    # Linux/macOS (cross-build or native)
    SRC_EXE="$TAURI_RELEASE_DIR/keiba-ai"
    DEST_EXE="$GAME_DIR/keiba-ai"
fi

if [[ -f "$SRC_EXE" ]]; then
    cp "$SRC_EXE" "$DEST_EXE"
    echo "[build_release] Done: $DEST_EXE"
else
    echo "[build_release] WARNING: Expected executable not found at $SRC_EXE"
    echo "  On Linux/WSL, produce the Windows EXE by running this script on Windows."
    exit 1
fi
