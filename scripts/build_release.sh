#!/usr/bin/env bash
# Full release build:
#   1. PyInstaller → sidecar を src-tauri/binaries/ と games/keiba-ai/binaries/
#      の両方に配置（実体は build_backend.sh が実行）
#   2. pnpm install + tauri build (invoked from games/keiba-ai/ so tauri.conf.json
#      is discovered correctly — running from frontend/ fails)
#   3. Copy the resulting exe to games/keiba-ai/keiba-ai.exe
#
# NOTE: Produces a Windows EXE only when run on Windows. On Linux/WSL the
# Tauri build step requires GTK/WebKit dev headers and is not supported.
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
GAME_DIR="$SCRIPT_DIR/.."

is_windows() {
    [[ "$(uname -s)" == *MINGW* ]] || [[ "$(uname -s)" == *CYGWIN* ]] || \
    [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win"* ]]
}

# ── Step 1: Build the Python sidecar ───────────────────────────────────────
# build_backend.sh は src-tauri/binaries/ と games/keiba-ai/binaries/ の
# 両方にサイドカーを配置するので、ここでは追加の staging は不要。
echo "[build_release] Step 1/3 — Building FastAPI sidecar with PyInstaller..."
bash "$SCRIPT_DIR/build_backend.sh"

# ── Step 2: Install frontend deps + run tauri build ────────────────────────
echo "[build_release] Step 2/3 — Installing frontend dependencies..."
( cd "$GAME_DIR/frontend" && pnpm install )

echo "[build_release] Step 2/3 — Building Tauri application..."
# Tauri CLI must be invoked from the directory that contains src-tauri/ as a
# subfolder so tauri.conf.json discovery succeeds. Call the locally-installed
# binary directly while keeping CWD at GAME_DIR.
TAURI_BIN="$GAME_DIR/frontend/node_modules/.bin/tauri"
if [[ ! -x "$TAURI_BIN" && -f "$TAURI_BIN.cmd" ]]; then
    TAURI_BIN="$TAURI_BIN.cmd"
fi
if [[ ! -e "$TAURI_BIN" ]]; then
    echo "[build_release] ERROR: Tauri CLI not found at $TAURI_BIN"
    echo "  Run 'pnpm install' inside frontend/ first."
    exit 1
fi
( cd "$GAME_DIR" && "$TAURI_BIN" build )

# ── Step 3: Copy the bundled exe to the game root ──────────────────────────
echo "[build_release] Step 3/3 — Copying executable to games/keiba-ai/..."

TAURI_RELEASE_DIR="$GAME_DIR/src-tauri/target/release"

if is_windows; then
    SRC_EXE="$TAURI_RELEASE_DIR/keiba-ai.exe"
    DEST_EXE="$GAME_DIR/keiba-ai.exe"
else
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
