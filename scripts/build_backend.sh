#!/usr/bin/env bash
# Build the FastAPI backend into a single-file executable with PyInstaller,
# then copy it into src-tauri/binaries/ with the Tauri externalBin naming
# convention: keiba-ai-backend-<target-triple>[.exe].
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

GAME_DIR="$SCRIPT_DIR/.."
BACKEND_DIR="$GAME_DIR/backend"
BINARIES_DIR="$GAME_DIR/src-tauri/binaries"

cd "$BACKEND_DIR"

echo "[build_backend] Syncing Python dependencies (including pyinstaller)..."
uv sync --group dev

echo "[build_backend] Running PyInstaller..."
# --windowed (alias --noconsole): Windows subsystem の EXE を生成し、起動時に
# 黒いコンソールウィンドウが現れないようにする。Tauri 側 (sidecar.rs) で
# Stdio::null() を設定済だが、PyInstaller exe 自身が console subsystem だと
# OS が console を allocate してしまうため、ここでも EXE を windowed にする。
uv run pyinstaller \
    --onefile \
    --windowed \
    --name keiba-ai-backend \
    --collect-all lightgbm \
    --hidden-import uvicorn.logging \
    --hidden-import uvicorn.loops \
    --hidden-import uvicorn.loops.auto \
    --hidden-import uvicorn.protocols \
    --hidden-import uvicorn.protocols.http \
    --hidden-import uvicorn.protocols.http.auto \
    --hidden-import uvicorn.protocols.websockets \
    --hidden-import uvicorn.protocols.websockets.auto \
    --hidden-import uvicorn.lifespan \
    --hidden-import uvicorn.lifespan.on \
    src/keiba_ai/main.py

# Determine target triple from rustc so the filename matches Tauri's expectation.
TARGET_TRIPLE=$(detect_target_triple)

# On Windows the PyInstaller output has .exe; on Linux/macOS it does not.
if is_windows; then
    SRC_EXE="dist/keiba-ai-backend.exe"
    DEST_NAME="keiba-ai-backend-${TARGET_TRIPLE}.exe"
else
    SRC_EXE="dist/keiba-ai-backend"
    DEST_NAME="keiba-ai-backend-${TARGET_TRIPLE}"
fi

mkdir -p "$BINARIES_DIR"
cp "$SRC_EXE" "$BINARIES_DIR/$DEST_NAME"

# unbundled な Tauri 配布 EXE (games/keiba-ai/keiba-ai.exe) は実行時に
# games/keiba-ai/binaries/ から sidecar を解決するので、Tauri 側にも同じ
# サイドカーを配置しておく（PyInstaller 出力は src-tauri 側と完全に一致）。
GAME_BINARIES_DIR="$GAME_DIR/binaries"
mkdir -p "$GAME_BINARIES_DIR"
cp "$SRC_EXE" "$GAME_BINARIES_DIR/$DEST_NAME"

echo "[build_backend] Staged sidecar:"
echo "  - $BINARIES_DIR/$DEST_NAME (Tauri externalBin)"
echo "  - $GAME_BINARIES_DIR/$DEST_NAME (unbundled exe runtime)"
