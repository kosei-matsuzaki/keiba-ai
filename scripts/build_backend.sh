#!/usr/bin/env bash
# Build the FastAPI backend into a single-file executable with PyInstaller,
# then copy it into src-tauri/binaries/ with the Tauri externalBin naming
# convention: keiba-ai-backend-<target-triple>[.exe].
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
GAME_DIR="$SCRIPT_DIR/.."
BACKEND_DIR="$GAME_DIR/backend"
BINARIES_DIR="$GAME_DIR/src-tauri/binaries"

cd "$BACKEND_DIR"

echo "[build_backend] Syncing Python dependencies (including pyinstaller)..."
uv sync --group dev

echo "[build_backend] Running PyInstaller..."
uv run pyinstaller \
    --onefile \
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
if command -v rustc &>/dev/null; then
    TARGET_TRIPLE=$(rustc -vV | sed -n 's/host: //p')
else
    # Fallback: detect OS/arch manually when rustc is unavailable.
    OS=$(uname -s)
    ARCH=$(uname -m)
    case "$OS" in
        Linux*)  TARGET_TRIPLE="${ARCH}-unknown-linux-gnu" ;;
        Darwin*) TARGET_TRIPLE="${ARCH}-apple-darwin" ;;
        MINGW*|MSYS*|CYGWIN*) TARGET_TRIPLE="${ARCH}-pc-windows-msvc" ;;
        *) TARGET_TRIPLE="${ARCH}-unknown-unknown" ;;
    esac
fi

# On Windows the PyInstaller output has .exe; on Linux/macOS it does not.
if [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win"* ]] || \
   [[ "$(uname -s)" == *MINGW* ]] || [[ "$(uname -s)" == *CYGWIN* ]]; then
    SRC_EXE="dist/keiba-ai-backend.exe"
    DEST_NAME="keiba-ai-backend-${TARGET_TRIPLE}.exe"
else
    SRC_EXE="dist/keiba-ai-backend"
    DEST_NAME="keiba-ai-backend-${TARGET_TRIPLE}"
fi

mkdir -p "$BINARIES_DIR"
cp "$SRC_EXE" "$BINARIES_DIR/$DEST_NAME"

echo "[build_backend] Done: $BINARIES_DIR/$DEST_NAME"
