#!/usr/bin/env bash
# ランチャー配布用フルビルド。
#
# Step 1: PyInstaller で FastAPI バックエンドを EXE 化し、
#         src-tauri/binaries/ と games/keiba-ai/binaries/ 両方に配置
#         （Tauri externalBin 命名規約: keiba-ai-backend-<target-triple>[.exe]）
# Step 2: pnpm install + tauri build
#         （tauri.conf.json は GAME_DIR から discovery させる必要があるため
#          frontend/ ではなく GAME_DIR を CWD にして tauri CLI を直接呼ぶ）
# Step 3: 生成された EXE を games/keiba-ai/keiba-ai.exe にコピー
#
# NOTE: Windows 環境でのみ Windows EXE を生成する。Linux/WSL では
#       Tauri build が GTK/WebKit dev headers を要求するため非対応。
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
GAME_DIR="$SCRIPT_DIR/.."
BACKEND_DIR="$GAME_DIR/backend"
TAURI_BINARIES_DIR="$GAME_DIR/src-tauri/binaries"
GAME_BINARIES_DIR="$GAME_DIR/binaries"

# ── helpers ───────────────────────────────────────────────────────────────
# 実行環境が Windows (Git Bash / MSYS / Cygwin) かどうか
is_windows() {
    [[ "$(uname -s)" == *MINGW* ]] || [[ "$(uname -s)" == *CYGWIN* ]] || \
    [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win"* ]]
}

# rustc の host triple を返す（無ければ uname から推定）
detect_target_triple() {
    if command -v rustc &>/dev/null; then
        rustc -vV | sed -n 's/host: //p'
        return
    fi
    local os arch
    os=$(uname -s)
    arch=$(uname -m)
    case "$os" in
        Linux*)  echo "${arch}-unknown-linux-gnu" ;;
        Darwin*) echo "${arch}-apple-darwin" ;;
        MINGW*|MSYS*|CYGWIN*) echo "${arch}-pc-windows-msvc" ;;
        *) echo "${arch}-unknown-unknown" ;;
    esac
}

# ── Step 1: PyInstaller で sidecar EXE を生成 ──────────────────────────────
echo "[build_release] Step 1/3 — Building FastAPI sidecar with PyInstaller..."
(
    cd "$BACKEND_DIR"

    echo "[build_release] Syncing Python dependencies (including pyinstaller)..."
    uv sync --group dev

    echo "[build_release] Running PyInstaller..."
    # --windowed: Windows subsystem の EXE を生成して起動時の黒コンソールを抑止。
    # Tauri 側 (sidecar.rs) で Stdio::null() を設定済だが、PyInstaller exe 自身が
    # console subsystem だと OS が console を allocate するため両方必要。
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
)

TARGET_TRIPLE=$(detect_target_triple)

if is_windows; then
    SRC_SIDECAR="$BACKEND_DIR/dist/keiba-ai-backend.exe"
    DEST_SIDECAR_NAME="keiba-ai-backend-${TARGET_TRIPLE}.exe"
else
    SRC_SIDECAR="$BACKEND_DIR/dist/keiba-ai-backend"
    DEST_SIDECAR_NAME="keiba-ai-backend-${TARGET_TRIPLE}"
fi

mkdir -p "$TAURI_BINARIES_DIR" "$GAME_BINARIES_DIR"
cp "$SRC_SIDECAR" "$TAURI_BINARIES_DIR/$DEST_SIDECAR_NAME"
# unbundled な配布 EXE (games/keiba-ai/keiba-ai.exe) は実行時に
# games/keiba-ai/binaries/ から sidecar を解決するため両方に配置する。
cp "$SRC_SIDECAR" "$GAME_BINARIES_DIR/$DEST_SIDECAR_NAME"

echo "[build_release] Staged sidecar:"
echo "  - $TAURI_BINARIES_DIR/$DEST_SIDECAR_NAME (Tauri externalBin)"
echo "  - $GAME_BINARIES_DIR/$DEST_SIDECAR_NAME (unbundled exe runtime)"

# ── Step 2: Frontend deps + tauri build ────────────────────────────────────
echo "[build_release] Step 2/3 — Installing frontend dependencies..."
( cd "$GAME_DIR/frontend" && pnpm install )

echo "[build_release] Step 2/3 — Building Tauri application..."
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

# ── Step 3: 生成 EXE をゲームルートにコピー ────────────────────────────────
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
