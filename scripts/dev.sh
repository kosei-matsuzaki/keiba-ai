#!/usr/bin/env bash
# Start backend (uvicorn), frontend (Vite), and Tauri dev concurrently.
# Killing this script (Ctrl-C) shuts down all child processes via the EXIT trap.
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

GAME_DIR="$SCRIPT_DIR/.."

trap 'kill 0' EXIT

echo "[dev] Starting FastAPI backend on port 8765..."
( cd "$GAME_DIR/backend" && uv run uvicorn keiba_ai.main:app --host 127.0.0.1 --port 8765 --reload ) &

echo "[dev] Starting Vite dev server..."
( cd "$GAME_DIR/frontend" && pnpm dev ) &

# Give backend and frontend a moment to start before launching Tauri dev,
# which attempts to connect to the Vite server immediately.
sleep 2

echo "[dev] Starting Tauri dev..."
( cd "$GAME_DIR/src-tauri" && cargo tauri dev ) &

wait
