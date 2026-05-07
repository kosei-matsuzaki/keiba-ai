#!/usr/bin/env bash
# ブラウザ確認用 dev サーバ一発起動。
# - Python deps を uv sync
# - DB migration を alembic upgrade head（新規テーブル/列の同期）
# - Frontend deps を pnpm install
# - uvicorn (port 8765) + Vite を並列起動（Tauri は起動しない）
#
# Ctrl-C で `trap 'kill 0' EXIT` により全子プロセス停止。
# PR 取り込み直後でもこれ一本で動かせるように、依存同期と migration を毎回実行する。
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
GAME_DIR="$SCRIPT_DIR/.."

trap 'kill 0' EXIT

echo "[dev] Syncing backend dependencies (uv sync)..."
( cd "$GAME_DIR/backend" && uv sync )

echo "[dev] Applying database migrations (alembic upgrade head)..."
( cd "$GAME_DIR/backend" && uv run alembic upgrade head )

echo "[dev] Installing frontend dependencies (pnpm install)..."
( cd "$GAME_DIR/frontend" && pnpm install )

echo "[dev] Starting FastAPI backend on http://127.0.0.1:8765 ..."
( cd "$GAME_DIR/backend" && uv run uvicorn keiba_ai.main:app --host 127.0.0.1 --port 8765 --reload ) &

echo "[dev] Starting Vite dev server (browse to http://localhost:5173) ..."
( cd "$GAME_DIR/frontend" && pnpm dev ) &

wait
