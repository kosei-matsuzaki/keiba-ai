#!/usr/bin/env bash
# ブラウザ確認用 dev サーバ一発起動。
# - Python deps を uv sync
# - DB migration を alembic upgrade head（新規テーブル/列の同期）
# - Frontend deps を pnpm install
# - uvicorn (port 8765) + Vite を並列起動
#
# Ctrl-C で `trap 'kill 0' EXIT` により全子プロセス停止。
# PR 取り込み直後でもこれ一本で動かせるように、依存同期と migration を毎回実行する。
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_ROOT="$SCRIPT_DIR/.."

trap 'kill 0' EXIT

echo "[dev] Syncing backend dependencies (uv sync)..."
( cd "$REPO_ROOT/backend" && uv sync )

echo "[dev] Applying database migrations (alembic upgrade head)..."
( cd "$REPO_ROOT/backend" && uv run alembic upgrade head )

# Frontend deps: pnpm-lock.yaml が node_modules/.modules.yaml より新しいときだけ install。
# pnpm install を毎回走らせると Windows でファイルロック起因の EACCES
# (.ignored_eslint / .ignored_autoprefixer 等) を踏みやすいため、PR 取り込み等で
# lockfile が更新されたタイミングだけ install を実行する。
LOCKFILE="$REPO_ROOT/frontend/pnpm-lock.yaml"
MODULES_MARKER="$REPO_ROOT/frontend/node_modules/.modules.yaml"
if [[ -f "$MODULES_MARKER" && "$MODULES_MARKER" -nt "$LOCKFILE" ]]; then
    echo "[dev] Frontend deps already in sync with pnpm-lock.yaml — skipping pnpm install"
else
    echo "[dev] Installing frontend dependencies (pnpm install)..."
    ( cd "$REPO_ROOT/frontend" && pnpm install )
fi

echo "[dev] Starting FastAPI backend on http://127.0.0.1:8765 ..."
# WATCHFILES_FORCE_POLLING=true: Windows + Git Bash 環境で uvicorn --reload の
# multiprocessing 子プロセス起動が WinError 87 で死ぬ問題への回避策。
# native fs event の代わりにポーリングを使うことでシグナル伝播の不具合を避ける。
( cd "$REPO_ROOT/backend" && WATCHFILES_FORCE_POLLING=true uv run uvicorn main:app --host 127.0.0.1 --port 8765 --reload --reload-dir src ) &

echo "[dev] Starting Vite dev server (browse to http://localhost:5173) ..."
( cd "$REPO_ROOT/frontend" && pnpm dev ) &

wait
