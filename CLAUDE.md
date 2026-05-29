# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

netkeiba スクレイピング + LightGBM / NN による競馬予想ツール (個人研究用)。FastAPI バックエンド + React 管理画面の単一リポジトリで、`scripts/dev.sh` が両方を一発起動する。直近コミット `62d59c2` で monorepo (`games/keiba-ai/...`) から単独リポジトリ構成へ再編成されているため、古いパスを参照するコードや設定が散見されたら更新候補と疑うこと。

詳しい仕様は `docs/` 配下 (`spec.md` / `design.md` / `ai-model.md` / `data-pipeline.md` / `operations.md`) を参照。本ファイルは「コード全体を読まないと掴めない big picture」のみを要約する。

## 日常コマンド

### 起動
```bash
bash scripts/dev.sh   # uv sync + alembic upgrade + pnpm install + uvicorn(:8765) + Vite(:5173)
```
PR 取り込み直後でもこれ一本で動くよう、毎回同期と migration を走らせる。Ctrl-C で `trap 'kill 0' EXIT` が全プロセス停止。

### テスト・リント
```bash
# backend
cd backend && uv run pytest                           # 全テスト
cd backend && uv run pytest tests/test_foo.py::TestX  # 単体
cd backend && uv run ruff check src tests

# frontend
cd frontend && pnpm test           # vitest run
cd frontend && pnpm test -- foo    # 単体（ファイル名フィルタ）
cd frontend && pnpm lint
cd frontend && pnpm build          # tsc -b && vite build
```

### CLI エントリ (backend/)
```bash
uv run python -m ai.gbm.train                                          # 学習
uv run python -m ai.gbm.train --loss plackett_luce                     # GBDT を PL モードで学習
uv run python -m ai.evaluate --model data/models/<ts> --persist    # 評価 + metrics_json 書き戻し
uv run python -m ai.gbm.tune --n-trials 50                             # Optuna ハイパラ探索
uv run keiba-ingest --date 2024-12-28                                       # 単日 ingest
uv run python -m jobs.ingest_range --start ... --end ...           # 期間 ingest（中断後の resume 対応）
uv run alembic upgrade head
```

### WSL から動かす場合
リポジトリは `/mnt/c/...` 上にあるため、`backend/.venv/` は **Windows 側 uv** が使う想定。WSL から `uv sync` / `uv run` を叩くときは Windows の `.venv/` を上書きしないよう、`UV_PROJECT_ENVIRONMENT` で WSL 専用 venv を `/tmp/` (ext4) に切る:

```bash
# 依存同期（WSL 用 venv を /tmp に作成）
UV_PROJECT_ENVIRONMENT=/tmp/keiba-linux-venv uv sync --extra nn

# テスト実行（PYTHONPATH=src は src/ レイアウト用）
UV_PROJECT_ENVIRONMENT=/tmp/keiba-linux-venv PYTHONPATH=src \
    uv run pytest -q

# 単体テスト
UV_PROJECT_ENVIRONMENT=/tmp/keiba-linux-venv PYTHONPATH=src \
    uv run pytest tests/test_foo.py::test_bar
```

ブランチ切替時の汚染回避用に `/tmp/keiba-linux-venv-main` のような別 venv を併存させる運用もある (`.claude/settings.local.json` 参照)。`/tmp/` 上に置く理由は (1) `/mnt/c/` の Linux I/O が遅い、(2) Windows 側の `.venv/` と上書きが起こると両方壊れる、の 2 点。逆に Windows 側から起動する uvicorn / pytest は素の `uv sync` / `uv run` で OK (環境変数なし)。

## アーキテクチャ要点

### 依存方向 (厳守)
```
api → jobs → ai / features / scraper → db (SQLAlchemy models)
core / settings は横断
```
`ai` は `scraper` を直接呼ばない (循環禁止)。`api/routers/*.py` はビジネスロジックを持たず、下層モジュールを呼ぶだけ。

### 推論パスは bundle 経由で必ず GBDT/NN 切替
- `registry.load_model_full(path)` が `ModelBundle` を返し、`bundle.model_type` が `"gbdt"` か `"nn"`
- GBDT は `bundle.lambdarank`、NN は `bundle.nn_model` を持つ (片側は None)
- 推論は **bundle-aware の `predict_race(bundle, frame)` / `predict_race_with_combinations(bundle, frame, ...)` / `predict_race_with_shap(bundle, frame, ...)`** を必ず使う (`ai/predict.py`)。内部で `bundle.model_type` を見て GBDT / NN に分岐する
- 低レイヤの `predict_race_gbdt` / `predict_race_with_combinations_gbdt` / `predict_race_with_shap_gbdt` は Booster を直接受け取る学習時専用 (train.py / calibrate.py / evaluate.py の内部用)
- SHAP TreeExplainer は GBDT 専用。NN 経路では `top_features=[]` を返す (ルーターで分岐)

### NN は optional dep
`torch` / `lightning` は `pyproject.toml` の `[project.optional-dependencies].nn`。インストールしていない環境で NN モデルを active にすると `registry._load_nn_bundle` が `ModuleNotFoundError`。導入は `uv pip install -e ".[nn]"`。

### モデル保存レイアウト
- GBDT: `data/models/<YYYYMMDD-HHMMSS>/` に `model.txt` (lambdarank 必須) + `binary.txt` / `calibrator.pkl` / `combo_calibrators.pkl` / `temperature_scaler.pkl` (optional) + `meta.json`
- NN: `data/models/<YYYYMMDDTHHMMSS>-nn/` に `model.pt` + `meta.json` (`model_type="nn"`)
- active は `model_runs.is_active` で管理。`registry._resolve_model_path` が **basename 比較** で WSL/Windows パス差を吸収する (`/mnt/c/...` と `C:\...` のどちらでも当たる)

### 特徴量とリーク防止
- `features/builder.py` の `FEATURE_COLUMNS` (38 列) が単一の真実
- ID 系 (`horse_id` / `jockey_id` / `trainer_id`) は **絶対に FEATURE_COLUMNS に入れない**
- `build_training_frame` / `build_inference_frame` ともに **race_date より厳密に過去** の情報しか参照しない (`_build_entry_row` 内で SQL 条件)。新規特徴量を足すときも同じ制約を維持すること
- `KEIBA_EXCLUDE_ODDS_FEATURES=1` で `ODDS_FEATURE_COLUMNS` を除外した特徴量リストになる (オッズ未確定時の検証用)

### DB スキーマの方針
- `race_id` / `horse_id` / `jockey_id` / `trainer_id` は **TEXT**。netkeiba ID は構造化文字列 (年+回+場+日+R) で算術対象ではない
- Alembic は `migrations/versions/0001` ~ `0003` (最新: `0003_add_scrape_log_fetched_at_index`)
- FK CASCADE: `entries.race_id` / `payouts.race_id` は CASCADE、`horse_id` は RESTRICT、`jockey_id` / `trainer_id` は SET NULL

### ジョブはインメモリ
- `api/jobs.py` の `JobRegistry` が `asyncio.create_task` でバックグラウンド実行を管理
- **プロセス再起動でジョブ状態は消える**。永続化を増やすときは明示的にユーザー合意を取る
- `/api/scraper/run` と `/api/models/train` は 202 Accepted を即時返却

### スクレイパー停止
`scraper/stop_flag.py` が「環境変数 `KEIBA_SCRAPER_STOP=1` か、プロセス内フラグ」をループのたびに見る。UI / API / 環境変数の 3 経路で止められる。新しいループを書くときも `is_stopped()` を呼ぶこと。

### バインドと CORS
- uvicorn は `127.0.0.1` のみにバインド (外部不可)
- CORS 許可は Vite dev (`localhost:5173` / `127.0.0.1:5173`) + 環境変数 `KEIBA_CORS_EXTRA` (カンマ区切り) 追加分のみ
- 認証なし (ローカル単体起動前提)

### フロントの規約
- `shadcn` CLI は走らせず、`src/components/ui/` に手書き配置 (`components.json` は設定の記録のみ)
- `badge.tsx` に `success` / `warning` / `info` バリアントを追加済み。色は CSS 変数 (`globals.css`) + `tailwind.config.ts` 経由。ハードコードの Tailwind カラークラスではなくバリアントを使う
- API クライアントは `lib/api.ts` (ky)。エラー整形は `formatErrorMessage` / `formatErrorMessageSync`、HTTP status は `getStatus` / `isNotFoundError` 等のヘルパで判定

## 注意ポイント

- 環境変数 `KEIBA_DATA_DIR` で `data/` の場所を切り替え可能。テストでは `tmp_path` ベースで上書きする (`conftest.py` 参照)
- `core/paths.py` の `data_dir()` を経由してパスを組み立てる。`data/` 配下の直書きは避ける
- WSL から Windows venv の Python を呼ばない (パス・改行・ロックの問題が出る)。Windows uvicorn が動いているときに WSL で `uv sync` する場合は別 `UV_PROJECT_ENVIRONMENT` を渡す
- `lambdarank` / `plackett_luce` の 2 モードはチェックポイントの `meta.json.loss_type` で識別。確率変換ロジックが分岐するので、新しいモードを追加するときは `predict_race` 内の分岐と calibrator まわりを揃えること
