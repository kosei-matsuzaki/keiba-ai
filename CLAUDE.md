# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

netkeiba スクレイピング + NN (Set Transformer ランキング) による競馬予想ツール (個人研究用)。FastAPI バックエンド + React 管理画面の単一リポジトリで、`scripts/dev.sh` が両方を一発起動する。直近コミット `62d59c2` で monorepo (`games/keiba-ai/...`) から単独リポジトリ構成へ再編成されているため、古いパスを参照するコードや設定が散見されたら更新候補と疑うこと。

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
uv run python -m ai.nn.train_nn --train-end 2025-04-30 --valid-months 6 --test-months 6  # NN 学習
uv run python -m ai.evaluate --model data/models/<ts>-nn --persist    # 評価 + metrics_json 書き戻し
uv run keiba-ingest --date 2024-12-28                                       # 単日 ingest
uv run python -m jobs.ingest_range --start ... --end ...           # 期間 ingest（中断後の resume 対応）
uv run keiba-backup                                                # keiba.db + odds.db を data/backups/ に世代バックアップ
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

### 推論パスは bundle 経由 (NN 専用)
- `registry.load_model_full(path)` が `ModelBundle` (`model_type="nn"`) を返す。`bundle.nn_model` が RaceModel、`nn_preprocessor` / `temperature_scaler` は optional
- 推論は **bundle-aware の `predict_race(bundle, frame)` / `predict_race_with_combinations(bundle, frame, ...)` / `predict_race_with_shap(bundle, frame, ...)`** を必ず使う (`ai/predict.py`)
- combo 確率は NN スコア → 解析的 Plackett-Luce で導出する。外部 isotonic 校正 (`combo_calibrators` / win / place) は全廃済み。連系の校正は `combo_nll` / `multi` 損失で **NN 内部に学習**させる (下記「連系の校正を NN 内部へ」)。`predict.py` の `_calibrate()` は identity スタブ
- SHAP は廃止。`predict_race_with_shap` は `top_features=[]` を返す (ルーター互換のための残置スタブ)

### NN は optional dep
`torch` / `lightning` は `pyproject.toml` の `[project.optional-dependencies].nn`。未インストール環境では `load_model_full` / 予測系が `ModuleNotFoundError`。導入は `uv pip install -e ".[nn]"`。scraper/ingest だけなら torch 不要。

### モデル保存レイアウト
- NN: `data/models/<YYYYMMDDTHHMMSS>-nn/` に `model.pt` + `meta.json` (`model_type="nn"`) + optional `preprocessor.pkl` / `temperature_scaler.pkl`
- active は `model_runs.is_active` で管理。`registry._resolve_model_path` が **basename 比較** で WSL/Windows パス差を吸収する (`/mnt/c/...` と `C:\...` のどちらでも当たる)

### 特徴量とリーク防止
- `features/builder.py` の `FEATURE_COLUMNS` (51 列) が単一の真実
- ID 系 (`horse_id` / `jockey_id` / `trainer_id`) は **絶対に FEATURE_COLUMNS に入れない**
- `build_training_frame` / `build_inference_frame` ともに **race_date より厳密に過去** の情報しか参照しない (`_build_entry_row` 内で SQL 条件)。新規特徴量を足すときも同じ制約を維持すること
- `KEIBA_EXCLUDE_ODDS_FEATURES=1` で `ODDS_FEATURE_COLUMNS` を除外した特徴量リストになる (オッズ未確定時の検証用)

### DB スキーマの方針
- `race_id` / `horse_id` / `jockey_id` / `trainer_id` は **TEXT**。netkeiba ID は構造化文字列 (年+回+場+日+R) で算術対象ではない
- Alembic は `migrations/versions/0001` ~ `0010` (最新: `0010_drop_live_odds`)
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
- NN の損失は **`log_growth` (default)** / `log_growth_place` / `log_growth_combo` / `combo_nll` / `multi` / `plackett_luce` / `listmle` / `time_margin` を `--loss` で選択し `meta.json.loss_type` に記録。既定は ROI 志向 (decision-focused): `log_growth` は実オッズの単勝回収率を fractional-Kelly log-growth で直接最適化し、モデル選択も `--monitor valid_tansho_roi` で行う。`log_growth_place`=複勝。最良構成は **二段階** (`plackett_luce` 事前学習 → `--init-from <model_dir>` で fine-tune)。OOS で単複ROIは順位損失・市場の人気1番を有意に上回るが依然 <1.0 (詳細 `docs/ai-model.md`)
- **連系の校正を NN 内部へ**: `log_growth_combo`=連系の賭けリターン (解析的PL combo確率, `--combo-bet-type`), `combo_nll`=連系 combo確率の **NLL (proper scoring rule)** で **外部 isotonic 校正を不要にする** (旧 `combo_calibrators` の代替, `--combo-bet-type all` で全連系), `multi`=`log_growth`+`combo_weight`·`combo_nll(all)` の **全馬券対応の本番目的** (`--combo-weight` 既定0.01)。解析ヘルパ `_pl_exacta`/`_pl_trifecta`/`_winning_combo_prob` (`ai/nn/loss.py`)。注: 連系は控除率25%で校正しても黒字化はしない
- ROI系損失・監視・温度スケーラは **標準化前の生オッズ**を使う必要があるため `odds_win_raw`(単勝)/`place_ret_raw`(複勝)/`combo_payoff_raw`(連系) を非特徴列として dataset/collate に通す (`odds_win` は特徴量で標準化される)。win_prob は softmax(score / T_win)、place_prob は PL Monte Carlo。combo確率は素の PL Monte Carlo (外部 isotonic 校正は全廃済み。`combo_nll`/`multi` 学習で NN 内部に校正が入る)。新しい損失を足すときも `predict_race` の確率変換は共通なので学習側だけ拡張すれば足りる
