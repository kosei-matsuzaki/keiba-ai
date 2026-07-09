# KEIBA AI 運用ガイド

このドキュメントは KEIBA AI の運用に必要な手順をまとめたガイドです。
日常的なデータ取り込みとモデル管理を行う運用者・開発者の双方が使えるよう、章立てを目的別に整理しています。

関連ドキュメント: [spec.md](spec.md) / [design.md](design.md) / [data-pipeline.md](data-pipeline.md) / [ai-model.md](ai-model.md) / [README.md](README.md)

---

## クイックスタート

1. `bash scripts/dev.sh` を実行して FastAPI と Vite を起動する
2. ブラウザで `http://localhost:5173` を開く
3. Settings 画面で User-Agent と取り込みレート（秒）を設定する
4. Ingest 画面で初回データ取り込みを実行する
5. Models 画面で初回学習を実行し、生成されたモデルを active に切り替える
6. Upcoming Races 画面でレース一覧を確認し、Race Detail 画面で馬ごとの予想を確認する

---

## 起動と終了

`scripts/dev.sh` は FastAPI (uvicorn :8765) と React 管理画面 (Vite :5173) を並列起動し、`trap 'kill 0' EXIT` で Ctrl-C 押下時に全子プロセスを停止します。詳細な依存同期・マイグレーションフローは後述の「開発」セクションを参照してください。

---

## 日常運用

このセクションは、データ取り込み・モデル学習・設定管理の日常的な操作を行う運用者向けです。

### データ取り込み

単日の取り込みは UI、CLI、API の 3 通りで実行できます。期間指定の連続取り込みには `ingest_range` を使います。

```bash
cd backend

# 単日（CLI）
uv run keiba-ingest --date 2024-12-28

# 単日・先頭 N レースのみ（動作確認用）
uv run keiba-ingest --date 2024-12-28 --limit 3

# 単日（API — 202 Accepted を即時返却、バックグラウンド実行）
curl -X POST http://127.0.0.1:8765/api/scraper/run \
     -H "Content-Type: application/json" \
     -d '{"start_date": "2024-01-01", "end_date": "2024-12-31"}'

# 期間指定（CLI）
uv run python -m jobs.ingest_range \
    --start 2021-01-01 --end 2025-12-31

# 期間指定・1 日あたりのレース数制限（負荷低減用）
uv run python -m jobs.ingest_range \
    --start 2021-01-01 --end 2021-01-31 --limit-per-day 3
```

取り込み結果は `data/keiba.db`（SQLite）に保存されます。HTML キャッシュは `data/raw/<yyyy>/<mm>/` に配置されます。中断後は同じコマンドを再実行するだけでレジュームできます（`scrape_log` に `status='ok'` が記録されている日は自動スキップ）。

### 即時停止

実行中のスクレイピングは UI、API、環境変数の 3 通りで停止できます。

```bash
# API 経由で停止
curl -X POST http://127.0.0.1:8765/api/scraper/stop

# 環境変数でプロセス起動前に停止フラグを立てる
KEIBA_SCRAPER_STOP=1 uv run keiba-ingest --date 2024-12-28
```

UI からは Settings 画面の停止スイッチ、または Ingest 画面の即時停止ボタンで操作できます。

内部的には `scraper/stop_flag.py` の `is_stopped()` がループのたびに環境変数とプロセス内フラグを検査します。

### 進捗確認

```bash
# 未取得日数を確認（直近 30 日）
curl http://127.0.0.1:8765/api/scraper/status

# 直近 N 日で確認
curl "http://127.0.0.1:8765/api/scraper/status?range=60"

# 直近 10 分の fetch 件数・rate・最新 race_id を確認（CLI ingest 監視用）
curl "http://127.0.0.1:8765/api/scraper/recent_activity?minutes=10"

# ジョブ一覧
curl http://127.0.0.1:8765/api/jobs

# 特定ジョブの状態
curl http://127.0.0.1:8765/api/jobs/{job_id}
```

UI の Ingest 画面でも ScraperStatusCard がポーリング表示します（アイドル: 30 秒間隔 / 実行中: 5 秒間隔）。ScraperStatusCard は「直近 10 分: N fetch (ok X, err Y)」「最新 race_id」「CLI 進行中バッジ」を表示するため、`ingest_range` を CLI で実行中も UI を開くだけで 1 分あたり何 fetch 進んでいるかを即座に確認できます。

UI の Models / Ingest 画面では JobProgressCard が API ジョブ（`POST /api/models/train`、`POST /api/scraper/run`）の進捗を 2 秒間隔で polling し、terminal status（completed / failed）で自動停止します。

### モデル学習

```bash
cd backend

# 二段階学習（推奨）: PL 事前学習 → multi で fine-tune（単複 ROI + 連系校正）
uv run python -m ai.training.train_nn --loss plackett_luce --monitor valid_ndcg3 \
    --train-end 2025-12-31 --valid-months 12 --test-months 6
uv run python -m ai.training.train_nn --loss multi --monitor valid_tansho_roi \
    --init-from data/models/<事前学習ts>-nn \
    --train-end 2025-12-31 --valid-months 12 --test-months 6
```

学習後はバックテスト評価で指標を確認します。

```bash
# バックテスト評価（NDCG@1/3/5, Top-1 hit, place_hit, payback_win, payback_place）
uv run python -m ai.evaluation.backtest --model data/models/20260501T120000-nn

# 評価結果を model_runs.metrics_json に保存する（Dashboard MetricCard に反映させる場合は必須）
uv run python -m ai.evaluation.backtest --model data/models/20260501T120000-nn --persist

# 1 番人気常時投票ベースラインとの比較（delta = model - baseline を追加出力）
uv run python -m ai.evaluation.backtest --model data/models/20260501T120000-nn \
    --baseline favorite
```

> **`--persist` を使う理由**: `train.py` が出力する `model_runs.metrics_json` には NDCG 系指標のみが含まれる。`top1_hit` / `place_hit` / `payback_win` / `payback_place` / `n_races` は `evaluate.py` が計算するため、`--persist` なしでは Dashboard の MetricCard がこれらを「—」と表示する。学習直後に `--persist` 付きで評価を実行することを推奨する。

評価指標は API からも確認できます。

```bash
curl http://127.0.0.1:8765/api/metrics/summary
curl http://127.0.0.1:8765/api/metrics/timeseries
```

Dashboard 画面の AccuracyChart にもメトリクス推移が表示されます。

### active モデルの切り替え

新モデルの評価指標が既存 active モデルより改善していれば active に切り替えます。

```bash
# API 経由（{id} は GET /api/models で確認した model_runs テーブルの id）
curl -X POST http://127.0.0.1:8765/api/models/{id}/activate
```

UI の Models 画面でも Activate ボタンから操作できます。

### モデル世代管理

学習済みモデルは以下の構造で `data/models/` に保存されます（`.gitignore` 対象）。

```text
backend/data/models/
├── 20260501T120000-nn/
│   ├── model.pt         # PyTorch state_dict
│   ├── meta.json        # params / train_range / valid_range / metrics / feature_columns / loss_type
│   ├── preprocessor.pkl # NNPreprocessor（標準化・カテゴリ符号化、optional）
│   ├── temperature_scaler.pkl  # win_prob 温度スケーリング（optional）
│   ├── history_norm.pkl        # 履歴 GRU トークンの正規化統計（optional）
│   └── speed_figure.pkl        # タイム指数の par-time 統計（KEIBA_SPEED_FIGURE 有効時のみ）
├── 20260601T090000-nn/
│   └── ...
```

- ディレクトリ名は UTC 時刻 `%Y%m%dT%H%M%S` + サフィックス `-nn` で生成します
- active モデルはシンボリックリンクではなく、`model_runs` テーブルの `is_active=1` で管理します
- 古いモデルは自動削除されません。手動で削除するまで保持されます
- ディスク容量の目安: 1 モデルあたり数 MB 程度（NN のパラメータ数による）

モデル一覧・詳細は `GET /api/models[/{id}]`、UI の Models 画面でも確認できます。

### 設定の永続化

`PUT /api/settings` で変更した設定値は `data/settings.json` に自動保存されます。サーバ再起動後も設定は引き継がれます。

設定スキーマ（`data/settings.json` のキー）:

| キー | 説明 | デフォルト |
|---|---|---|
| `user_agent` | スクレイパーの User-Agent 文字列 | 個人研究目的の文字列 |
| `rate_min_seconds` | レート制御の最小待機秒数 | 3.0 |
| `rate_max_seconds` | レート制御の最大待機秒数 | 6.0 |
| `night_min_seconds` | 夜間の最小待機秒数 | 5.0 |
| `win_ev_threshold` | 単勝 EV 閾値（1.0 以上が必須） | 1.1 |
| `place_ev_threshold` | 複勝 EV 閾値（1.0 以上が必須） | 1.05 |
| `scraper_stopped` | スクレイパー停止フラグ | false |

`data/settings.json` は `.gitignore` 対象です。手動で削除するとデフォルト値にリセットされます。API は `GET /api/settings` / `PUT /api/settings`、UI は Settings 画面で操作できます。

### バックアップ方針

DB のバックアップは `keiba-backup` コマンドで取得する。SQLite の **online backup API**
を使うため、ingest 稼働中でも一貫したスナップショットが安全に取れる（`cp` は WAL の
途中状態を拾う恐れがあるため使わない）。出力先は `data/backups/<name>-<YYYYMMDD-HHMMSS>.db`、
DB ごとに最新 N 世代（既定 7）を保持し古いものは自動削除される。

```bash
cd backend
uv run keiba-backup                 # keiba.db + odds.db、各 7 世代
uv run keiba-backup --db keiba      # keiba.db のみ
uv run keiba-backup --keep 14       # 保持世代数を変更
```

SQLite ファイルを素の `cp` で複製する場合は、書き込みプロセスが停止していることを確認してから行うこと。

| 対象 | 推奨頻度 | 備考 |
|---|---|---|
| `data/keiba.db`（SQLite） | 取り込み完了直後 + 週次 | `keiba-backup` で `data/backups/keiba-*.db` に保存。スキーマは Alembic 管理 |
| `data/odds.db`（確定オッズ） | 取り込み完了直後 | `keiba-backup` で `data/backups/odds-*.db` に保存。再スクレイプ可能なため優先度は中 |
| `data/models/`（モデルファイル） | 月次再学習後 + 重要モデル随時 | active モデルは失うと再学習が必要 |
| `data/raw/<yyyy>/<mm>/`（レース結果 HTML） | 任意 | 再フェッチ可能なため低優先。misc/ は ingest_range が自動削除するためバックアップ対象外 |

ログ: 長時間ジョブのログを残すには `KEIBA_LOG_DIR=data/logs` を設定して実行する
（`data/logs/<script>-<ts>.log` に出力）。未設定ならコンソールのみ（従来どおり）。


---

## 本番運用フロー（実 netkeiba データ）

このセクションは、実際の netkeiba データを使って本番品質のモデルを構築・運用する場合の手順です。

### 前提確認

取り込みを開始する前に以下を確認します。

- netkeiba の利用規約（https://www.netkeiba.com/）および `robots.txt` を読み、スクレイピングが現時点で禁止されていないことを確認する
- User-Agent に個人研究目的と連絡先を含めて設定する（Settings 画面または `data/settings.json`）
- ディスク容量の目安: レース結果 HTML（`data/raw/<yyyy>/<mm>/`）と SQLite DB 合計で約 1 GB（5 年分）。misc キャッシュ（馬詳細・血統・カレンダー）は `ingest_range` の各日完了後に自動削除されるため蓄積しない。余裕を見て 5 GB 以上の空きを確保してから開始する
- misc キャッシュを残してデバッグしたい場合は `KEIBA_KEEP_MISC_CACHE=1` を設定して実行する（デフォルトは自動削除）

### 過去データの取り込み

```bash
cd backend
uv run python -m jobs.ingest_range \
    --start 2021-01-01 \
    --end 2025-12-31
```

推定所要時間: デフォルト（中央のみ）では約 5 年 × 270 開催日 × 36 レース × 4 秒平均（レート制御込み）で **3〜5 時間 / 1 開催日** に相当し、総計で数日程度の連続稼働を想定します。`KEIBA_INCLUDE_NAR=1`（地方込み）の場合は 1 開催日あたり最大 92 レースとなり **8〜12 時間 / 1 開催日** に増加します。非開催日のスキップで実際には短縮されます。中断後は同じコマンドを再実行するだけでレジュームできます（`scrape_log` の ok ログがある日は自動スキップ）。

**新規 horse フェッチによる追加時間**: 初回取り込み時、horse ごとに詳細ページ・血統ページの 2 ページを追加フェッチするため 1 頭あたり 6〜12 秒（3〜6 秒 × 2 リクエスト）が加算されます。5 年分の初回取り込みでは数千頭の新規 horse が登場するため、総所要時間は数時間単位で増加します（同一 horse が複数レースに出走するケースは 2 回目以降スキップされるため実際の増加は全頭数より小さくなります）。

### 本番モデル学習

十分なデータが蓄積された後（目安: 1〜2 年以上の実データ）に本番パラメータで学習します。

```bash
cd backend

# PL 事前学習 → multi fine-tune
uv run python -m ai.training.train_nn --loss plackett_luce --monitor valid_ndcg3 \
    --train-end 2025-12-31 --valid-months 12 --test-months 6
uv run python -m ai.training.train_nn --loss multi --monitor valid_tansho_roi \
    --init-from data/models/<事前学習ts>-nn \
    --train-end 2025-12-31 --valid-months 12 --test-months 6
```

`multi` は log_growth（単勝 betting return）と combo_nll（連系 校正）の重み付き和で、単勝〜三連単までを 1 モデルで賄います。評価指標を確認し、既存の active モデルより改善していれば active に切り替えます。

### 月次再学習

毎月 1 日を目安に、以下の流れで再学習します。

1. 最新データを取り込む（Ingest 画面または `keiba-ingest`）
2. `uv run python -m ai.training.train_nn --loss multi --monitor valid_tansho_roi --init-from <事前学習ts>-nn --train-end $(date +%Y-%m-%d)` で再学習する
3. バックテスト評価で指標を確認する
4. 旧モデルより改善していれば active に切り替える
5. netkeiba の利用規約と robots.txt を確認する（月次確認のタイミングとして推奨）

---

## 開発者向けセットアップ（コードを修正する場合）

このセクションは、バックエンド・フロントエンドのコードを修正する開発者向けです。

### 前提ツール

| ツール | インストール方法 |
|---|---|
| Python 3.12 以上 | python.org または pyenv |
| uv | `pip install uv` または `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js 20 LTS 以上 | nodejs.org または nvm |
| pnpm | `npm install -g pnpm` |

### 初回セットアップ

```bash
# バックエンド
cd backend
uv sync
uv run alembic upgrade head  # 現在の最新: 0012_add_horses_sire_dam_index

# フロントエンド（別ターミナル）
cd frontend
pnpm install
```

`uv sync` で SQLAlchemy / Alembic / pandas / numpy / scikit-learn / FastAPI / uvicorn / pydantic-settings を含む全依存関係がインストールされます（NN 学習・推論には `uv sync --extra nn` で torch / lightning を追加）。`alembic upgrade head` を実行すると `data/keiba.db` に全 10 テーブル（races / horses / jockeys / trainers / entries / payouts / scrape_log / model_runs / bet_records / simulation_runs）が作成されます。確定オッズ用の `data/odds.db`（race_odds テーブル）は Alembic 管理外で、`db/odds_db.py` が初回アクセス時に自動作成します。

既存 DB を持つ場合（例: 途中のリビジョンまで適用済みの DB を使い続けている場合）も同じコマンドで差分のみ適用されます。

```bash
cd backend
uv run alembic current   # 適用済みリビジョンを確認
uv run alembic upgrade head  # 未適用のリビジョン（最新: 0012）を差分適用
```

### 開発サーバ起動

ブラウザ確認用の dev サーバ（uvicorn + Vite）一発起動:

```bash
bash scripts/dev.sh
# → http://localhost:5173 (Vite) / http://127.0.0.1:8765 (FastAPI)
# Ctrl-C で全プロセス停止
```

`scripts/dev.sh` は実行のたびに `uv sync` / `alembic upgrade head` / `pnpm install` を行うため、PR 取り込み直後でもこれ一本で動く。

個別起動が必要な場合:

```bash
# バックエンドのみ（ポート 8765）
cd backend
uv run uvicorn main:app --host 127.0.0.1 --port 8765 --reload

# 別ポートを使う場合
KEIBA_API_PORT=9000 uv run python -m main

# フロントエンドのみ（http://localhost:5173）
cd frontend
pnpm dev
```

主な環境変数:

| 変数 | 説明 | デフォルト |
|---|---|---|
| `KEIBA_API_PORT` | バックエンドのポート番号 | 8765（直接起動時） |
| `KEIBA_CORS_EXTRA` | 追加で許可する CORS オリジン（カンマ区切り） | なし |
| `KEIBA_DATA_DIR` | data/ ディレクトリのパス | backend/data/ |
| `VITE_KEIBA_API_BASE_URL` | フロントの API ベース URL（ブラウザ直接起動時） | http://127.0.0.1:8765 |

### テスト

```bash
# バックエンドテスト（pytest）
cd backend
uv run pytest

# フロントエンドテスト（Vitest）
cd frontend
pnpm test

# リント
uv run ruff check src tests
pnpm lint
```

---

## 障害対応（トラブルシューティング）

このセクションは、症状別に原因・確認方法・対処を示します。

### UI に表示されるエラーメッセージの意味

フロントエンドの `lib/api.ts` `formatErrorMessage` が HTTP ステータスコードを以下の日本語メッセージに変換して toast 通知で表示します。

| HTTP ステータス | UI 表示メッセージ（日本語） |
|---|---|
| 400 | リクエストが不正です |
| 404 | データが見つかりません |
| 422 | 入力値が正しくありません |
| 503 | サービスが利用できません。しばらく待ってから再試行してください |
| 5xx（上記以外） | サーバーエラーが発生しました |
| その他 / ネットワーク系 | エラーが発生しました |

FastAPI が `HTTPException` の `detail` フィールドに文字列を返している場合は、ステータスマッピングより `detail` の内容が優先されます。

### API 起動失敗

**症状**: `uv run uvicorn main:app --port 8765` がエラーで起動しない

| 原因 | 確認・対処 |
|---|---|
| `uv sync` が未実施または不完全 | `cd backend && uv sync` を再実行する |
| `data/keiba.db` が存在しない | `uv run alembic upgrade head` で DB を初期化する |
| ポート 8765 が既に使われている | `KEIBA_API_PORT=<別ポート>` を指定して起動するか、競合プロセスを停止する |

### CORS エラー

**症状**: ブラウザの開発ツールに CORS エラーが表示される

デフォルトで `http://localhost:5173` / `http://127.0.0.1:5173`（Vite dev サーバ）は許可済みです。それ以外のオリジンを使う場合は環境変数 `KEIBA_CORS_EXTRA=http://localhost:<port>` を設定してサーバを再起動してください。

### 推論リクエストで 503

**症状**: `GET /api/predictions/{race_id}` が 503 を返す。RaceDetail 画面に「学習済みモデルが見つかりません。Models 画面でモデルをトレーニングしてください。」と表示される

active モデルが存在しません。`GET /api/models` でモデル一覧を確認し、`POST /api/models/{id}/activate` で active モデルを設定してください。モデルがなければ `uv run python -m ai.training.train_nn` で学習します。

### pnpm install 失敗

**症状**: `pnpm install` が失敗する、または `pnpm dev` が起動しない

| 原因 | 確認・対処 |
|---|---|
| Node.js バージョン不足 | `node -v` で 20 LTS 以上であることを確認する。古い場合は nvm 等でアップグレードする |
| pnpm 未インストール | `npm install -g pnpm` でインストールする |
| `node_modules` が壊れている | `rm -rf frontend/node_modules && pnpm install` を再実行する |

### フロントが API に接続できない

**症状**: `pnpm dev` で開いたブラウザで「ネットワークエラー」が表示される

| 原因 | 確認・対処 |
|---|---|
| バックエンドが起動していない | 別ターミナルで `cd backend && uv run uvicorn main:app --host 127.0.0.1 --port 8765 --reload` を実行する |
| ポートが異なる | `VITE_KEIBA_API_BASE_URL=http://127.0.0.1:<port> pnpm dev` で正しいポートを指定する |

### NN 学習エラー

**症状**: `uv run python -m ai.training.train_nn` がエラーまたは異常終了する

| 原因 | 確認・対処 |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | NN は optional dep。`uv sync --extra nn`（または `uv pip install -e ".[nn]"`）で torch / lightning を導入する |
| valid_df のみが空になる（`--valid-months 0` など） | early stopping のモニタが計算できず最大エポックまで学習する。test セットは独立して保持されるためスコアは正常。過学習リスクがあるため本番では非推奨 |
| `RuntimeError: No training data found in the database.` | `data/keiba.db` が存在しないか races / entries テーブルが空。`uv run keiba-ingest --date <YYYY-MM-DD>` でデータを取り込んでから再実行する |
| `--loss multi` の学習が極端に遅い | combo 項が per-race の Python ループのため。`--combo-bet-type 馬連`（デフォルト・ペアのみ）が高速。`all`（三連系含む）は数倍遅くなる |
| valid_tansho_roi が NaN | バリデーション期間に確定オッズ付きの勝ち馬レースが無い。`--valid-months` を広げるか実オッズが揃った期間で評価する |

### DB マイグレーション失敗

**症状**: `uv run alembic upgrade head` がエラーになる

| 原因 | 確認・対処 |
|---|---|
| `alembic.ini` の `sqlalchemy.url` が未設定 | `alembic.ini` の `sqlalchemy.url` が `sqlite:///%(here)s/../data/keiba.db` になっているか確認する |
| `data/` ディレクトリが存在しない | `mkdir -p backend/../data` で作成する |
| 古い inline DDL で作成した DB | 既存の `keiba.db` を削除してから `alembic upgrade head` を再実行する |
| `agari_3f` / `passing` 列が存在しないエラー | 古い DB を継続使用している場合は `uv run alembic upgrade head`（`0002_add_agari_passing` が適用される）を実行する |
| `ix_scrape_log_fetched_at` インデックスが存在しないエラー | `uv run alembic upgrade head`（`0003_add_scrape_log_fetched_at_index` が適用される）を実行する |

```bash
cd backend
uv run alembic current

# ダウングレードして再適用
uv run alembic downgrade -1
uv run alembic upgrade head
```

### スクレイピング失敗

**症状**: Ingest 画面にエラーが多発する / データが取得できない

| 原因 | 確認・対処 |
|---|---|
| netkeiba のサイト構造変更 | `scrape_log` でエラーの URL を確認し、HTML セレクタを修正する |
| 429 Too Many Requests が連続する | 自動で 60 秒ペナルティ待機が入る。それでも続く場合は手動停止して数時間〜半日置いてから再開する |
| robots.txt の Disallow 変更 | robots.txt を確認し、対象 URL が Disallow されていないか確認する |

### Dialog が閉じない / Settings バリデーション残留

**症状**: TrainModelDialog / IngestRunDialog が閉じない、または Settings フォームにエラーが残る

| 原因 | 確認・対処 |
|---|---|
| mutation の `isPending` フラグが true のまま | API が応答を返すまで待つ。応答がなければバックエンドのログを確認する |
| `rate_min > rate_max` になっている | rate_min ≤ rate_max の制約がある。値を修正してから再送信する |
| EV 閾値が 1.0 未満 | 単勝・複勝 EV 閾値はそれぞれ 1.0 以上が必須 |
| `PUT /api/settings` が 422 を返す | バックエンドのログで Pydantic バリデーションエラーの詳細を確認する |

### Vitest / jsdom で ResizeObserver エラー

**症状**: `pnpm test` 実行時に `ResizeObserver is not defined` エラーが発生する

`frontend/src/__tests__/setup.ts` に `vi.stubGlobal('ResizeObserver', ...)` スタブが定義されているか確認します。Recharts が ResizeObserver を使用するため jsdom 環境ではスタブが必要です。

## 規約上の注意

### 停止手順

netkeiba からスクレイピング停止要請を受けた場合、または利用規約の改訂を確認した場合は直ちに以下を実行します。

1. Settings 画面の停止スイッチをクリックするか、`POST /api/scraper/stop` を実行する
2. 既存のローカルデータ・モデルはそのまま保持してよい（新規取得を停止するだけ）
3. 規約内容を確認し、対応方針を検討する（完全廃止・URL 変更・公式 API 利用等）

### 定期確認

- 月次再学習のタイミングで netkeiba の robots.txt と利用規約を確認する
- サイト構造変更（HTML セレクタ変更）を検知した場合はスクレイパーを停止して修正する
- 本ツールは個人研究目的のみ。取得データ・学習済みモデルの第三者提供・公開は行わない
