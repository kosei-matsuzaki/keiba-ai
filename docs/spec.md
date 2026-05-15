# KEIBA AI — 技術仕様書

関連ドキュメント: [design.md](design.md) / [data-pipeline.md](data-pipeline.md) / [ai-model.md](ai-model.md) / [operations.md](operations.md)

---

## 技術スタック

### バックエンド（Python）

| 項目 | 内容 |
|---|---|
| 言語 | Python 3.12 以上 |
| パッケージ管理 | uv |
| Web フレームワーク | FastAPI 0.111 以上 |
| ASGI サーバ | Uvicorn |
| ORM | SQLAlchemy 2.x（非同期対応） |
| マイグレーション | Alembic |
| DB | SQLite 3 |
| AI / ML | lightgbm 4.x, pandas 2.x, numpy 1.26 以上, scikit-learn 1.4 以上 |
| SHAP | shap 0.45 以上（特徴量寄与表示用） |
| スクレイピング | httpx（非同期 HTTP）, BeautifulSoup4 |
| スケジューラ | APScheduler（週次自動取り込み用） |

### フロントエンド（TypeScript）

| 項目 | 内容 |
|---|---|
| 言語 | TypeScript 5.x |
| フレームワーク | React 18.3 |
| ルーティング | react-router-dom 6.x |
| ビルドツール | Vite 5.x |
| UI コンポーネント | shadcn/ui（手書き配置）+ Tailwind CSS 3.x |
| データフェッチ | TanStack Query (React Query) v5 |
| 状態管理 | Zustand 4.x |
| HTTP クライアント | ky 1.x |
| フォーム | react-hook-form 7.x + @hookform/resolvers + zod 3.x |
| Toast 通知 | sonner 1.x |
| Radix UI | @radix-ui/react-{dialog,select,label,slot,tabs} |
| チャート | Recharts 2.x |
| テスト | Vitest 2.x + @testing-library/react + @testing-library/user-event |
| リンター | ESLint v9 flat config |
| パッケージ管理 | pnpm 9.x |

> **shadcn/ui 配置方針**: `shadcn` CLI は CI 安定性のため走らせず、button / card / table / tabs / badge / skeleton を `src/components/ui/` に手書き配置する。`components.json` は Tailwind 設定（baseColor: slate、cssVariables: true）の記録のみに使用する。`badge.tsx` には shadcn 標準バリアント（default / secondary / destructive / outline）に加えて **success / warning / info** の 3 バリアントを追加しており、`globals.css` の CSS 変数（`--success` / `--warning` / `--info` およびそれぞれの `-foreground`）と `tailwind.config.ts` の `theme.extend.colors` を通じて light / dark 双方に対応する。ハードコードされた Tailwind カラークラスの代わりにこれらバリアントを使用すること。

> **Web フォント**: `index.html` に Google Fonts preconnect + **Inter**（400/500/600/700）・**JetBrains Mono**（400/500）を `display=swap` で読み込む。`globals.css` の `--font-sans` / `--font-mono` CSS 変数の先頭に各フォントを設定し、フォールバックはシステムスタックを維持する。`body` に `font-feature-settings: 'cv11', 'ss01', 'tnum'` を適用し、Inter の代替字形と等幅数字（テーブル内数値の桁揃え）を有効化する。

---

## ディレクトリ構成

```text
.
├── backend/                   # FastAPI + AI + スクレイパー（Python）
│   ├── pyproject.toml         # uv 管理 (Python 依存関係)
│   ├── src/
│   │   └── keiba_ai/
│   │       ├── main.py        # FastAPI app エントリポイント・Uvicorn 起動
│   │       ├── api/           # ルーター群（races, predictions, models, scraper, settings 等）
│   │       ├── core/          # 設定（Settings）・ロギング・DB セッション管理
│   │       ├── db/            # SQLAlchemy モデル定義・Alembic マイグレーション
│   │       ├── scraper/       # netkeiba スクレイパー実装
│   │       │   └── parsers/
│   │       │       ├── race_calendar.py   # 開催日カレンダー
│   │       │       ├── race_result.py     # レース結果
│   │       │       ├── payout.py          # 払戻金
│   │       │       ├── horse_detail.py    # 馬詳細 (name/sex/birth_date)
│   │       │       └── horse_pedigree.py  # 馬血統 (sire/dam)
│   │       ├── features/      # 特徴量エンジニアリング
│   │       │   ├── builder.py         # FEATURE_COLUMNS (38 列) 定義・build_training_frame / build_inference_frame（レース単位バッチ処理）
│   │       │   ├── course.py          # レース・馬番・馬体重系特徴量
│   │       │   ├── horse_history.py   # 馬の過去成績（直近平均着順・上がり3F・同コース実績 等 11 列）
│   │       │   ├── jockey.py          # 騎手成績統計
│   │       │   ├── odds.py            # オッズ・人気系特徴量
│   │       │   ├── pedigree.py        # 血統特徴量（父/母の産駒勝率）
│   │       │   ├── relative_features.py # 同レース内相対特徴量（馬体重 percentile・オッズ順位 等 6 列）
│   │       │   └── trainer.py         # 調教師成績統計
│   │       ├── ai/            # LightGBM 学習・推論・SHAP 計算
│   │       └── jobs/          # APScheduler ジョブ定義（週次取り込み・月次再学習）
│   └── tests/                 # pytest テスト群
│
├── frontend/                  # React + Vite + TypeScript
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── components.json        # shadcn/ui 設定（CLI 不使用・手書き配置の記録用）
│   ├── eslint.config.js       # ESLint v9 flat config
│   ├── .prettierrc
│   ├── public/
│   │   ├── favicon.svg        # ブランド favicon（馬蹄モノグラム、HSL 直書き）
│   │   └── logo.svg           # サイドナビ用ロゴ（favicon 同モチーフ・CSS 変数で色指定）
│   ├── src/
│   │   ├── main.tsx           # React + QueryClient + Router マウント
│   │   ├── App.tsx            # Outlet レイアウト（Sidebar 含む）
│   │   ├── router.tsx         # createBrowserRouter（6 ルート定義）
│   │   ├── globals.css        # Tailwind ベース + CSS 変数
│   │   ├── routes/            # ページコンポーネント
│   │   │   ├── Dashboard.tsx        # ActiveModelCard + MetricCard + AccuracyChart
│   │   │   ├── UpcomingRaces.tsx    # RaceCard 一覧
│   │   │   ├── RaceDetail.tsx       # レース概要 + PredictionTable + SHAP
│   │   │   ├── Models.tsx           # ActiveModelCard + ModelTable + Activate + TrainModelDialog
│   │   │   ├── Ingest.tsx           # ScraperStatusCard + IngestRunDialog + 停止
│   │   │   └── Settings.tsx         # react-hook-form + zod バリデーション
│   │   ├── components/        # 共通コンポーネント
│   │   │   ├── Sidebar.tsx          # サイドナビ（全画面共通）
│   │   │   ├── PageHeader.tsx       # ページ見出し共通コンポーネント（icon: LucideIcon / title / description? / children?）。左の primary tinted アイコンタイル + h1 + サブテキスト + 右側 actions slot。全 6 ルートに適用
│   │   │   ├── MetricCard.tsx       # KPI カード
│   │   │   ├── AccuracyChart.tsx    # 精度推移グラフ（Recharts）
│   │   │   ├── RaceCard.tsx         # 出走レースカード
│   │   │   ├── EmptyState.tsx       # 空状態表示（icon prop: LucideIcon、デフォルト InboxIcon。サイズ 16、opacity-30、stroke-width 1.5）
│   │   │   ├── PlaceholderScreen.tsx # 仮表示（未使用）
│   │   │   ├── PredictionTable.tsx  # 全馬予想テーブル + BUY バッジ
│   │   │   ├── ActiveModelCard.tsx  # active モデルのサマリカード（ID / 学習日時 / NDCG@3 / 単勝回収率）。未設定時は「未設定」バッジ + train ガイドを表示
│   │   │   ├── ModelTable.tsx       # 学習済みモデル一覧テーブル（is_active 行に bg-emerald-500/5 ハイライト）
│   │   │   ├── ScraperStatusCard.tsx # スクレイパー稼働状況カード（直近 N 分 fetch 集計・CLI 進行中バッジを含む）
│   │   │   ├── JobProgressCard.tsx  # ジョブ進捗カード（Models / Ingest に統合）
│   │   │   ├── SettingsForm.tsx     # react-hook-form 設定フォーム。Section（icon + title + description ヘッダ）/ FieldRow（help text 付きフィールド行）ヘルパで 3 セクション構成（スクレイパー / ベッティング期待値 / 運用）
│   │   │   ├── TrainModelDialog.tsx # 再学習確認ダイアログ（react-hook-form + Zod、inline error、open のたびに reset）
│   │   │   ├── IngestRunDialog.tsx  # スクレイピング実行ダイアログ（react-hook-form + Zod、inline error、open のたびに reset）
│   │   │   └── ui/                  # shadcn 手書きコンポーネント
│   │   │       ├── button.tsx / card.tsx / table.tsx
│   │   │       ├── tabs.tsx / badge.tsx / skeleton.tsx
│   │   │       ├── dialog.tsx / form.tsx / input.tsx
│   │   │       ├── label.tsx / select.tsx
│   │   │       ├── toast.tsx / toaster.tsx  # sonner ラッパ
│   │   ├── hooks/             # カスタムフック（TanStack Query ラッパ）
│   │   │   ├── useUpcomingRaces.ts
│   │   │   ├── useMetricsSummary.ts
│   │   │   ├── useMetricsTimeseries.ts
│   │   │   ├── useRaceDetail.ts
│   │   │   ├── usePredictions.ts
│   │   │   ├── useModels.ts
│   │   │   ├── useActivateModel.ts
│   │   │   ├── useTrainModel.ts
│   │   │   ├── useScraperStatus.ts  # refetchInterval による polling
│   │   │   ├── useScraperRun.ts
│   │   │   ├── useScraperStop.ts
│   │   │   ├── useJobStatus.ts      # jobId を 2 秒 polling、terminal status で停止
│   │   │   ├── useScraperRecentActivity.ts  # 実行中 5 秒 / アイドル 30 秒 polling
│   │   │   └── useSettings.ts
│   │   ├── store/             # Zustand ストア
│   │   │   └── app.ts         # useAppStore / useScraperStore（trackedJobId を含む）
│   │   ├── lib/               # API クライアント・ユーティリティ
│   │   │   ├── api.ts         # ky ベース API クライアント（lazy 初期化・getApiBaseUrl() 経由）。getStatus / isNotFoundError / isServiceUnavailableError / isValidationError / formatErrorMessage(async) / formatErrorMessageSync の error helpers を含む
│   │   │   ├── api-base.ts    # getApiBaseUrl()（VITE_KEIBA_API_BASE_URL or http://127.0.0.1:8765）
│   │   │   ├── formatters.ts  # display formatter 集約（8 関数。null/NaN/Infinity を「—」に統一）
│   │   │   ├── query-client.ts
│   │   │   └── cn.ts          # clsx + tailwind-merge ユーティリティ
│   │   └── types/
│   │       └── api.ts         # API レスポンス型定義（JobInfo / ScraperRecentActivity を含む）
│   └── src/__tests__/         # Vitest + @testing-library/react
│       ├── App.test.tsx
│       ├── Dashboard.test.tsx
│       ├── UpcomingRaces.test.tsx
│       ├── RaceDetail.test.tsx
│       ├── Models.test.tsx
│       ├── Ingest.test.tsx
│       ├── Settings.test.tsx
│       ├── lib_api_errors.test.ts  # error helpers 単体テスト（9 ケース）
│       ├── lib_formatters.test.ts  # formatters.ts の全関数ユニットテスト
│       └── setup.ts
│
├── data/                      # ローカルデータ（.gitignore 対象）
│   ├── raw/                   # HTML キャッシュ（<yyyy>/<mm>/<race_id>.html）
│   ├── keiba.db               # SQLite DB 本体
│   └── models/                # 学習済みモデル（<YYYYMMDD-HHMMSS>/{model.txt, meta.json}）
│
└── scripts/                   # 運用スクリプト
    └── dev.sh                 # uv sync + alembic + pnpm install + uvicorn + Vite を一発起動
```

---

## DB スキーマ

SQLite を使用する。ORM は SQLAlchemy 2.x DeclarativeBase + naming_convention で実装し、DB 初期化は `alembic upgrade head` で行う。マイグレーションファイルは `migrations/versions/` に格納されており、現在 3 ファイルが定義されている。

| ファイル | revision | 内容 |
|---|---|---|
| `0001_initial.py` | 0001 | 初期スキーマ（全 8 テーブル作成） |
| `0002_add_agari_passing.py` | 0002 | entries テーブルに `agari_3f` / `passing` 列を追加 |
| `0003_add_scrape_log_fetched_at_index.py` | 0003 | `scrape_log.fetched_at` に単一カラムインデックスを追加（`recent_activity` エンドポイントの full scan 防止） |

### ID 型の方針

`race_id` / `horse_id` / `jockey_id` / `trainer_id` はすべて `TEXT` で扱う。理由:

- netkeiba の race_id は 12 桁（例: `202406010101` = 年 + 開催回 + 競馬場 + 開催日 + R）で **構造化された識別子**。算術演算の対象ではない
- horse_id 等は先頭ゼロを含むケースがあり、INTEGER 化すると情報が失われる
- 文字列のままパースせず透過的に扱うことでスクレイパーとの整合が取りやすい

整数化の余地がある列（年齢・出走頭数・斤量・払戻金等）のみ `INTEGER` を採用する。

### FK CASCADE 方針

| FK | ondelete | 理由 |
|---|---|---|
| entries.race_id → races | CASCADE | レース削除時に出走記録も連動削除 |
| entries.horse_id → horses | RESTRICT | 馬の履歴を保持するため entries を先に消す必要あり |
| entries.jockey_id → jockeys | SET NULL | 騎手引退後もエントリ記録を残す |
| entries.trainer_id → trainers | SET NULL | 調教師も同様 |
| payouts.race_id → races | CASCADE | レースに付随する払戻情報は連動削除 |

### 複合インデックス一覧

| インデックス名 | テーブル | カラム | 用途 |
|---|---|---|---|
| ix_entries_race_id_horse_id | entries | race_id, horse_id | レース × 馬の検索 |
| ix_entries_horse_id_finish_position | entries | horse_id, finish_position | 馬の着順統計（特徴量計算） |
| ix_payouts_race_id_bet_type | payouts | race_id, bet_type | レース × 券種別払戻参照 |
| ix_scrape_log_url_status | scrape_log | url, status | 再試行対象の検索 |
| ix_scrape_log_fetched_at | scrape_log | fetched_at | `recent_activity` の `WHERE fetched_at >= cutoff` 高速化（migration 0003） |
| uq_entries_race_id_horse_id | entries | race_id, horse_id | 同一レース内の馬重複防止（UNIQUE） |

### スキーマ定義

8 テーブル全て ORM 化されている（`races` / `horses` / `jockeys` / `trainers` / `entries` / `payouts` / `scrape_log` / `model_runs`）。

```sql
-- レース基本情報
CREATE TABLE races (
    race_id         TEXT PRIMARY KEY,   -- netkeiba レース ID（例: 202406010101）
    date            TEXT NOT NULL,      -- 開催日 (YYYY-MM-DD)
    course          TEXT NOT NULL,      -- 競馬場名（東京・中山 等）
    surface         TEXT NOT NULL,      -- 馬場種別: '芝' | 'ダ'
    distance        INTEGER NOT NULL,   -- 距離 (m)
    weather         TEXT,               -- 天候
    track_condition TEXT,               -- 馬場状態（良・稍重・重・不良）
    race_class      TEXT,               -- クラス（G1・G2・G3・条件戦 等）
    n_runners       INTEGER,            -- 出走頭数
    payout_win      INTEGER,            -- 単勝払戻金 (円)
    payout_place    TEXT                -- 複勝払戻金 JSON（着順→金額）
);

-- 出走・着順記録（agari_3f・passing 列を含む）
CREATE TABLE entries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id             TEXT NOT NULL REFERENCES races(race_id),
    horse_id            TEXT NOT NULL,
    post_position       INTEGER,        -- 馬番
    jockey_id           TEXT,
    trainer_id          TEXT,
    weight_carried      REAL,           -- 斤量
    age                 INTEGER,
    sex                 TEXT,           -- 牡・牝・セ
    horse_weight        INTEGER,        -- 馬体重 (kg)
    horse_weight_diff   INTEGER,        -- 馬体重増減
    odds_win            REAL,           -- 単勝オッズ
    popularity          INTEGER,        -- 人気順
    finish_position     INTEGER,        -- 着順（完走できなかった場合は NULL）
    finish_time         REAL,           -- タイム (秒)
    margin              TEXT,           -- 着差
    agari_3f            REAL,           -- 上がり3ハロンタイム
    passing             TEXT            -- 通過順（"2-2-3-3" 等の生文字列）
);

-- 馬マスタ
-- name は nullable。race_result HTML から取得し COALESCE upsert する
-- 新規 horse に限り sex / birth_date / sire / dam を馬詳細・血統ページから取得する
-- 既存 horse（name IS NOT NULL）はスキップし追加フェッチを行わないため、
-- 過去取り込み分の sire/dam は NULL のままになる場合がある
CREATE TABLE horses (
    horse_id    TEXT PRIMARY KEY,
    name        TEXT,
    sex         TEXT,
    birth_date  TEXT,
    sire        TEXT,                   -- 父馬名
    dam         TEXT                    -- 母馬名
);

-- 騎手マスタ
-- name は nullable。race_result HTML から取得し COALESCE upsert する
CREATE TABLE jockeys (
    jockey_id   TEXT PRIMARY KEY,
    name        TEXT
);

-- 調教師マスタ
-- name は nullable。race_result HTML から取得し COALESCE upsert する
CREATE TABLE trainers (
    trainer_id  TEXT PRIMARY KEY,
    name        TEXT
);

-- 払戻詳細
CREATE TABLE payouts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id     TEXT NOT NULL REFERENCES races(race_id),
    bet_type    TEXT NOT NULL,          -- '単勝' | '複勝' | '馬連' 等
    combo       TEXT NOT NULL,          -- 対象馬番組み合わせ（例: "3" / "3-7"）
    amount      INTEGER NOT NULL,       -- 払戻金 (円)
    popularity  INTEGER                 -- 払戻人気
);

-- スクレイピングログ
CREATE TABLE scrape_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    url          TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,         -- ISO 8601
    status       TEXT NOT NULL,         -- 'ok' | 'error' | 'skipped'
    etag         TEXT,
    content_hash TEXT                   -- SHA-256 ハッシュ
);

-- モデル学習履歴
CREATE TABLE model_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,          -- ISO 8601
    model_path  TEXT NOT NULL,          -- data/models/<YYYYMMDD-HHMMSS>/  （ディレクトリパス）
    params_json TEXT,                   -- LightGBM パラメータ JSON
    train_range TEXT,                   -- 学習期間（例: "2022-01-01/2024-01-01"）
    valid_range TEXT,                   -- 検証期間
    metrics_json TEXT,                  -- 評価指標 JSON
    notes       TEXT,
    is_active   INTEGER DEFAULT 0       -- 推論に使用する active モデルフラグ (0/1)
);
```

---

## API エンドポイント仕様

ベース URL: `http://127.0.0.1:${KEIBA_API_PORT}`

バインドは必ず `127.0.0.1` のみ。CORS は Vite dev サーバのオリジンのみ許可する。

### ヘルスチェック

| メソッド | パス | ステータス | 概要 |
|---|---|---|---|
| GET | `/api/health` | 200 | サーバ稼働確認 |

```json
// GET /api/health レスポンス例
{ "status": "ok", "version": "0.1.0" }
```

### レース

| メソッド | パス | ステータス | 概要 |
|---|---|---|---|
| GET | `/api/races/upcoming?days=7` | 200 | 直近 N 日の出馬表一覧 |
| GET | `/api/races/{race_id}` | 200 / 404 | レース詳細（出走馬・オッズ・天候等） |

```json
// GET /api/races/upcoming レスポンス例（抜粋）
{
  "races": [
    {
      "race_id": "202406010101",
      "date": "2024-06-01",
      "course": "東京",
      "surface": "芝",
      "distance": 2400,
      "race_class": "G1",
      "n_runners": 18
    }
  ]
}
```

### 予想

| メソッド | パス | ステータス | 概要 |
|---|---|---|---|
| GET | `/api/predictions/{race_id}` | 200 / 404 / 503 | 全馬の単勝・複勝予想確率 |

- active モデルが存在しない場合は **503** を返す
- `top_features` は `predict_race_with_shap()` が TreeExplainer で各馬の上位 3 特徴量列名を返す

```json
// GET /api/predictions/{race_id} レスポンス例（抜粋）
{
  "race_id": "202406010101",
  "model_id": 3,
  "predictions": [
    {
      "horse_id": "2019100001",
      "post_position": 5,
      "win_prob": 0.183,
      "place_prob": 0.452,
      "rank_score": 2.41,
      "top_features": ["odds_win", "recent_avg_finish", "jockey_recent_win_rate"]
    }
  ]
}
```

### モデル管理

| メソッド | パス | ステータス | 概要 |
|---|---|---|---|
| GET | `/api/models` | 200 | 学習済みモデル一覧 |
| GET | `/api/models/{id}` | 200 / 404 | モデル詳細（パラメータ・評価指標） |
| POST | `/api/models/train` | 202 | 再学習ジョブをバックグラウンド起動（即時 JobAccepted 返却） |
| POST | `/api/models/{id}/activate` | 200 / 404 | 指定モデルを active に設定 |

非同期ジョブ（`POST /api/models/train`）は `asyncio.create_task` でバックグラウンド起動し、以下を即時返却する。

```json
// POST /api/models/train レスポンス例（202 Accepted）
{ "job_id": "train-20260428-120000", "status": "accepted" }
```

ジョブの進捗状態は JobRegistry がインメモリで管理する。プロセス再起動でジョブ状態は消失する。

### メトリクス

| メソッド | パス | ステータス | 概要 |
|---|---|---|---|
| GET | `/api/metrics/summary` | 200 | モデル評価指標サマリ |
| GET | `/api/metrics/timeseries` | 200 | 時系列メトリクス（グラフ用） |

**`GET /api/metrics/summary` のフォールバック仕様**: `model_runs.metrics_json` の `valid_ndcg*` が NaN の場合（`--valid-months 0` で学習した場合など）、`test_ndcg*` に自動フォールバックして返す。`top1_hit` / `place_hit` / `payback_win` / `payback_place` / `n_races` は `train.py` が出力しないため、`evaluate.py --persist` を実行して `metrics_json` に保存するまで null（Dashboard 表示: 「—」）になる。

### スクレイパー管理

| メソッド | パス | ステータス | 概要 |
|---|---|---|---|
| GET | `/api/scraper/status` | 200 | スクレイパー稼働状況・最終取得日・未取得日数 |
| GET | `/api/scraper/recent_activity?minutes=N` | 200 | scrape_log 直近 N 分の集計（status 内訳・rate_per_min・最新 race_id） |
| POST | `/api/scraper/run` | 202 | 手動スクレイピング実行（非同期・JobAccepted 即時返却） |
| POST | `/api/scraper/stop` | 200 | スクレイピング即時停止（緊急停止スイッチ） |

`ScraperStatus.missing_dates_count`: `?range=N` クエリで日数を指定（デフォルト 30 日）し、ok ログ 0 件の日数を返す。カレンダー参照ベースではなく簡素な日数カウント実装。

`ScraperRecentActivity`（`GET /api/scraper/recent_activity?minutes=N`）: scrape_log を直近 N 分でフィルタし、ok / error / skipped の件数内訳、1 分あたりフェッチ数（rate_per_min）、最新 race_id を返す。CLI ingest 実行中も UI から進捗をリアルタイムに確認するための用途。

`POST /api/scraper/run` も `POST /api/models/train` と同様に JobAccepted（202）を即時返却し、バックグラウンドで実行する。

### ジョブ管理

| メソッド | パス | ステータス | 概要 |
|---|---|---|---|
| GET | `/api/jobs` | 200 | ジョブ一覧（インメモリ管理） |
| GET | `/api/jobs/{job_id}` | 200 / 404 | 指定ジョブの状態取得 |

ジョブ状態は JobInfoSchema（`job_id` / `status` / `created_at` / `updated_at` / `detail`）で返却する。ジョブ情報はインメモリ管理のためプロセス再起動で消失する。

### 設定

| メソッド | パス | ステータス | 概要 |
|---|---|---|---|
| GET | `/api/settings` | 200 | 現在の設定値取得 |
| PUT | `/api/settings` | 200 | 設定値更新（User-Agent・レート制御値・ベットルール閾値等） |

設定値は `data/settings.json` に永続化される（`core/settings_store.py`）。

---

## セキュリティ

- バインドは `127.0.0.1` のみ（外部アクセス不可）
- 認証なし（ローカル単体起動前提）
- CORS 許可オリジン:
  - `http://localhost:5173` / `http://127.0.0.1:5173`（Vite dev サーバ）
  - 環境変数 `KEIBA_CORS_EXTRA` にカンマ区切りで追加可能

---

## 環境変数

| 変数名 | デフォルト値 | 適用対象 | 概要 |
|---|---|---|---|
| `KEIBA_API_PORT` | `8765` | バックエンド | FastAPI バインドポート。任意のポートを手動指定してよい |
| `KEIBA_CORS_EXTRA` | （なし） | バックエンド | 追加 CORS 許可オリジン（カンマ区切り） |
| `KEIBA_DATA_DIR` | `backend/data/` | バックエンド | DB・モデル・settings.json の保存ルートディレクトリ |
| `KEIBA_KEEP_MISC_CACHE` | `0` | バックエンド（ingest_range） | `1` に設定すると `ingest_range` が各日完了後の `data/raw/misc/` 自動削除をスキップする。デバッグ用 opt-out フラグ |
| `KEIBA_INCLUDE_NAR` | `0` | バックエンド（ingest） | `1` に設定すると地方競馬（NAR）のレース ID も ingest 対象に含める。デフォルトは中央（JRA）のみ |
| `VITE_KEIBA_API_BASE_URL` | `http://127.0.0.1:8765` | フロントエンド | `src/lib/api-base.ts` の `getApiBaseUrl()` が返すベース URL を上書き |

---

## ビルド手順

### ローカル開発

ブラウザ確認用 dev サーバ（uvicorn + Vite）を一発起動:

```bash
bash scripts/dev.sh
# → http://localhost:5173 (Vite) / http://127.0.0.1:8765 (FastAPI)
# Ctrl-C で全プロセス停止
```

`scripts/dev.sh` は実行のたびに `uv sync` / `alembic upgrade head` / `pnpm install` を行うため、PR 取り込み直後でも追加コマンド不要でこれ一本で動く。

個別に起動する場合:

```bash
# バックエンド
cd backend
uv sync
uv run uvicorn keiba_ai.main:app --host 127.0.0.1 --port 8765 --reload

# フロント（別ターミナル）
cd frontend
pnpm install
pnpm dev
```

---

## 開発環境の前提

| ツール | バージョン目安 | 備考 |
|---|---|---|
| Python | 3.12 以上 | |
| uv | 0.4 以上 | `uv sync` / `uv run keiba-ingest` が動作すること |
| FastAPI / uvicorn | pyproject.toml 経由で導入 | `uv run uvicorn keiba_ai.main:app --port 8765` で起動確認 |
| LightGBM | 4.x 以上 | `uv sync` で自動導入 |
| Alembic | pyproject.toml 経由で導入 | `uv run alembic upgrade head` で動作確認 |
| Node.js | 20 LTS 以上 | フロントエンド実装に必要 |
| pnpm | 9.x 以上 | `pnpm test`・`pnpm build`・`pnpm lint` が通ること |
| Optuna | `uv sync` で自動導入 | `python -m keiba_ai.ai.tune` で動作確認 |

### AI 学習・評価 CLI

```bash
cd backend

# モデル学習（DB から全データを読み込み、時系列分割して学習）
uv run python -m keiba_ai.ai.train

# 学習終了日を指定（学習データの上限を固定する）
uv run python -m keiba_ai.ai.train --train-end 2025-12-31

# バックテスト評価（学習済みモデルディレクトリを指定）
uv run python -m keiba_ai.ai.evaluate --model data/models/20260101-120000

# 評価結果を model_runs.metrics_json にマージ保存する（Dashboard MetricCard に反映させる場合は必須）
uv run python -m keiba_ai.ai.evaluate --model data/models/20260101-120000 --persist

# 評価期間を絞る
uv run python -m keiba_ai.ai.evaluate --model data/models/20260101-120000 \
    --start 2025-06-01 --end 2025-12-31

# 1 番人気常時投票ベースラインとの比較（{model, baseline_favorite, delta} を出力）
uv run python -m keiba_ai.ai.evaluate --model data/models/20260101-120000 \
    --baseline favorite
```
