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
| AI / ML | PyTorch 2.x + Lightning（NN, optional extra）, pandas 2.x, numpy 1.26 以上, scikit-learn 1.4 以上 |
| スクレイピング | httpx（非同期 HTTP）, BeautifulSoup4 |

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
│   │   ├── main.py        # FastAPI app エントリポイント・Uvicorn 起動
│   │   ├── api/           # ルーター群（races, predictions, models, scraper, settings 等）
│   │   ├── core/          # 設定（Settings）・ロギング・DB セッション管理
│   │   ├── db/            # SQLAlchemy モデル定義・Alembic マイグレーション
│   │   ├── scraper/       # netkeiba スクレイパー実装
│   │   │   └── parsers/
│   │   │       ├── race_calendar.py   # 開催日カレンダー
│   │   │       ├── race_result.py     # レース結果
│   │   │       ├── payout.py          # 払戻金
│   │   │       ├── horse_detail.py    # 馬詳細 (name/sex/birth_date)
│   │   │       └── horse_pedigree.py  # 馬血統 (sire/dam)
│   │   ├── features/      # 特徴量エンジニアリング
│   │   │   ├── builder.py         # FEATURE_COLUMNS (46 列) 定義・build_training_frame / build_inference_frame（レース単位バッチ処理）
│   │   │   ├── history_sequence.py # 履歴 GRU 用の過去走トークン列生成
│   │   │   ├── race_info.py       # レース単位の情報量判定（新馬戦など「履歴が無いレース」の検出）
│   │   │   └── extractors/        # ドメイン別抽出器
│   │   │       ├── course.py          # レース・馬番・馬体重系特徴量
│   │   │       ├── horse_history.py   # 馬の過去成績（直近平均着順・上がり3F・脚質 等）
│   │   │       ├── jockey.py          # 騎手成績統計
│   │   │       ├── odds.py            # オッズ・人気系特徴量
│   │   │       ├── pedigree.py        # 血統特徴量（父/母の産駒勝率）
│   │   │       ├── relative_features.py # 同レース内相対特徴量（馬体重 percentile 等）
│   │   │       └── trainer.py         # 調教師成績統計
│   │   ├── ai/            # NN 学習・推論・評価（依存 DAG の層で機能サブパッケージ化）
│   │   │   ├── core/       # types / labels / splits / temperature / probabilities（最下層）
│   │   │   ├── model/      # registry / _artifacts_nn + NN 実装 (net / loss / dataset / preprocess)
│   │   │   ├── training/   # train_nn
│   │   │   ├── inference/  # predict（bundle-aware 推論）
│   │   │   ├── betting/    # odds / strategy
│   │   │   ├── simulation/ # engine / persistence
│   │   │   └── evaluation/ # backtest
│   │   └── jobs/          # 取り込み・運用 CLI（ingest / ingest_range / ingest_odds / backup_db 等）
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
│   │   ├── favicon.svg        # ブランド favicon（馬蹄モノグラム・タイル版、HSL 直書き）
│   │   └── logo.svg           # アプリ外（README / 資料）用の単体グリフ。アプリ内は BrandMark.tsx
│   ├── src/
│   │   ├── main.tsx           # React + QueryClient + Router マウント
│   │   ├── App.tsx            # Outlet レイアウト（Topbar 含む）
│   │   ├── router.tsx         # createBrowserRouter（7 画面 + 旧 URL の Navigate リダイレクト 3 本）
│   │   ├── globals.css        # Tailwind ベース + CSS 変数（デザイントークン）
│   │   ├── routes/            # ページコンポーネント
│   │   │   ├── Dashboard.tsx        # ActiveModelCard + MetricBand + AccuracyChart
│   │   │   ├── Races.tsx            # RaceCalendar + DayIngestPanel（旧 UpcomingRaces / PastRaces / Ingest を統合）
│   │   │   ├── RaceDetail.tsx       # レース概要 + 出走馬表 + 推奨買目 + 答え合わせ
│   │   │   ├── Ledger.tsx           # 購入記録と収支（回収率・的中率・損益推移）
│   │   │   ├── Models.tsx           # ActiveModelCard + ModelTable + Activate + TrainModelDialog
│   │   │   ├── ModelDetail.tsx      # モデル 1 件の詳細 + ModelSimulationPanel
│   │   │   └── Settings.tsx         # react-hook-form + zod バリデーション
│   │   ├── components/        # 共通コンポーネント
│   │   │   ├── Topbar.tsx           # 上部ナビ（全画面共通）。等幅英字のみ
│   │   │   ├── BrandMark.tsx        # ブランドマーク（馬蹄）。inline SVG + currentColor でテーマ追従
│   │   │   ├── PageHeader.tsx       # ページ見出し共通コンポーネント
│   │   │   ├── MetricBand.tsx       # KPI 帯（罫線区切り。旧 MetricCard を置換）
│   │   │   ├── AccuracyChart.tsx    # 精度推移グラフ（Recharts）
│   │   │   ├── RaceCalendar.tsx     # 月カレンダー。日ごとの取込状況を色で示す
│   │   │   ├── DayIngestPanel.tsx   # 選択日の取込操作（過去=結果 / 当日=両方 / 未来=出馬表）
│   │   │   ├── DataCoverageBand.tsx # データ取込のカバレッジ表示
│   │   │   ├── Umaban.tsx           # 馬番チップ（枠色）
│   │   │   ├── RecommendationsCard.tsx / RecommendationParamsBar.tsx  # 推奨買目と、その条件（予算 / 1点 / 券種 / 狙い方）
│   │   │   ├── ModelSimulationPanel.tsx # 期間・予算・戦略を選んでバックテストを回す
│   │   │   ├── BankrollChart.tsx    # シミュレーションの資産推移
│   │   │   ├── AddBetDialog.tsx     # 購入記録の手動登録
│   │   │   ├── EmptyState.tsx / JobProgressCard.tsx / ScraperStatusCard.tsx
│   │   │   ├── ActiveModelCard.tsx / ModelTable.tsx
│   │   │   ├── SettingsForm.tsx     # 設定フォーム（Section / FieldRow ヘルパ）
│   │   │   ├── TrainModelDialog.tsx / IngestRunDialog.tsx / RunResultsDialog.tsx
│   │   │   ├── DeleteModelDialog.tsx / EditModelNameDialog.tsx / DateYMDPicker.tsx
│   │   │   ├── PredictionTable.tsx  # ※未使用（RaceDetail が独自の表を持つ）
│   │   │   └── ui/                  # shadcn 手書きコンポーネント
│   │   │       ├── button.tsx / card.tsx / table.tsx
│   │   │       ├── tabs.tsx / badge.tsx / skeleton.tsx
│   │   │       ├── dialog.tsx / form.tsx / input.tsx
│   │   │       ├── label.tsx / select.tsx
│   │   │       ├── toast.tsx / toaster.tsx  # sonner ラッパ
│   │   ├── hooks/             # カスタムフック（TanStack Query ラッパ）
│   │   │   ├── useRacesCalendar.ts / useRacesByDate.ts / useRaceDetail.ts / useThisWeekendRaces.ts
│   │   │   ├── usePredictions.ts    # 予想は重いのでボタン主導（enabled で gate）
│   │   │   ├── useRecommendations.ts
│   │   │   ├── useMetricsSummary.ts / useMetricsTimeseries.ts
│   │   │   ├── useModels.ts / useActivateModel.ts / useTrainModel.ts / useUpdateModel.ts / useDeleteModel.ts
│   │   │   ├── useBetList.ts / useBetSummary.ts / useBetBreakdown.ts / useBetTimeseries.ts
│   │   │   ├── useCreateBet.ts / useCreateBetsBulk.ts / useDeleteBets.ts
│   │   │   ├── useScraperStatus.ts / useScraperRun.ts / useScraperStop.ts / useScraperRecentActivity.ts
│   │   │   ├── useRunShutuba.ts / useRunResults.ts
│   │   │   ├── useJobStatus.ts      # jobId を 2 秒 polling、terminal status で停止
│   │   │   ├── useSettings.ts / useTheme.ts
│   │   ├── store/             # Zustand ストア
│   │   │   └── app.ts         # useAppStore / useScraperStore（trackedJobId を含む）
│   │   ├── lib/               # API クライアント・ユーティリティ
│   │   │   ├── api.ts         # ky ベース API クライアント。error helpers（getStatus / isNotFoundError / formatErrorMessage 等）を含む
│   │   │   ├── api-base.ts    # getApiBaseUrl()
│   │   │   ├── formatters.ts  # display formatter 集約（null/NaN/Infinity を「—」に統一）
│   │   │   ├── betTypes.ts    # 選択できる馬券種（枠連は AI が買い目を生成しないので含めない）
│   │   │   ├── betCombos.ts / labels.ts / waku.ts  # 買い目の組立 / 表示ラベル / 枠番と枠色
│   │   │   ├── query-client.ts
│   │   │   └── cn.ts          # clsx + tailwind-merge ユーティリティ
│   │   └── types/
│   │       └── api.ts         # API レスポンス型定義
│   └── src/__tests__/         # Vitest + @testing-library/react
│       ├── App.test.tsx / Dashboard.test.tsx / Races.test.tsx / RaceDetail.test.tsx
│       ├── Ledger.test.tsx / Models.test.tsx / Settings.test.tsx
│       ├── DayIngestPanel.test.tsx / RecommendationsCard.test.tsx
│       ├── betCombos.test.ts / waku.test.ts
│       ├── lib_api_errors.test.ts  # error helpers 単体テスト（9 ケース）
│       ├── lib_formatters.test.ts  # formatters.ts の全関数ユニットテスト
│       └── setup.ts
│
├── data/                      # ローカルデータ（.gitignore 対象）
│   ├── raw/                   # HTML キャッシュ（<yyyy>/<mm>/<race_id>.html）
│   ├── keiba.db               # SQLite DB 本体
│   ├── odds.db                # 確定オッズ DB（race_odds テーブル、keiba.db と分離）
│   └── models/                # 学習済みモデル（<YYYYMMDDTHHMMSS>-nn/{model.pt, meta.json, ...}）
│
└── scripts/                   # 運用スクリプト
    └── dev.sh                 # uv sync + alembic + pnpm install + uvicorn + Vite を一発起動
```

---

## DB スキーマ

SQLite を使用する。ORM は SQLAlchemy 2.x DeclarativeBase + naming_convention で実装し、DB 初期化は `alembic upgrade head` で行う。マイグレーションファイルは `migrations/versions/` に格納されており、現在 12 ファイル（`0001`〜`0012`）が定義されている。

| ファイル | revision | 内容 |
|---|---|---|
| `0001_initial.py` | 0001 | 初期スキーマ（8 テーブル作成） |
| `0002_add_agari_passing.py` | 0002 | entries テーブルに `agari_3f` / `passing` 列を追加 |
| `0003_add_scrape_log_fetched_at_index.py` | 0003 | `scrape_log.fetched_at` に単一カラムインデックスを追加（`recent_activity` エンドポイントの full scan 防止） |
| `0004_add_bet_records.py` | 0004 | `bet_records` テーブル追加（ベット記録台帳） |
| `0005_add_race_name.py` | 0005 | races テーブルに `name` 列（レース名）を追加 |
| `0006_add_live_odds.py` | 0006 | `live_odds` テーブル追加（後に 0010 で廃止） |
| `0007_add_simulation_runs.py` | 0007 | `simulation_runs` テーブル追加（バックテスト履歴） |
| `0008_add_model_type.py` | 0008 | model_runs テーブルに `model_type` 列を追加 |
| `0009_add_simulation_model_run_id.py` | 0009 | simulation_runs に `model_run_id` 列を追加 |
| `0010_drop_live_odds.py` | 0010 | `live_odds` テーブル廃止（オッズは odds.db に分離） |
| `0011_model_type_default_nn.py` | 0011 | `model_type` のデフォルトを `"nn"` に変更（NN 専用化） |
| `0012_add_horses_sire_dam_index.py` | 0012 | horses の `sire` / `dam` にインデックス追加（血統特徴量の集計高速化） |

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

keiba.db の 10 テーブル全て ORM 化されている（`races` / `horses` / `jockeys` / `trainers` / `entries` / `payouts` / `scrape_log` / `model_runs` / `bet_records` / `simulation_runs`）。このほか確定オッズは別ファイル `data/odds.db` の `race_odds` テーブル（`db/odds_db.py`。Alembic 管理外・`is_confirmed` 列でライブ snapshot と確定値を区別）に保持する。以下は主要テーブルのスキーマ抜粋。

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
    model_path  TEXT NOT NULL,          -- data/models/<YYYYMMDDTHHMMSS>-nn/  （ディレクトリパス）
    params_json TEXT,                   -- 学習ハイパーパラメータ JSON
    train_range TEXT,                   -- 学習期間（例: "2022-01-01/2024-01-01"）
    valid_range TEXT,                   -- 検証期間
    metrics_json TEXT,                  -- 評価指標 JSON
    notes       TEXT,
    is_active   INTEGER DEFAULT 0,      -- 推論に使用する active モデルフラグ (0/1)
    model_type  TEXT NOT NULL DEFAULT 'nn'  -- 常に "nn"（NN 専用化済み。履歴互換のため残置）
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
| GET | `/api/races/calendar?year=&month=` | 200 | 月カレンダー。日ごとの開催有無と取込状況 |
| GET | `/api/races/by_date?date=` | 200 | 指定日のレース一覧 |
| GET | `/api/races/coverage` | 200 | 取込済みデータの期間カバレッジ |
| GET | `/api/races/this_weekend` | 200 | 今週末のレース一覧 |
| GET | `/api/races/upcoming?days=7` | 200 | 直近 N 日の出馬表一覧 |
| GET | `/api/races/recent?days=7` | 200 | 直近 N 日の結果確定済みレース一覧 |
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
| GET | `/api/predictions/bulk?race_ids=` | 200 / 503 | 複数レース分をまとめて予想（一覧画面用） |

- active モデルが存在しない場合は **503** を返す
- `top_features` は特徴量寄与表示用フィールドだが、寄与計算は廃止済みのため常に空配列を返す（API 互換のため残置）
- `info_coverage` は「このレースにどれだけ判断材料があるか」（`features/race_info.py`）。
  出走馬の過去走ゼロ率が 0.5 以上なら `is_low_information: true` になり、UI は
  新馬戦などで「情報が少ない」注意書きを出す

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
| PATCH | `/api/models/{id}` | 200 / 404 | 表示名の変更 |
| DELETE | `/api/models/{id}` | 200 / 404 | モデル削除（active は削除不可） |
| POST | `/api/models/compact` | 200 | 不要なモデル成果物を削除して容量を回収 |

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

**`GET /api/metrics/summary` の指標ソース**: `backtest --persist` が走っていれば、4 指標すべてを **同じ 1 回の評価**（同じレース集合・アプリと同じ賭けルール）から取る。混ぜると「valid の NDCG と test の回収率」のように出所の違う数字が 1 枚のカードに並ぶため、`ndcg*` も backtest 側を優先する。

backtest 未実行のときだけ学習時の指標に fallback するが、**fallback 先は同じ量ではない**ので注意:

| API のキー | backtest（優先） | 学習時 fallback | 一致するか |
|---|---|---|---|
| `top1_hit` | 予想1位が1着 | `test_tansho_hit` | ほぼ同義 |
| `place_hit` | **上位3頭のうち1頭以上**が3着以内 | `test_fukusho_hit`（予想1位が3着以内）| **別物**（実測 0.885 vs 0.503）|
| `payback_win` | 予想1位に1点定額（`--win-bet-rule top1` が既定）| `test_tansho_roi`（top-1 に賭け続ける）| ほぼ同義（差はオッズ下限とレース集合）|

フロントのヒットは backtest 側の定義で書いてあるので、fallback 中の値はラベルとずれる。正確を期すなら `--persist` を回すこと。

### スクレイパー管理

| メソッド | パス | ステータス | 概要 |
|---|---|---|---|
| GET | `/api/scraper/status` | 200 | スクレイパー稼働状況・最終取得日・未取得日数 |
| GET | `/api/scraper/recent_activity?minutes=N` | 200 | scrape_log 直近 N 分の集計（status 内訳・rate_per_min・最新 race_id） |
| POST | `/api/scraper/run` | 202 | 手動スクレイピング実行（非同期・JobAccepted 即時返却） |
| POST | `/api/scraper/run_shutuba` | 202 | 指定日の出馬表だけを取り込む（未来日・当日） |
| POST | `/api/scraper/run_results` | 202 | 指定日の結果・払戻だけを取り込む（過去日・当日） |
| GET | `/api/scraper/discover_today_race_ids` | 200 | 当日開催の race_id を列挙（取込対象の事前確認） |
| GET | `/api/scraper/discover_this_weekend_race_ids` | 200 | 今週末開催の race_id を列挙 |
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

### 推奨買目

| メソッド | パス | ステータス | 概要 |
|---|---|---|---|
| GET | `/api/recommendations/{race_id}` | 200 / 404 / 503 | 予算内に収まる買い目一覧 |

主なクエリ: `top_n_horses`（連系の候補にする上位頭数）/ `top_k`（券種ごとの上限点数）/
`race_budget` / `stake_unit` / `bet_types`（カンマ区切り）。未指定は Settings の値を使う。

買い方は券種で異なる（`ai/betting/strategy.py`、根拠は `docs/ai-model.md`）:

- **単勝**: モデル 1 位の 1 頭のみ。EV 条件は使わず、オッズ下限 `win_min_odds` だけ見る
- **複勝**: モデル 1 位の 1 頭のみ。EV 条件は使わない
- **連系**（馬連 / ワイド / 馬単 / 三連複 / 三連単）: `win_ev_threshold` を超える組み合わせ

賭け金は EV 順ではなく **単勝 → 複勝 → 連系** の順に、同券種内は的中確率順で
`stake_units`（券種別の 1 点あたり金額）を割り当て、`race_budget` を超えたら打ち切る。

### シミュレーション

| メソッド | パス | ステータス | 概要 |
|---|---|---|---|
| POST | `/api/simulation/start` | 202 | バックテストをバックグラウンド起動（`job.result.run_id` に結果 id）|
| GET | `/api/simulation/active_model` | 200 / 503 | active モデルで同期実行（短い window 用）|
| GET | `/api/simulation/runs` | 200 | 保存済み run 一覧 |
| GET | `/api/simulation/runs/{run_id}` | 200 / 404 | run 詳細（資産推移・券種別内訳）|
| DELETE | `/api/simulation/runs/{run_id}` | 200 / 404 | run 削除 |

主なクエリ: `start` / `end` / `budget` / `strategy`（conservative / balanced / aggressive / selective）/
`model_id` / `max_stake_per_race_yen` / `exclude_low_information`。
`exclude_low_information=true` で、出走馬全員が初出走のレース（新馬戦など）を集計から外す。
買い方・賭け金配分は推奨買目 API と同じ経路（`ai/simulation/engine.py` → `strategy.py`）を通る。

### 購入記録（Ledger）

| メソッド | パス | ステータス | 概要 |
|---|---|---|---|
| GET | `/api/bets` | 200 | 購入記録一覧 |
| POST | `/api/bets` | 200 | 1 件登録 |
| POST | `/api/bets/bulk` | 200 | 推奨買目からまとめて登録 |
| POST | `/api/bets/bulk_delete` | 200 | まとめて削除 |
| GET / PUT / DELETE | `/api/bets/{bet_id}` | 200 / 404 | 1 件の取得・更新・削除 |
| GET | `/api/bets/summary` | 200 | 投資額・払戻・回収率・的中率 |
| GET | `/api/bets/breakdown` | 200 | 券種別の内訳 |
| GET | `/api/bets/timeseries` | 200 | 損益推移（グラフ用）|
| GET | `/api/bets/export.csv` | 200 | CSV 書き出し |

### 設定

| メソッド | パス | ステータス | 概要 |
|---|---|---|---|
| GET | `/api/settings` | 200 | 現在の設定値取得 |
| PUT | `/api/settings` | 200 | 設定値更新（User-Agent・レート制御値・ベットルール閾値等） |

設定値は `data/settings.json` に永続化される（`core/settings_store.py`）。主なキー:

| キー | 概要 | デフォルト |
|---|---|---|
| `user_agent` | スクレイパーの User-Agent | 研究用 UA 文字列 |
| `rate_min_seconds` / `rate_max_seconds` | リクエスト間隔（秒） | 3.0 / 6.0 |
| `night_min_seconds` | 深夜帯の最小間隔（秒） | 5.0 |
| `scraper_stopped` | 緊急停止フラグ | `false` |
| `win_ev_threshold` | **連系のみ**の EV 閾値 | 1.1 |
| `win_min_odds` | 単勝で買うオッズ下限（EV 条件の代わり） | 1.1 |
| `race_budget` | 1 レースに使う上限（円） | 5000 |
| `stake_unit` | 1 点あたりの既定額（円） | 100 |
| `stake_units` | 券種別の 1 点あたり（円） | 単勝 500 / 複勝 500 / 連系 100 |
| `enabled_bet_types` | 対象券種 | 単勝・複勝・馬連・ワイド・馬単・三連複・三連単 |

- 枠連は AI が買い目を生成しないので選択肢に出さない（`core/bet_types.py` の
  `supported_bet_types()` が保存済み設定からも落とす）
- 複勝の EV 閾値（旧 `place_ev_threshold`）は**廃止**。複勝は EV 条件を使わない
- 旧 Kelly 設定（`bankroll` / `kelly_fraction` / `max_stake_per_race_pct`）は読み込み時に
  `race_budget` へ読み替えて破棄する（`_migrate_legacy`）

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
| `KEIBA_ODDS_DB` | `<data>/odds.db` | バックエンド | オッズ DB のパス上書き |
| `KEIBA_LOG_DIR` | （なし） | バックエンド | 設定するとファイルログを `<dir>/<name>-<ts>.log` に出す（opt-in）|
| `KEIBA_SCRAPER_STOP` | `0` | バックエンド（scraper） | `1` でスクレイピングを停止（UI / API と並ぶ 3 経路目）|
| `KEIBA_USER_AGENT` / `KEIBA_RATE_MIN_SECONDS` / `KEIBA_RATE_MAX_SECONDS` / `KEIBA_NIGHT_MIN_SECONDS` | settings.json の値 | バックエンド（scraper） | スクレイパー設定の環境変数上書き |
| `KEIBA_TLS_RELAX_STRICT` | `0` | バックエンド（scraper） | TLS 検証の緩和を明示的に opt-in（既定は厳格）|
| `KEIBA_PLACE_PROB_METHOD` | `plackett_luce` | バックエンド（推論） | 複勝確率の算出方式 |
| `KEIBA_DISABLE_FRAME_CACHE` | `0` | バックエンド（特徴量） | `1` で特徴量フレームのキャッシュを無効化 |
| `KEIBA_DEBUG_SIM_MISSES` | `0` | バックエンド（シミュレーション） | `1` で外れ買い目の内訳をログ出力 |
| `KEIBA_EXCLUDE_ODDS_FEATURES` | `0` | バックエンド（学習・推論） | `1` でオッズ系特徴量を除外（オッズ未確定時の検証用）|
| `KEIBA_MISSING_INDICATORS` / `KEIBA_LOG_FEATURES` / `KEIBA_SPEED_FIGURE` / `KEIBA_PACE_FEATURES` | `0` | バックエンド（学習・推論） | 実験用の特徴量ノブ。**すべて default-off**（A/B で本番 ROI 改善せず。`docs/ai-model.md`「実験ノブと A/B 知見」）。有効化するときは学習と推論で必ず揃える |
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
uv run uvicorn main:app --host 127.0.0.1 --port 8765 --reload

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
| FastAPI / uvicorn | pyproject.toml 経由で導入 | `uv run uvicorn main:app --port 8765` で起動確認 |
| PyTorch (extra `nn`) | 2.x 以上 | `uv sync --extra nn` で導入 |
| Alembic | pyproject.toml 経由で導入 | `uv run alembic upgrade head` で動作確認 |
| Node.js | 20 LTS 以上 | フロントエンド実装に必要 |
| pnpm | 9.x 以上 | `pnpm test`・`pnpm build`・`pnpm lint` が通ること |

### AI 学習・評価 CLI

```bash
cd backend

# モデル学習（DB から全データを読み込み、時系列分割して学習）
uv run python -m ai.training.train_nn --loss multi --monitor valid_tansho_roi

# 分割の基準日を指定する（**学習終了日ではない**。学習が終わるのは
#   基準日 - test_months - valid_months。詳細は docs/ai-model.md「時系列分割」）
uv run python -m ai.training.train_nn --loss multi --train-end 2025-12-31

# バックテスト評価（学習済みモデルディレクトリを指定）
uv run python -m ai.evaluation.backtest --model data/models/20260101T120000-nn

# 評価結果を model_runs.metrics_json にマージ保存する（Dashboard の KPI に反映させる場合は必須）
uv run python -m ai.evaluation.backtest --model data/models/20260101T120000-nn --persist

# 評価期間を絞る
uv run python -m ai.evaluation.backtest --model data/models/20260101T120000-nn \
    --start 2025-06-01 --end 2025-12-31

# 1 番人気常時投票ベースラインとの比較（{model, baseline_favorite, delta} を出力）
uv run python -m ai.evaluation.backtest --model data/models/20260101T120000-nn \
    --baseline favorite

# 買い方を変えて評価する（既定はアプリと同じ「本命 1 点」ルール）
uv run python -m ai.evaluation.backtest --model data/models/20260101T120000-nn \
    --win-bet-rule top1 --place-bet-rule topk --place-top-k 1
```

最良構成は **二段階**（`plackett_luce` で事前学習 → `--init-from <model_dir>` で
`multi` に fine-tune）。損失・監視指標の選び方は `docs/ai-model.md` を参照。
