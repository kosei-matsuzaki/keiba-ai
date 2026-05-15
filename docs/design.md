# KEIBA AI — 設計方針書

関連ドキュメント: [spec.md](spec.md) / [data-pipeline.md](data-pipeline.md) / [ai-model.md](ai-model.md) / [operations.md](operations.md)

---

## 設計の出発点

### ユーザー要件

- 今週末の出走予定レースを一覧表示し、馬ごとの単勝・複勝予想確率を確認したい
- どの特徴量がその予想に寄与しているか（SHAP）を見たい
- モデルの精度推移（NDCG@3・Top-1 ヒット率・複勝的中率・ROI）を時系列グラフで把握したい
- 手元の PC 上で完結して動作し、外部サービスへデータを送出しない

### 非機能要件

| 項目 | 方針 |
|---|---|
| 配布形態 | Windows 単一 EXE。追加インストール不要 |
| データプライバシー | 全データをローカル保持。クラウド同期なし |
| 停止容易性 | スクレイピングを任意のタイミングで即時停止できるスイッチを設ける |
| レート制御 | netkeiba への最低 3 秒間隔（詳細は [data-pipeline.md](data-pipeline.md)）|
| 拡張性 | モデルの差し替え・特徴量追加が最小変更で可能な設計 |
| 保守性 | バックエンド・フロントエンド・AI モジュールの責務を明確に分離 |

---

## アーキテクチャ図

```text
                  ┌──────────────┐
                  │   ブラウザ   │
                  │ (Vite dev :5173) │
                  └──────┬───────┘
                         │  HTTP → http://127.0.0.1:8765/api/*
                         ▼
                ┌────────────────┐
                │   FastAPI      │
                │   backend      │ uvicorn :8765
                │   (Python)     │
                └──┬──┬──┬───────┘
                   │  │  │
       ┌───────────┘  │  └──────────┐
       │              │             │
┌──────▼──────┐ ┌─────▼──────┐ ┌───▼──────────┐
│  Scraper    │ │  AI 推論   │ │  SQLite DB   │
│  (netkeiba) │ │ (LightGBM) │ │  keiba.db    │
└─────────────┘ └────────────┘ └──────────────┘
                                     ▲
                               モデルファイル
                            (data/models/<run-id>/)
```

`scripts/dev.sh` で FastAPI (uvicorn :8765) と React 管理画面 (Vite :5173) を並列起動し、ブラウザでアクセスする。外部へのネットワーク通信はスクレイパーのみ。

---

## AI モジュール設計

各モジュールの責務を明確に分離し、独立してテスト・置き換えができるようにする。

```text
backend/src/keiba_ai/
├── main.py       FastAPI app factory (create_app) + lifespan + CORS + uvicorn __main__
├── scraper/      スクレイピング専用。HTML 取得・パース・DB 保存のみ。AI を知らない
├── features/     DB から生データを読み取り、学習・推論用の特徴量 DataFrame を生成
│                 （リーク防止のため「予測時点での情報のみ使用する」制約を徹底管理）
├── ai/           特徴量を受け取り LightGBM の学習・評価・推論を実行。features を知らない
│                 ├── trainer.py   学習・ハイパーパラメータ管理
│                 ├── predictor.py 推論・確率変換（softmax / Plackett-Luce）
│                 └── evaluator.py NDCG@k・ヒット率・ROI 計算
├── core/         設定（Settings）・ロギング・settings_store（JSON 永続化）
├── api/          FastAPI ルーター群（schemas / deps / routers/*）
│                 ビジネスロジックは持たず、上記モジュールを呼ぶだけ
└── jobs/         APScheduler ジョブ（週次スクレイプ・月次再学習）。上記モジュールを呼ぶ
```

### 依存方向

```text
api → jobs → ai / features / scraper → db (SQLAlchemy models)
```

循環依存は禁止。`ai` は `scraper` を直接呼び出さない。

### DI 構成

FastAPI の依存注入（`api/deps.py`）で以下を提供する。

| DI 関数 | 提供するオブジェクト | 概要 |
|---|---|---|
| `get_engine` | `Engine` | SQLAlchemy 同期エンジン |
| `get_session` | `Session` | リクエストスコープの DB セッション（yield / finally で close） |
| `get_settings_store` | `SettingsStore` | `core/settings_store.py` の JSON 永続化オブジェクト |
| `get_job_registry` | `JobRegistry` | バックグラウンドジョブのインメモリ管理オブジェクト |

### JobRegistry の性質

- `asyncio.create_task` でバックグラウンドジョブを起動し、`JobInfo` をインメモリで保持する
- **プロセス再起動でジョブ状態は消失する**（永続化なし）

---

## フロントエンド スタイル設計

### デザイントークン（CSS 変数）

`globals.css` で定義する CSS 変数のうち、shadcn/ui のベーストークン（`--background` / `--foreground` 等）に加えて以下のセマンティックトークンを管理する。

| トークン | 用途 | light 値 | dark 値 |
|---|---|---|---|
| `--success` / `--success-foreground` | 成功・アクティブ状態（emerald 系） | `oklch(0.765 0.177 158)` | `oklch(0.765 0.177 158)` |
| `--warning` / `--warning-foreground` | 警告・進行中状態（amber 系） | `oklch(0.769 0.188 70.08)` | `oklch(0.769 0.188 70.08)` |
| `--info` / `--info-foreground` | 情報・補足状態（sky 系） | `oklch(0.685 0.169 237)` | `oklch(0.685 0.169 237)` |
| `--font-sans` | UI 全体のサンセリフフォントスタック | system-ui ほか OS デフォルト | 同左 |
| `--font-mono` | コード・ID 表示用等幅フォントスタック | ui-monospace ほか OS デフォルト | 同左 |

`tailwind.config.ts` の `theme.extend.colors` に `success` / `warning` / `info`（CSS 変数経由）を登録し、クラス名（例: `bg-success text-success-foreground`）として利用できる。

> フォント変数（`--font-sans` / `--font-mono`）は将来の web font 導入（PR-V-B 予定）に向けた差し替えポイントとして確保しており、現時点では OS デフォルトスタックを初期値とする。

### Badge バリアント一覧

`src/components/ui/badge.tsx` で定義する全バリアント:

| バリアント | 対応トークン | 主な使用箇所 |
|---|---|---|
| default | `--primary` | 汎用ラベル |
| secondary | `--secondary` | サブ情報 |
| destructive | `--destructive` | エラー・停止状態 |
| outline | border のみ | 軽量ラベル |
| success | `--success` | active モデル・完了状態 |
| warning | `--warning` | 実行中・保留状態 |
| info | `--info` | 情報補足 |

ハードコードされた Tailwind カラークラス（`bg-emerald-600 text-white` 等）は使用せず、対応バリアントの `<Badge>` に統一する。

---

## UI 画面構成

### 画面一覧と役割

| # | 画面名 | ルート | 役割 | 対応 API |
|---|---|---|---|---|
| 1 | Dashboard | `/` | モデル評価指標のサマリ（NDCG@3・Top-1 ヒット率・複勝的中率・ROI）と精度推移グラフ | `GET /api/metrics/summary`, `GET /api/metrics/timeseries` |
| 2 | Upcoming Races | `/upcoming` | 今週末の出馬表一覧。RaceCard で各レースを表示 | `GET /api/races/upcoming?days=7` |
| 3 | Race Detail | `/races/:race_id` | レース概要 + 出走馬一覧 + PredictionTable（BUY バッジ付き）+ SHAP 寄与欄 | `GET /api/races/{race_id}`, `GET /api/predictions/{race_id}` |
| 4 | Models | `/models` | 学習履歴テーブル（ModelTable）。active モデルの切り替え（Activate ボタン）と再学習トリガ（TrainModelDialog） | `GET /api/models`, `POST /api/models/train`, `POST /api/models/{id}/activate` |
| 5 | Ingest | `/ingest` | ScraperStatusCard でスクレイパー稼働状況表示。IngestRunDialog で手動実行。即時停止 confirm 付き | `GET /api/scraper/status`, `POST /api/scraper/run`, `POST /api/scraper/stop` |
| 6 | Settings | `/settings` | react-hook-form + zod によるバリデーション付き設定フォーム。率閾値バリデーション（rate_min ≤ rate_max / EV ≥ 1.0）込み | `GET /api/settings`, `PUT /api/settings` |

### 画面遷移図

```text
[Dashboard]
    │
    ├─ "今週のレース" リンク → [Upcoming Races]
    │       └─ レース行クリック → [Race Detail]
    │
    ├─ サイドナビ: [Models]
    ├─ サイドナビ: [Ingest]
    └─ サイドナビ: [Settings]
```

サイドナビバーは全画面共通。React Router でルーティングを管理する。

### 各画面の主要コンポーネント

#### Dashboard

```text
┌─────────────────────────────────────────┐
│  ActiveModelCard（Models ページへの Link付き）
│    active モデルが null の場合は「未設定」バッジ + train ガイド
├─────────────────────────────────────────┤
│  MetricCard × 4（NDCG@3 / Top-1 / 複勝的中率 / ROI）
├─────────────────────────────────────────┤
│  MetricsTimeseriesChart（30日推移、Recharts LineChart）
└─────────────────────────────────────────┘
```

#### Upcoming Races

```text
┌─────────────────────────────────────────┐
│  DateFilter（日付タブ or セレクタ）
├─────────────────────────────────────────┤
│  RaceCard × N                          │
│    レース名 / 競馬場 / 距離 / 馬場     │
│    ProbabilityBar（上位 3 頭）         │
└─────────────────────────────────────────┘
```

#### Race Detail

```text
┌─────────────────────────────────────────┐
│  RaceHeader（コース・距離・天候・頭数）  │
├─────────────────────────────────────────┤
│  PredictionTable（全馬、スコア降順）    │
│    馬 ID / スコア / 単勝 prob /         │
│    複勝 prob / BUY バッジ / SHAP 欄    │
│    BUY 判定: win_prob × odds_win > 1.1 │
│    （→ [ai-model.md](ai-model.md) 参照）│
│    SHAP 欄: top_features（上位 3 列名）│
└─────────────────────────────────────────┘
```

#### Models

```text
┌─────────────────────────────────────────┐
│  ActiveModelCard（linkToModels=false で自リンク回避）
│    active モデルが null の場合は「未設定」バッジ + train ガイド
├─────────────────────────────────────────┤
│  ModelTable（学習済みモデル一覧）        │
│    モデル ID / 学習日時 / 評価指標      │
│    is_active 行: bg-emerald-500/5 ハイライト
│    Activate ボタン（active 切り替え）   │
├─────────────────────────────────────────┤
│  再学習ボタン → TrainModelDialog        │
│    （学習中は disabled、ローカル state）│
└─────────────────────────────────────────┘
```

#### Ingest

```text
┌─────────────────────────────────────────┐
│  ScraperStatusCard                      │
│    稼働状況 / 最終取得日 / ジョブ情報  │
│    ポーリング: 実行中 5 秒 / アイドル 30 秒
├─────────────────────────────────────────┤
│  手動実行ボタン → IngestRunDialog       │
│  即時停止ボタン（confirm ダイアログ付き）│
└─────────────────────────────────────────┘
```

#### Settings

```text
┌─────────────────────────────────────────┐
│  PageHeader（Settings2 アイコン）        │
├─────────────────────────────────────────┤
│  SettingsForm（react-hook-form + zod）  │
│  ├─ Section: スクレイパー               │
│  │    User-Agent / rate_min / rate_max  │
│  │    （rate_min ≤ rate_max バリデーション）
│  ├─ Section: ベッティング期待値         │
│  │    EV 閾値（≥ 1.0 バリデーション）  │
│  └─ Section: 運用                       │
│       scraper_stopped（説明付き         │
│       clickable label スタイル）        │
└─────────────────────────────────────────┘
```

Card ラッパは撤廃し、SettingsForm を直接配置する。各 Section は `icon + title + description` のヘッダを持ち、FieldRow には help text を追加する。

---

## 状態管理

### 基本方針

- **TanStack Query（React Query v5）**: サーバーデータのフェッチ・キャッシュ・再取得を管理。API 呼び出しは `src/hooks/` のカスタムフックに集約
- **Zustand**: ページをまたいで保持する UI 状態のみ管理。サーバーデータは一切持たない
- **sonner**: Toast 通知ライブラリ。`src/components/ui/toast.tsx` / `toaster.tsx` を sonner ラッパとして手書き配置。`main.tsx` で `<Toaster />` をマウント
- **react-hook-form + zod**: SettingsForm / IngestRunDialog / TrainModelDialog の共通フォームバリデーションパターン。`zodResolver` + `mode: 'onChange'` で inline error を表示し、submit ボタンを自動 disable する。ダイアログは `open` のたびに `reset` で初期値を復元する。`src/components/ui/form.tsx` で react-hook-form と shadcn フォームコンポーネントを統合
- API クライアントは `src/lib/api.ts` の `ky` インスタンスに集約し、各フックから呼び出す

### React Query（TanStack Query）

| クエリキー | 対応フック | 対象 API | 更新間隔 |
|---|---|---|---|
| `['races', 'upcoming']` | `useUpcomingRaces` | `GET /api/races/upcoming` | 5 分（staleTime） |
| `['races', raceId]` | `useRaceDetail` | `GET /api/races/{race_id}` | ユーザー操作時のみ（refetch） |
| `['predictions', raceId]` | `usePredictions` | `GET /api/predictions/{race_id}` | ユーザー操作時のみ（refetch） |
| `['metrics', 'summary']` | `useMetricsSummary` | `GET /api/metrics/summary` | 10 分 |
| `['metrics', 'timeseries']` | `useMetricsTimeseries` | `GET /api/metrics/timeseries` | 10 分 |
| `['scraper', 'status']` | `useScraperStatus` | `GET /api/scraper/status` | アイドル: 30 秒 / 実行中: 5 秒（refetchInterval を Zustand `isRunning` で切り替え） |
| `['models']` | `useModels` | `GET /api/models` | ユーザー操作時のみ |
| `['settings']` | `useSettings` | `GET /api/settings` | ユーザー操作時のみ |

### Zustand（`src/store/app.ts`）

| ストア | 保持する状態 |
|---|---|
| `useAppStore` | `sidebarOpen`（サイドナビ開閉状態） |
| `useScraperStore` | `isRunning`（スクレイパー手動実行中フラグ — ポーリング間隔の切り替えに使用） |

### フロント側 API クライアント（`src/lib/api.ts` + `src/lib/api-base.ts`）

- **HTTP ライブラリ**: `ky` 1.x
- **ベース URL 解決**: `src/lib/api-base.ts` の `getApiBaseUrl()` が `VITE_KEIBA_API_BASE_URL` 環境変数 または デフォルト `http://127.0.0.1:8765` を返す
- **lazy 初期化**: `api.ts` の ky インスタンスは最初の API 呼び出し時に初期化される

### テスト戦略

- `vi.mock('../lib/api')` で API モジュール全体を差し替える方式を採用
- MSW + jsdom + ky の組み合わせが不安定だったため MSW は使用しない
- `@testing-library/user-event` を使用（フォームインタラクションテスト用）

---

## 拡張ポイント

| 拡張内容 | 設計上の配慮 |
|---|---|
| DL アンサンブル（TabNet / CatBoost 等との ensemble） | `ai/predictor.py` がモデルの種別に依存しない抽象インターフェースを持つ |
| Plackett-Luce モンテカルロによる複勝確率変換 | `ai/predictor.py` の確率変換ロジックを差し替え可能な関数として分離 |
| 馬連・ワイド以上の券種拡張 | `bet_type` を Settings で設定可能にし、予想テーブルを汎化 |
| 週次自動取り込み・月次自動再学習 | `jobs/` モジュールに APScheduler ジョブとして定義済みのスロットを用意 |
| データ可視化の高度化（オッズ動向チャート等） | Recharts コンポーネントを page 配下に追加するのみで対応可能 |

---

## UI スタイル方針

### PageHeader コンポーネント

全 6 ルート（Dashboard / UpcomingRaces / RaceDetail / Models / Ingest / Settings）の最上部に `PageHeader` を配置し、ページ見出しを統一する。

| prop | 型 | 概要 |
|---|---|---|
| `icon` | `LucideIcon` | 左タイルに表示するアイコン（primary tinted 背景） |
| `title` | `string` | `<h1>` に出力するページ名 |
| `description` | `string?` | タイトル下のサブテキスト（省略可） |
| `children` | `ReactNode?` | 右端 actions slot（ボタン類）。Models は TrainModelDialog、Ingest は IngestRunDialog + 即時停止ボタンを配置 |

各ルートで使用するアイコン: LayoutDashboard（Dashboard）/ CalendarClock（UpcomingRaces）/ Trophy（RaceDetail）/ Brain（Models）/ Database（Ingest）/ Settings2（Settings）。RaceDetail は `course + race_class` を title に、開催日・距離・race_id を description に動的設定する（3 状態: loading / error / loaded 対応）。

### タイポグラフィ階層

| 用途 | クラス | 備考 |
|---|---|---|
| ページ h1 | `text-3xl font-bold tracking-tight` | PageHeader が全ルートに適用 |
| CardTitle | `text-base font-semibold leading-tight` | `src/components/ui/card.tsx` の CardTitle デフォルト値 |
| Sidebar ロゴ span | `text-base` | ヘッダーロゴの文字サイズ |

ページ h1 は `tracking-tight` を加えて視認性を高め、CardTitle は `text-2xl` から `text-base` に縮小してカード内コンテンツとのバランスを改善している。

### Sidebar active state

| 項目 | 値 |
|---|---|
| 幅 | `w-60`（変更前: `w-56`） |
| active 背景 | `bg-primary/10 text-primary` + 左 inset shadow（変更前: `bg-primary text-primary-foreground` 反転塗りつぶし） |
| 項目間隔 | `space-y-0.5`（変更前: `space-y-1`） |

active state を反転塗りつぶしから inset shadow + tint に変更することで、選択中アイテムをより控えめに示し、コンテンツエリアへの視線誘導を妨げないようにしている。

### ブランド資産

| ファイル | 役割 |
|---|---|
| `public/favicon.svg` | 馬蹄モノグラムをモチーフとした favicon。カラーは HSL 直書き（CSS 変数非依存） |
| `public/logo.svg` | favicon 同モチーフのサイドナビ用ロゴ。`<img src=...>` 経由で読み込むため CSS 変数は解決されないので、カラーは HSL 直書き（テーマ追従させたい場合は inline SVG / React component 化が必要） |

`index.html` の favicon link は `/favicon.svg` を参照する。Sidebar のヘッダ領域に `<img src="/logo.svg" />` を 24x24 で配置する。

### micro-interactions

#### Card hover

- `ui/card.tsx` に `transition-shadow duration-150` を全 Card 共通で付与し、hover 時の影変化を滑らかにする
- クリック可能なカード（`RaceCard` / `ActiveModelCard`）は hover 時に `shadow-lg` + `border-primary/30` アクセントを追加し、インタラクティブであることを視覚的に示す

#### Dialog overlay

- `ui/dialog.tsx` の overlay を `bg-black/80` から `bg-black/60 backdrop-blur-sm` に変更し、奥行き感を演出する

#### Skeleton shimmer

- `ui/skeleton.tsx` のアニメーションを Tailwind デフォルトの `animate-pulse` から独自 keyframes `animate-skeleton-shimmer` に変更する
- `tailwind.config.ts` に keyframes を定義（opacity 0.6 → 1 → 0.6、1.8s、ease-in-out）。デフォルトの pulse より控えめで目に優しい点滅にする
