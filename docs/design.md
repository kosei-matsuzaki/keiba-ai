# KEIBA AI — 設計方針書

関連ドキュメント: [spec.md](spec.md) / [data-pipeline.md](data-pipeline.md) / [ai-model.md](ai-model.md) / [operations.md](operations.md)

---

## 設計の出発点

### ユーザー要件

- 今週末の出走予定レースを一覧表示し、馬ごとの単勝・複勝予想確率を確認したい
- モデルの成績（回収率・的中率・確率の質）を、出所と評価窓つきで把握したい
- 手元の PC 上で完結して動作し、外部サービスへデータを送出しない

### 非機能要件

| 項目 | 方針 |
|---|---|
| 動作形態 | ローカル dev サーバ（`scripts/dev.sh` で uvicorn :8765 + Vite :5173 を起動）+ ブラウザアクセス |
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
│  (netkeiba) │ │ (NN推論)   │ │  keiba.db    │
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
backend/src/
├── main.py       FastAPI app factory (create_app) + lifespan + CORS + uvicorn __main__
├── scraper/      スクレイピング専用。HTML 取得・パース・DB 保存のみ。AI を知らない
├── features/     DB から生データを読み取り、学習・推論用の特徴量 DataFrame を生成
│                 （リーク防止のため「予測時点での情報のみ使用する」制約を徹底管理）
│                 race_info.py がレース単位の情報量（新馬戦などの「履歴が無いレース」）を判定する
├── ai/           特徴量を受け取り NN の学習・評価・推論を実行。features を知らない
│                 依存 DAG の層で機能サブパッケージ化されている:
│                 ├── core/       types / labels / splits / temperature / probabilities（最下層）
│                 ├── model/      registry / _artifacts_nn + NN 実装（net / loss / dataset / preprocess）
│                 ├── training/   train_nn（学習 CLI）
│                 ├── inference/  predict — bundle-aware 推論（predict_race / *_with_combinations）
│                 ├── betting/    odds / strategy（ベット選定・賭け金配分）
│                 ├── simulation/ engine / persistence（バックテストシミュレーション）
│                 └── evaluation/ backtest — NDCG@k・ヒット率・ROI 計算
├── core/         設定（Settings）・ロギング・settings_store（JSON 永続化）
├── api/          FastAPI ルーター群（schemas / deps / routers/*）
│                 ビジネスロジックは持たず、上記モジュールを呼ぶだけ
└── jobs/         取り込み・運用 CLI（ingest / ingest_range / ingest_odds / backup_db 等）。上記モジュールを呼ぶ
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
| 1 | Dashboard | `/` | **モデルの 1 画面**。KPI（単勝回収率 / 複勝回収率 / 本命の的中率 / log-loss）+ 一覧 + 学習 + 計測 + 役割の割り当て | `GET /api/metrics/summary`, `GET /api/models`, `POST /api/models/train`, `POST /api/models/{id}/evaluate`, `POST /api/models/{id}/activate`, `PUT /api/settings` |
| 2 | Race | `/races` | 月カレンダーで日を選び、その日のレース一覧と取込操作をまとめる | `GET /api/races/calendar`, `GET /api/races/by_date`, `POST /api/scraper/run_shutuba`, `POST /api/scraper/run_results` |
| 3 | Race Detail | `/races/:race_id` | レース概要 + 出走馬表（予想確率・BUY バッジ）+ 推奨買目（1 点ずつ / 購入用の 2 タブ）+ 結果の答え合わせ | `GET /api/races/{race_id}`, `GET /api/predictions/{race_id}`, `GET /api/recommendations/{race_id}` |
| 4 | Ledger | `/ledger` | 購入記録と収支（回収率・的中率・券種別内訳・損益推移） | `GET /api/bets*` |
| 5 | Model Detail | `/models/:model_id` | モデル 1 件の詳細と、期間・予算を指定したバックテスト | `GET /api/models/{id}`, `POST /api/simulation/start`, `GET /api/simulation/runs/{run_id}` |
| 6 | Settings | `/settings` | 全レース共通の予想パラメータとスクレイパー設定（SCRAPER / BETTING / BET TYPES タブ） | `GET /api/settings`, `PUT /api/settings` |

旧 `/upcoming` `/past` `/ingest` は Race 画面へ、旧 `/models` は Dashboard へ統合済みで、
ブックマーク互換のため `router.tsx` が `Navigate` でリダイレクトするだけの経路として残している。

### 画面遷移図

```text
[Topbar: DASHBOARD / RACE / LEDGER / SETTINGS]（全画面共通）

[Dashboard] ─ モデル一覧の行 → [Model Detail]（重いバックテストはここ）
[Race] ─ カレンダーで日を選ぶ → レース行クリック → [Race Detail]
[Race Detail] ─ 推奨買目をまとめて記録 → [Ledger]
```

ナビは左サイドバーではなく上部の `Topbar`。アイコンも番号も置かず、等幅の英字ラベルだけで、
選択中は面ではなく色（primary）で示す。

### 各画面の主要コンポーネント

#### Dashboard

```text
┌─────────────────────────────────────────┐
│  PageHeader（ID を詰める / 再学習を実行）
├─────────────────────────────────────────┤
│  OperatingModels（運用中の 2 モデル + それぞれの数字）
│    左 買い目を決める (active): 単勝/複勝回収率・本命的中率・log-loss
│    右 確からしさを出す (確率モデル): log-loss と順位精度だけ
│    見出し右に出所チップ・レース数・評価窓。未設定側は
│    「未設定」バッジ + 設定すると何が変わるかを出す
│    **役割カードと KPI 帯は分けない** — 分けると同じ active の回収率が
│    上下 2 箇所に出て、どのモデルの数字か読み取れなくなる
├─────────────────────────────────────────┤
│  ModelTable（評価窓つき。Activate / 計測 / 確率に設定 / 名称編集 / 削除）
│    「計測」= 実運用の賭けルールで測り直す。学習時の指標しか無い行の
│    「未算出」はこれで埋まる (10 分前後・JobProgressCard で進捗)
│    推移グラフは置かない — モデルごとに評価窓が違い時系列に並べても読めない
└─────────────────────────────────────────┘
```

#### Race

```text
┌─────────────────────────────────────────┐
│  DataCoverageBand（どこまで取り込めているか）
├──────────────────┬──────────────────────┤
│  RaceCalendar    │  DayIngestPanel      │
│   月表示。日ごと │   選んだ日の取込操作 │
│   に開催と取込   │   過去=結果 /        │
│   状況を色で示す │   当日=両方 /        │
│                  │   未来=出馬表        │
├──────────────────┴──────────────────────┤
│  選択日のレース一覧（行クリックで Race Detail）
└─────────────────────────────────────────┘
```

#### Race Detail

```text
┌─────────────────────────────────────────┐
│  レース概要（コース・距離・馬場・頭数）  │
│  LowInformationNotice                   │
│    出走馬全員が初出走のレース（新馬戦など）で
│    「判断材料が少ない」と明示する       │
├─────────────────────────────────────────┤
│  出走馬表（スコア降順）                 │
│    馬番（Umaban・枠色）/ 馬名 / スコア  │
│    単勝 prob / 複勝 prob / BUY バッジ   │
│    BUY バッジは置かない — 買うのは常にモデル 1 位の 1 頭で、
│      列を 1 つ使って「1 位かどうか」を二重に示していただけ
│    買い方は表の下に 1 箇所だけ: 券種 × 買う条件 × 点数
│      使う数字は 的中確率 と 確信度 の 2 つ。EV は参考列のみ
├─────────────────────────────────────────┤
│  RecommendationParamsBar + RecommendationsCard
│    予算 / 1 点あたり / 券種 を切り替えて再計算
│    タブ 1「1 点ずつ」= 買う順序どおりに並べる
│      単勝 → 複勝 → 連系 → 的中確率の高い順 (エンジンと同じ)
│      確信度の列 = 確率モデルから見た「その買い目が当たる確率」
│      (券種をまたいで同じ意味。単勝=1着 / 複勝=3着以内 / 連系=組合せ)
│      EV は「参考」列に降格。買う判定に使っていないため
│    タブ 2「購入用」= 流し / ボックス / フォーメーションに畳む
│      畳めるのは推奨と点数が一致するときだけ。行を開くと 1 点ずつ
│      券種ごと / 全部まとめて購入記録に入れられる (POST /api/bets/bulk)
│    まとめて Ledger に記録できる         │
├─────────────────────────────────────────┤
│  結果の答え合わせ（確定後）             │
└─────────────────────────────────────────┘
```

#### Ledger

```text
┌─────────────────────────────────────────┐
│  サマリ（投資額 / 払戻 / 回収率 / 的中率）
├─────────────────────────────────────────┤
│  BankrollChart（損益推移）              │
├─────────────────────────────────────────┤
│  券種別内訳 + 購入記録テーブル          │
│    AddBetDialog で手動登録 / CSV 書き出し
└─────────────────────────────────────────┘
```

#### Model Detail

```text
┌─────────────────────────────────────────┐
│  モデル概要（学習条件・評価指標）        │
├─────────────────────────────────────────┤
│  ModelSimulationPanel                   │
│    期間 / 予算 / 買う馬券 / 狙い方 / 1 レースの上限
│    **RACE 画面と同じ語彙・同じ刻みにする** —
│      シミュレーションは「RACE 画面の予想を全レースで
│      やったらどうなるか」を見るものなので、条件の
│      呼び名が違うと比べられない
│    狙い方: 本命中心 3 / 標準 5 / 穴も拾う 8 頭
│      (旧「戦略」は 1 点の額も動かしていたが、
│       1 点 = 100 円に固定したので頭数だけになった)
│    使うモデル: 買い目を決める側 (この画面) │
│      と確からしさを出す側 (Settings 由来) │
│    結果に「この実行の条件」を併記        │
│    「履歴の無いレースを除外」チェック   │
│      → exclude_low_information          │
│    実行は POST /simulation/start（非同期）
├─────────────────────────────────────────┤
│  BankrollChart + 券種別内訳             │
└─────────────────────────────────────────┘
```

取込の手動実行・スクレイパーの稼働状況・緊急停止は `DayIngestPanel`（Race 画面）に集約する。

#### Settings

```text
┌─────────────────────────────────────────┐
│  PageHeader                             │
├─────────────────────────────────────────┤
│  Tabs: SCRAPER / BETTING / BET TYPES    │
│  SettingsForm（react-hook-form + zod。  │
│    activeSection で 1 タブ分だけ表示し、│
│    マウントは維持して入力値を保つ）     │
│  ├─ SCRAPER                             │
│  │    User-Agent / rate_min / rate_max /│
│  │    night_min（rate_min ≤ rate_max）  │
│  ├─ BETTING                             │
│  │    単勝のオッズ下限                  │
│  │    複勝を買う確信度の下限            │
│  │    1 レースに使う上限 / 1 点あたり   │
│  └─ BET TYPES                           │
│       買う券種のチェックと、その券種の  │
│       1 点あたり金額を同じ場所で編集    │
└─────────────────────────────────────────┘
```

BETTING に **EV 閾値は無い**。どの券種でも期待値は買う/買わないの判定に使わず、
「的中確率の高い順に予算まで」買う。EV 条件を入れると回収率が落ちることが実測で
分かっているため（`docs/ai-model.md`）。枠連は AI が買い目を生成しないので
BET TYPES の選択肢に出さない。

**確率モデルの割り当ては Settings ではなく Dashboard のモデル一覧で行う**（モデルを見比べて
いる場所で選べないと意味がないため）。Settings に残るのは「複勝を買う確信度」という
買い方のパラメータだけ。

Card ラッパは撤廃し、SettingsForm を直接配置する。各 Section は description ヘッダを持ち、
FieldRow には help text を添える。

---

## 状態管理

### 基本方針

- **TanStack Query（React Query v5）**: サーバーデータのフェッチ・キャッシュ・再取得を管理。API 呼び出しは `src/hooks/` のカスタムフックに集約
- **Zustand**: ページをまたいで保持する UI 状態のみ管理。サーバーデータは一切持たない
- **sonner**: Toast 通知ライブラリ。`src/components/ui/toast.tsx` / `toaster.tsx` を sonner ラッパとして手書き配置。`main.tsx` で `<Toaster />` をマウント
- **react-hook-form + zod**: SettingsForm / TrainModelDialog の共通フォームバリデーションパターン。`zodResolver` + `mode: 'onChange'` で inline error を表示し、submit ボタンを自動 disable する。ダイアログは `open` のたびに `reset` で初期値を復元する。`src/components/ui/form.tsx` で react-hook-form と shadcn フォームコンポーネントを統合
- API クライアントは `src/lib/api.ts` の `ky` インスタンスに集約し、各フックから呼び出す

### React Query（TanStack Query）

| クエリキー | 対応フック | 対象 API | 更新間隔 |
|---|---|---|---|
| `['races', 'calendar', year, month]` | `useRacesCalendar` | `GET /api/races/calendar` | 5 分（staleTime） |
| `['races', 'by_date', date]` | `useRacesByDate` | `GET /api/races/by_date` | 5 分（staleTime） |
| `['races', raceId]` | `useRaceDetail` | `GET /api/races/{race_id}` | ユーザー操作時のみ（refetch） |
| `['predictions', raceId]` | `usePredictions` | `GET /api/predictions/{race_id}` | ユーザー操作時のみ（refetch） |
| `['metrics', 'summary']` | `useMetricsSummary` | `GET /api/metrics/summary` | 10 分 |
| `['scraper', 'status']` | `useScraperStatus` | `GET /api/scraper/status` | アイドル: 30 秒 / 実行中: 5 秒（refetchInterval を Zustand `isRunning` で切り替え） |
| `['models']` | `useModels` | `GET /api/models` | ユーザー操作時のみ |
| `['settings']` | `useSettings` | `GET /api/settings` | ユーザー操作時のみ |
| `['recommendations', raceId, params]` | `useRecommendations` | `GET /api/recommendations/{race_id}` | ユーザー操作時のみ |
| `['bets', ...]` | `useBetList` / `useBetSummary` / `useBetBreakdown` / `useBetTimeseries` | `GET /api/bets*` | 記録の追加・削除時に invalidate |
| `['jobs', jobId]` | `useJobStatus` | `GET /api/jobs/{job_id}` | 2 秒 polling（terminal status で停止）|

### Zustand（`src/store/app.ts`）

| ストア | 保持する状態 |
|---|---|
| `useAppStore` | `sidebarOpen`（旧サイドナビの名残。ナビは Topbar に移行済み） |
| `useScraperStore` | `isRunning`（スクレイパー手動実行中フラグ — ポーリング間隔の切り替えに使用）/ `trackedJobId`（JobProgressCard が追う取込ジョブ） |
| `useTrainingStore` | `trackedJobId`（JobProgressCard が追う学習ジョブ） |

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
| DL アンサンブル（TabNet / CatBoost 等との ensemble） | `ai/model/registry.py` の `ModelBundle` と `ai/inference/predict.py` の bundle-aware 推論が、呼び出し側からモデル実装の詳細を隠蔽する |
| Plackett-Luce モンテカルロによる複勝確率変換 | `ai/core/probabilities.py` の確率変換ロジックを差し替え可能な関数として分離 |
| 券種の追加 | 対応券種は `core/bet_types.py` に集約し、Settings の `enabled_bet_types` / `stake_units` と `ai/betting/strategy.py` が同じ定義を参照する（現状 枠連は未対応で、UI の選択肢にも出さない）|
| 週次自動取り込み・月次自動再学習 | `jobs/` の CLI（`ingest_range` 等）が冪等・レジューム可能なため、外部スケジューラ（cron / タスクスケジューラ）から定期実行するだけで自動化できる |
| データ可視化の高度化（オッズ動向チャート等） | Recharts コンポーネントを page 配下に追加するのみで対応可能 |

---

## UI スタイル方針

### PageHeader コンポーネント

各ルートの最上部に `PageHeader` を配置し、ページ見出しを統一する。

| prop | 型 | 概要 |
|---|---|---|
| `icon` | `LucideIcon` | 左タイルに表示するアイコン（primary tinted 背景） |
| `title` | `string` | `<h1>` に出力するページ名 |
| `description` | `string?` | タイトル下のサブテキスト（省略可） |
| `children` | `ReactNode?` | 右端 actions slot（ボタン類）。Models は TrainModelDialog、Race は取込・即時停止のボタンを配置 |

RaceDetail は `course + race_class` を title に、開催日・距離・race_id を description に動的設定する（3 状態: loading / error / loaded 対応）。

### タイポグラフィ階層

| 用途 | クラス | 備考 |
|---|---|---|
| ページ h1 | `text-3xl font-bold tracking-tight` | PageHeader が全ルートに適用 |
| CardTitle | `text-base font-semibold leading-tight` | `src/components/ui/card.tsx` の CardTitle デフォルト値 |
| Topbar ロゴ span | `text-base` | ヘッダーロゴの文字サイズ |

ページ h1 は `tracking-tight` を加えて視認性を高め、CardTitle は `text-2xl` から `text-base` に縮小してカード内コンテンツとのバランスを改善している。

### ナビの active state

| 項目 | 値 |
|---|---|
| 幅 | `w-60`（変更前: `w-56`） |
| active 背景 | `bg-primary/10 text-primary` + 左 inset shadow（変更前: `bg-primary text-primary-foreground` 反転塗りつぶし） |
| 項目間隔 | `space-y-0.5`（変更前: `space-y-1`） |

active state を反転塗りつぶしから inset shadow + tint に変更することで、選択中アイテムをより控えめに示し、コンテンツエリアへの視線誘導を妨げないようにしている。

### ブランド資産

モチーフは馬蹄（∩ + 両端の studs）。デザイントークン刷新（青 → 琥珀、角丸 7px → 2px）に合わせて
配色とジオメトリを更新済みで、パス座標は 3 か所すべてで共有する（viewBox だけが違う）。

| ファイル | 役割 |
|---|---|
| `src/components/BrandMark.tsx` | **アプリ内のマーク**。inline SVG + `currentColor` でテーマ追従する。地（タイル）を持たない素のグリフ |
| `public/favicon.svg` | ブラウザタブ用のタイル。`<link rel="icon">` 経由では CSS 変数が解決されないため、`--card` / `--border-strong` / `--primary` の dark 値を HSL リテラルで写している |
| `public/logo.svg` | アプリ外（README / 資料 / OG 画像）用の単体グリフ。同じ理由で琥珀を直書き。**アプリ内では使わない** |

favicon のタイル地は明暗どちらのタブ地色でも成立するよう常に炭（`#14120f`）で、light テーマ用の
別タイルは持たない。`index.html` の favicon link は `/favicon.svg` を参照する。

Topbar のロゴは `<BrandMark className="h-[18px] w-[18px] text-primary" />` を 18x18 で置く。
ここでタイルではなく素のグリフを使うのは、globals.css の「箱をやめて罫線で区切る」に従い
ヘッダに app タイルを刺さないため。

### micro-interactions

#### Card hover

- `ui/card.tsx` に `transition-shadow duration-150` を全 Card 共通で付与し、hover 時の影変化を滑らかにする
- クリック可能なカード（レース一覧の行）は hover 時に `shadow-lg` + `border-primary/30` アクセントを追加し、インタラクティブであることを視覚的に示す

#### Dialog overlay

- `ui/dialog.tsx` の overlay を `bg-black/80` から `bg-black/60 backdrop-blur-sm` に変更し、奥行き感を演出する

#### Skeleton shimmer

- `ui/skeleton.tsx` のアニメーションを Tailwind デフォルトの `animate-pulse` から独自 keyframes `animate-skeleton-shimmer` に変更する
- `tailwind.config.ts` に keyframes を定義（opacity 0.6 → 1 → 0.6、1.8s、ease-in-out）。デフォルトの pulse より控えめで目に優しい点滅にする
