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
├── services/     買い目の決済と集計。`bet_settlement` が payouts と買い目を突き合わせて
│                 bet_records を確定し（連系は `core.bet_types.normalize_combo` を通す）、
│                 `bet_analytics` が回収率・的中率を集計する（DB / HTTP 非依存の純関数）
├── core/         設定（Settings）・ロギング・settings_store（JSON 永続化）・bet_types。
│                 買い方の設定の解決は `settings_store.resolve_betting_settings` が単一の入口
├── api/          FastAPI ルーター群（schemas / deps / routers/*）
│                 ビジネスロジックは持たず、上記モジュールを呼ぶだけ
└── jobs/         取り込み・運用 CLI（ingest / ingest_range / ingest_odds / backup_db 等）。上記モジュールを呼ぶ
```

### 依存方向

```text
api / jobs → ai / features / services → db (SQLAlchemy models)
scraper → db          （api / jobs からのみ呼ばれる）
core は横断
```

循環依存は禁止。`ai` は `scraper` を直接呼び出さない。`ai` / `features` / `services` は
同じ層（policy の `domain`）なので、この 3 つの間に順序は無い。

フロント側も同じ向きを持つ。

```text
routes / App / router → components → lib / hooks / store / types
```

`hooks` から `components` を呼ばない。両方向とも `.claude/policy.yml` の `code.layers` に
書き起こしてあり、`/claude-keeper:check` が実際の import を数えて逸脱を出す。
sonner の `toast` を `components/ui/` に置いていた頃はここが 5 本逆流していた
（中身は関数であってコンポーネントではないので `lib/toast.ts` へ移した）。

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

## UI 画面構成

### 画面一覧と役割

| # | 画面名 | ルート | 役割 | 対応 API |
|---|---|---|---|---|
| 1 | Dashboard | `/` | **モデルの 1 画面**。KPI（単勝回収率 / 複勝回収率 / 本命の的中率 / log-loss）+ 一覧 + 学習 + 計測 + 役割の割り当て | `GET /api/metrics/summary`, `GET /api/models`, `POST /api/models/train`, `POST /api/models/{id}/evaluate`, `POST /api/models/{id}/activate`, `PUT /api/settings` |
| 2 | Race | `/races` | 月カレンダーで日を選び、その日のレース一覧と取込操作をまとめる | `GET /api/races/calendar`, `GET /api/races/by_date`, `POST /api/scraper/run_shutuba`, `POST /api/scraper/run_results` |
| 3 | Race Detail | `/races/:race_id` | レース概要 + 出走馬表（予想確率）+ 推奨買目（1 点ずつ / 購入用 / 答え合わせ の 3 タブ）| `GET /api/races/{race_id}`, `GET /api/predictions/{race_id}`, `GET /api/recommendations/{race_id}` |
| 4 | Ledger | `/ledger` | 購入記録と収支（回収率・的中率・券種別内訳・損益推移） | `GET /api/bets*` |
| 5 | Model Detail | `/models/:model_id` | モデル 1 件の詳細と、期間・予算を指定したバックテスト | `GET /api/models/{id}`, `POST /api/simulation/start`, `GET /api/simulation/runs/{run_id}` |
| 6 | Settings | `/settings` | 全レース共通の予想パラメータとスクレイパー設定（SCRAPER / BETTING タブ） | `GET /api/settings`, `PUT /api/settings` |

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
│  「手持ちのモデル」= 一覧 + その操作を同じ塊に
│    見出し行の右に [ID を詰める] [再学習を実行]
│      (ページ見出しの横に置くと何に対する操作か読めない)
│  ModelTable（評価窓つき。Activate / 計測 / 確率に設定 / 名称編集 / 削除）
│    「計測」= 実運用の賭けルールで測り直す。学習時の指標しか無い行の
│    「未算出」はこれで埋まる (10 分前後・JobProgressCard で進捗)
│    推移グラフは置かない — モデルごとに評価窓が違い時系列に並べても読めない
└─────────────────────────────────────────┘

画面は **2 つの塊**に分ける: 「いま予想に使っているモデル」(運用中の 2 つと
その数字) と「手持ちのモデル」(一覧と、増やす / 詰める操作)。以前は帯と表が
同じ重さで縦に並び、どこからどこまでが 1 つの話か読み取れなかった。
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
│    1着確率 / 3着内率（数値だけ・背後のバーは置かない）
│    行をクリックすると、その馬の**このレース日より前**の
│      過去走が下に開く (HorsePastRuns / 複数頭を同時に開ける)
│    BUY バッジは置かない — 買うのは常にモデル 1 位の 1 頭で、
│      列を 1 つ使って「1 位かどうか」を二重に示していただけ
│    「本命」カードも置かない — 推奨買目の 1 行目がそれ
│    使う数字は 的中確率 と 確信度 の 2 つ。EV は参考列のみ
├─────────────────────────────────────────┤
│  RecommendationParamsBar + RecommendationsCard
│    予算 / 券種 を切り替えて再計算 (1 点 = 100 円は固定)
│    タブ 1「1 点ずつ」= 買う順序どおりに並べる
│      単勝 → 複勝 → 連系 → 的中確率の高い順 (エンジンと同じ)
│      確信度の列 = 確率モデルから見た「その買い目が当たる確率」
│      (券種をまたいで同じ意味。単勝=1着 / 複勝=3着以内 / 連系=組合せ)
│      EV は「参考」列に降格。買う判定に使っていないため
│    タブ 2「購入用」= 流し / ボックス / フォーメーションに畳む
│      畳めるのは推奨と点数が一致するときだけ。行を開くと 1 点ずつ
│      券種ごと / 全部まとめて購入記録に入れられる (POST /api/bets/bulk)
│    タブ 3「答え合わせ」= 確定後だけ出す
│      この買い目を全部買った場合の収支 / 回収率
│      券種ごとの内訳つき (payouts と combo を突き合わせ)
│    買い方の説明は折り畳み 1 つに集約 (BettingRuleDetails)
└─────────────────────────────────────────┘
```

出走馬表は `components/EntryPredictionTable.tsx`、並べ替えの規則は
`lib/entrySort.ts` に分けてある（`routes/RaceDetail.tsx` が 958 行に伸びたため）。
表は props だけで完結していて画面側の状態を見ない。`LowInformationNotice` /
`RunProgress` / `RaceStepper` はこの画面固有のつくりものなので置いたまま。

#### Ledger

```text
┌─────────────────────────────────────────┐
│  サマリ（投資額 / 払戻 / 回収率 / 的中率）
├─────────────────────────────────────────┤
│  ProfitChart（0 起点の損益推移）        │
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
│  ModelSimulationPanel — 5 つの塊に分ける │
│   1 条件   期間 / 1 レースに使う上限 / 買う馬券 を横 1 列
│            使うモデル (この画面 + Settings の確率モデル)
│            **RACE 画面と同じ仕組みで回す** — 入力も買い方も揃える
│            (初期資産・賭け金の決め方・戦略・狙い方は廃止)
│   2 結果   累計損益 / 必要だった資金 / 回収率。**実行条件はこの中**
│            最大益・最大損は出さない (最大損は「必要だった資金」と
│            符号違いの同じ数字で、山と谷は 3 のグラフが示す)
│   3 損益推移  ProfitChart (0 起点)
│   4 内訳   馬券種別 / レース格別 / コース別 をタブで切り替え
│            (3 表を積むと縦に伸びるだけで見比べられない)
│   5 過去の実行      入力と結果の間に挟まないよう最後に置く
│    実行は POST /simulation/start（非同期）
└─────────────────────────────────────────┘
```

取込の手動実行・スクレイパーの稼働状況・緊急停止は `DayIngestPanel`（Race 画面）に集約する。

#### Settings

```text
┌─────────────────────────────────────────┐
│  PageHeader                             │
├─────────────────────────────────────────┤
│  Tabs: SCRAPER / BETTING                │
│  SettingsForm（react-hook-form + zod。  │
│    activeSection で 1 タブ分だけ表示し、│
│    マウントは維持して入力値を保つ）     │
│  ├─ SCRAPER                             │
│  │    User-Agent / rate_min / rate_max /│
│  │    night_min（rate_min ≤ rate_max）  │
│  └─ BETTING                             │
│       複勝を買う確信度の下限 (3 着内率) │
│       単勝のオッズ下限                  │
│       1 レースに使う上限                │
│       連系を買う確信度の下限（券種ごと）│
└─────────────────────────────────────────┘
```

BETTING に **EV 閾値は無い**。どの券種でも期待値は買う/買わないの判定に使わず、
「的中確率の高い順に予算まで」買う。EV 条件を入れると回収率が落ちることが実測で
分かっているため（`docs/ai-model.md`）。券種ごとの 1 点あたり金額（旧 `stake_units`）と
ふだん買う券種（旧 `enabled_bet_types`）は 2026-09-01 に設定から廃止した。枠連は
`core/bet_types.py` の `supported_bet_types()` が落とすので、画面に選択肢として出ない。

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
- **sonner**: Toast 通知ライブラリ。`toast` は関数なので `src/lib/toast.ts`、画面に出す `<Toaster />` は `src/components/ui/toaster.tsx` に置き、`main.tsx` でマウントする
- **react-hook-form + zod**: SettingsForm / TrainModelDialog の共通フォームバリデーションパターン。`zodResolver` + `mode: 'onChange'` で inline error を表示し、submit ボタンを自動 disable する。ダイアログは `open` のたびに `reset` で初期値を復元する。ライブラリを直接使う（shadcn の `ui/form.tsx` ラッパは使っていないので 2026-09-02 に削除した）
- **購入記録を変える mutation は `useBetMutation` に集約**（`hooks/useBetMutation.ts`）。記録・まとめ記録・削除の 3 つは成功トーストの文言以外が同じで、失敗時の扱いと無効化するキー `['bets']` は 3 つとも同時に変わる
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
| 券種の追加 | 対応券種は `core/bet_types.py` に集約し、`ai/betting/strategy.py` と `combo_min_hit_prob`（買う下限）が同じ定義を参照する（現状 枠連は未対応で、UI の選択肢にも出さない）|
| 週次自動取り込み・月次自動再学習 | `jobs/` の CLI（`ingest_range` 等）が冪等・レジューム可能なため、外部スケジューラ（cron / タスクスケジューラ）から定期実行するだけで自動化できる |
| データ可視化の高度化（オッズ動向チャート等） | Recharts コンポーネントを page 配下に追加するのみで対応可能 |

---

## フロントエンド スタイル設計

### 方向 —「実測」（2026-09-05）

この道具でいちばん多い操作は買い目を決めることではなく、**測った値を読むこと**
（回収率・的中率・95% 区間・in-sample か OOS か）。見た目もそこに合わせる。

決めたことは 3 つ。

1. **chrome は色を持たない。** 地・罫線・文字は無彩に近い黒鉛で組む
2. **アクセント 1 色は〈測れているか〉を指す。** 金額でもオッズでもない
3. **彩度はデータのもの。** 枠色と損益にしか強い色を使わない

3 が制約の本体。馬が並ぶ画面には枠色チップ（`Umaban`）が常に 8 色出ているので、
chrome にブランド色を足すと **chrome がデータと彩度を奪い合う**。旧トークンの
琥珀（`--primary`、46 か所）を外したのはこれが理由で、好みではない。外したことで
琥珀は `--warning` に戻り、本当の警告に使えるようになった。

避けたものを 3 つ記録しておく。**次に迷ったらここへ戻る。**

| 避けた形 | なぜ |
|---|---|
| クリーム地（旧 light の `#f7f4ef`） | 生成物によくある配色で、既視感が強い |
| 同じ角丸カードに全部を切り分ける形 | どこかの SaaS ダッシュボードに見える。指標カードは 1 画面 1〜3 個まで |
| 琥珀を面で使う | 「的中」を煽る見た目になり、主役が数字でなくなる |

### 色の 3 層

| 層 | 何が入るか | 誰が決めるか | 変えてよいか |
|---|---|---|---|
| データ | 枠色 8 色（`lib/waku.ts`）、曜日（土 = 青 / 日 = 赤） | JRA と暦 | **変えない。**デザインの都合で動かさない |
| 値の向き | `--success` / `--destructive`（損益・的中・成否） | 実装 | 面積が小さくても常に色を持たせる |
| chrome | `--primary`（実測）と `--warning`（警告）の 2 つだけ | この規定 | **ここを増やさない** |

chrome の色を 3 つ目にしない。増えた時点で 3 層の区別が読めなくなり、
「色が付いている = 何かある」以上の意味を運べなくなる。

**アクセントを CTA に使わない。**`--primary` が指すのは
〈測れているか〉と〈いまどこにいるか〉（現在地・選択中）の 2 つだけ。
ボタンの `default` は**無彩色の反転**（`bg-foreground text-background`）で、
強さは色ではなくコントラストで出す。アクセントを面で塗ったボタンは
画面でいちばん強いものが「実行」になり、煽る見た目に寄る
（実際にそうなっていたのを 2026-09-05 に直した）。`soft` と `outline` も同じ理由で
アクセントを持たない。**アクセントの面を持つのは `Badge` の `solid` / `tone="default"` だけ。**

### デザイントークン（CSS 変数）

`globals.css` が唯一の出どころ。値は **HSL 三つ組**（Tailwind が `hsl(var(--x))` で読む）。
**dark が主・light が従**で、light は面が明るいぶんアクセントと状態色の明度を落とす。

| トークン | 用途 | dark | light |
|---|---|---|---|
| `--background` | 地 | `205 18% 9%` `#13181B` | `205 16% 96%` `#F3F5F6` |
| `--card` | 面 | `205 16% 12%` `#1A1F23` | `0 0% 100%` `#FFFFFF` |
| `--card-elevated` | 一段上げた面 | `205 14% 16%` `#232A2F` | `205 14% 92%` `#E8EBED` |
| `--foreground` | 文字 | `205 12% 92%` `#E8EBED` | `205 20% 12%` `#182025` |
| `--muted-foreground` | 副次の文字 | `205 9% 64%` `#9BA5AB` | `205 10% 40%` `#5C6870` |
| `--subtle-foreground` | ラベル・単位 | `205 8% 44%` `#677279` | `205 9% 54%` `#7F8B94` |
| `--border` | 区切りの罫 | `205 12% 20%` `#2D3439` | `205 14% 87%` `#D9DFE2` |
| `--border-strong` | 囲い・表ヘッダ下の太罫 | `205 10% 32%` `#49535A` | `205 12% 72%` `#AFB9C0` |
| `--primary` | **実測**。OOS バッジ・95% 区間・現在地・選択中 | `188 58% 55%` `#4ABDCF` | `188 62% 34%` `#217E8C` |
| `--success` | プラス収支・的中・成功 | `158 52% 50%` `#3DC291` | `158 58% 29%` `#1F7555` |
| `--destructive` | マイナス収支・失敗・日曜 | `6 68% 60%` `#DE6254` | `6 68% 45%` `#C13425` |
| `--warning` | 本当の警告のみ | `38 85% 58%` `#EFAC39` | `38 88% 36%` `#AD710B` |
| `--info` | **土曜日だけ。**暦の慣習で青 | `218 62% 62%` `#628EDA` | `218 64% 42%` `#2759B0` |

地の色相を 205°（わずかに寒色寄りの黒鉛）にしてあるのは、**枠色の赤・橙を
暖めないため**。暖色の地に置くと枠色 3（赤）と 7（橙）の差が縮む。

`--info` は補足でもリンクでもなく、**土曜日を青く出すためだけ**にある（`RaceCalendar` /
`DateYMDPicker` の 3 か所。日曜は `--destructive`）。色相を 218° に置いたのは
`--primary` の 188° と 30° 離すためで、近いと「実測」と「土曜」が同じ色に見える。

各色には `-foreground` の対（その面に載せる文字色）があり、`tailwind.config.ts` の
`theme.extend.colors` に登録してある。`bg-success text-success-foreground` の形で使う。

> web font は `index.html` が Google Fonts から Inter / JetBrains Mono / Noto Sans JP を
> 読み込む（ウェイトは 400/500/700 の 3 段のみ。Inter 600 と Noto Sans JP 600 は太さが
> 揃わないので混ぜない）。Inter に和文字形が無いことを利用し、**フォールバック順序だけで
> 「英数字 = Inter / かな漢字 = Noto Sans JP」**になる。指定しないと Windows は Yu Gothic UI、
> macOS はヒラギノに落ちて環境ごとに別アプリのように見える。

### 字の尺度（5 段）

**この 5 つ以外を使わない。**`tailwind.config.ts` は `fontSize` を丸ごと差し替えて
**この 4 キーしか生成しない**ので、`text-base` / `text-xl` / `text-2xl` を書いても
クラスが出ず無効になる。中間値を作れなくするための仕掛けで、消し忘れではない。

| 段 | クラス | px | 用途 |
|---|---|---|---|
| 1 | `text-2xs` | 11 | ラベル・単位・バッジ・補足 |
| 2 | `text-xs` | 12 | 表のセル、小さい本文 |
| 3 | `text-sm` | 14 | 本文（`body` の既定） |
| 4 | `text-lg` | 18 | 見出し（`h1` / `CardTitle`） |
| 5 | `.text-kpi` | 26 | 数値。等幅 + `tabular-nums` |

**見出しを太らせない。**画面の中でいちばん強いのは見出しではなく数字なので、
`h1` は `text-lg font-semibold tracking-tight` にとどめる。段が 5 つで足りるのは
「読み物」を持たない画面だから。長い説明が要るなら本文（14px）に収める。

数字まわりの補助クラスは `globals.css` に置く（`.text-kpi` / `.text-label` /
`.text-label-ja` / `.text-num` / `.text-unit` / `.cell-num`）。**すべて等幅 +
`tabular-nums`** で、桁が縦に揃うことで「計測した値」に見える。プロポーショナル数字だと
ただの大きい文字になる。

ラベルを**大文字化しない・字間を開けない**（`text-transform: none`、`0.04em` まで）。
トラッキングした全大文字は「読み物」の署名で、和文ラベル（単勝回収率）と並べたときに
段差が出る。

### 余白の尺度

**4 の倍数のみ。**ただし最小の詰めとして 2px（`0.5`）だけ例外に置く。

```
0.5 = 2px   チップと数字のように、隣接を示すだけの詰め
1   = 4px    2 = 8px    3 = 12px    4 = 16px
6   = 24px   8 = 32px  12 = 48px
```

**6px / 10px / 14px（`1.5` / `2.5` / `3.5`）を作らない。**1 つ許すと、次の画面が
その隣をまた作る。4px と 8px の間に迷ったら **8px を取る**（詰めるより離すほうが、
表の行が増えたときに崩れにくい）。**例外は表の中だけ** — セル内の
アイコンとラベルは 4px で詰める（8px にすると 1 行の高さが伸び、
1 画面に入るレース数が減る）。

**アイコンの寸法はこの尺度に乗せない。**余白ではなく文字に合わせる量なので、
`14 / 16 / 18 / 20px`（`h-3.5` / `h-4` / `h-[18px]` / `h-5`）の 4 つを持つ。
本文 14px の隣に 14px のアイコンが並ぶのは正しく、4 の倍数へ丸めると
文字よりアイコンが大きく見える。

### 角丸

**2px の 1 値だけ。**ピル（バッジ）の `rounded-full` が唯一の例外で、
「基本は直角、意図があるときだけ完全な丸」というリズムを作る。

| 値 | クラス | 使いどころ |
|---|---|---|
| 2px | `rounded` / `rounded-sm` / `rounded-md` | ボタン・入力・チップ・馬番・ダイアログ・囲う Card |
| 9999px | `rounded-full` | バッジだけの例外 |

丸みは「やわらかい・親しみやすい」記号で、計測する道具の方向とは逆なのでほぼ捨てる。
**ダイアログも 2px**（面が大きいほど丸みが目に付くので、ここを緩めると方向が崩れる）。

`tailwind.config.ts` は `borderRadius` を **`extend` ではなく丸ごと差し替える**。
`extend` に置くと Tailwind 既定の `rounded-lg`（8px）などが生き残り、名前だけ違う
中間値が生えてくる。`DEFAULT` も 2px に寄せてあるので、素の `rounded` が
別の値を指す抜け道も無い。`rounded-[2px]` のような直書きはしない。

shadcn 既定の `sm = radius - 4px` という引き算スケールは使わない。`--radius` を 2px に
すると `calc(2px - 4px)` が負値になり、**指定ごと無効になって完全な直角に落ちる**。

### 状態の扱い

**空が既定の状態。**スクレイピング済みデータも学習済みモデルもリポジトリに含まれないので、
**初回起動は全画面が空**になる。ここを飛ばすと、いちばん多い状態を見ていないことになる。

| 状態 | 出し方 | 決めたこと |
|---|---|---|
| 空 | `EmptyState` | `message` は**次にする操作**を書く（「データがありません」で終えない）。アイコンは 40px・`text-muted-foreground/25` |
| 読み込み中 | `ui/skeleton.tsx` | 行数を実データの行数に寄せる。出た瞬間に高さが変わると読む位置を失う |
| エラー | `EmptyState` + 状況に合った lucide アイコン | 何が起きたかを書く。謝らない・ぼかさない |
| 長文 | `truncate` + `title` 属性 | 折り返して行数を変えない。表の行高が揃わなくなる |

### Badge の組み方

`src/components/ui/badge.tsx` は **形 × 意味の 2 軸**で組む。以前は
`default / secondary / destructive / outline / success / warning / info` ＋ `soft-*` 6 種で
13 通りあり、どれを使うかが場当たりになっていたのを畳んだもの。

| 軸 | 値 | 使いどころ |
|---|---|---|
| `variant`（形） | `solid` | 「結論」を示すもの。**1 画面に 1 種類まで** |
| | `soft`（既定） | 状態の表示（実行中・完了・失敗） |
| | `outline` | 分類の表示（クラス・券種）。**常に無彩色** |
| `tone`（意味） | `default`（= 実測） / `success` / `destructive` / `warning` | `solid` と `soft` にだけ効く |

**`info` と `secondary` は Badge には無い**（`--info` トークン自体は土曜日の表示に使う）。
`tone="default"` は `--primary` を引くので、**「測った値」を指すバッジ**になる
（OOS・実測・95% 区間）。学習時の値には付けない。

ハードコードされた Tailwind カラークラス（`bg-emerald-600 text-white` 等）は使わない。

---

## UI スタイル方針

**画面はこの規定の値だけを使う。ベタ書きしない。**規定に無い値が要ると分かったら、
画面に足さずここへ戻る。ここで例外を作ると、規定は 1 か月で飾りになる。

### PageHeader コンポーネント

各ルートの最上部に `PageHeader` を配置し、ページ見出しを統一する。

| prop | 型 | 概要 |
|---|---|---|
| `icon` | `LucideIcon` | 左タイルに表示するアイコン |
| `title` | `string` | `<h1>` に出力するページ名 |
| `description` | `string?` | タイトル下のサブテキスト（省略可） |
| `children` | `ReactNode?` | 右端 actions slot（ボタン類） |

RaceDetail は `course + race_class` を title に、開催日・距離・race_id を description に
動的設定する（loading / error / loaded の 3 状態に対応）。

### 領域の作り方（罫線 / 面 / 箱）

区切る手段は 3 つ。**どれを使うかは中身で決まる。**

| 手段 | 見た目 | 使うもの |
|---|---|---|
| 罫線と余白 | 線 1 本 + 余白 | **表とグラフ**、節の区切り |
| 面（`.block-surface` / `-compact`） | 背景 `--card` + 角丸 2px、**罫線なし** | **ラベルと値の塊** |
| 箱（`Card boxed`） | 背景 + 罫線 | 注意喚起、その画面の答えになる指標 |

**表とグラフは囲わない。**それ自体でまとまって見えるので、囲うと枠が二重になる
（画面直下の塊をすべて箱にしたことがあるが、表もグラフも囲われて散らかった）。

**ラベルと値の塊には面を使い、罫線を足さない。**以前は上下を罫線で挟む
（`border-y`）か節ごとに `border-t` を引いていたが、そういうブロックが縦に積むと
**引きで見たときに線が縞になり、内容より先に線が目に付く**（2026-09-05）。
面なら罫線 0 本で同じまとまりが作れる。該当するのは Race の取込状況と
Race Detail のレース概要。

**指標カードが並ぶ帯には面を使わない** — `MetricCard` 自身が箱なので、
面に載せると枠が二重になる（Ledger で実際にそうなった）。

面の内側の余白は 2 段。

| クラス | 余白 | 使いどころ |
|---|---|---|
| `.block-surface` | 24px | 見出し + 複数行を載せる面（レース概要）。12px だと中身が縁に張り付き、「面に置いた」ではなく「背景が変わった」に見える |
| `.block-surface-compact` | 縦 12px / 横 16px | 1 行だけの帯（取込状況）。24px だと帯が厚くなりすぎる |

**面の直後に罫線を引かない。**面の下端が既に区切りなので、次の節に `border-t` を
足すと線が二重になる（Race Detail のレース概要とその下の節が実際にそうだった）。

**箱（罫線あり）は 2 つだけ。**

1. **注意喚起** — 「判断材料が少ない」など、面で出したいもの（`Card boxed`）
2. **その画面の答えになる指標** — `MetricCard`。Dashboard なら単勝・複勝の回収率、
   シミュレーションなら累計損益

指標カードは **1 画面 1〜3 個まで**。5 つ並べた時点でどれも目立たなくなり、
囲う意味が消える。残りの指標は素の値のまま並べる（`MetricBand` / `Figure`）。

**面と箱を入れ子にしない。**入れ子にすると外側の意味が消える。

### ブランド資産

モチーフは馬蹄（∩ + 両端の studs）。パス座標は 3 か所すべてで共有する（viewBox だけが違う）。

| ファイル | 役割 |
|---|---|
| `src/components/BrandMark.tsx` | **アプリ内のマーク**。inline SVG + `currentColor` でテーマ追従する。地（タイル）を持たない素のグリフ |
| `public/favicon.svg` | ブラウザタブ用のタイル。`<link rel="icon">` 経由では CSS 変数が解決されないため、`--card` / `--border-strong` / `--primary` の dark 値を HSL リテラルで写している |
| `public/logo.svg` | アプリ外（README / 資料 / OG 画像）用の単体グリフ。同じ理由で色を直書き。**アプリ内では使わない** |

favicon のタイル地は明暗どちらのタブ地色でも成立するよう常に黒鉛（`#1A1F23`）で、
light テーマ用の別タイルは持たない。**トークンを変えたらこの 2 つの svg も直す**
（CSS 変数が届かないので自動では追随しない）。

Topbar のロゴは `<BrandMark className="h-[18px] w-[18px] text-primary" />` を 18x18 で置く。
ここでタイルではなく素のグリフを使うのは、「箱をやめて罫線で区切る」に従い
ヘッダに app タイルを刺さないため。

### micro-interactions

動きは 2 つだけ。**画面に入るときのアニメーションは持たない**（測った値を読む道具で、
値より先に動きが目に入る理由がない）。

- **Dialog overlay** — `bg-black/60 backdrop-blur-sm`。奥行きで前後を示す
- **Skeleton** — `animate-skeleton-shimmer`（opacity 0.6 → 1 → 0.6、1.8s）。
  Tailwind 既定の `animate-pulse`（0.5 → 1）は振れ幅が大きく、読んでいる最中に目に付く
