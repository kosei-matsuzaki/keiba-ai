# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

netkeiba スクレイピング + NN (Set Transformer ランキング) による競馬予想ツール (個人研究用)。FastAPI バックエンド + React 管理画面の単一リポジトリで、`scripts/dev.sh` が両方を一発起動する。

詳しい仕様は `docs/` 配下 (`spec.md` / `design.md` / `ai-model.md` / `data-pipeline.md` / `operations.md`) を参照。本ファイルは「コード全体を読まないと掴めない big picture」のみを要約する。環境固有の運用メモは `CLAUDE.local.md` (gitignored) に分離している。

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
uv run python -m ai.training.train_nn --train-end 2025-04-30 --valid-months 6 --test-months 6  # NN 学習 (train-end は分割の基準日。学習終了は その日 - test - valid = 2024-04-30)
uv run python -m ai.evaluation.backtest --model data/models/<ts>-nn --persist    # 評価 + metrics_json 書き戻し
uv run keiba-ingest --date 2024-12-28                                       # 単日 ingest
uv run python -m jobs.ingest_range --start ... --end ...           # 期間 ingest（中断後の resume 対応）
uv run keiba-backup                                                # keiba.db + odds.db を data/backups/ に世代バックアップ
uv run alembic upgrade head
```

## アーキテクチャ要点

### 依存方向 (厳守)
```
api → jobs → ai / features / scraper → db (SQLAlchemy models)
core / settings は横断
```
`ai` は `scraper` を直接呼ばない (循環禁止)。`api/routers/*.py` はビジネスロジックを持たず、下層モジュールを呼ぶだけ。

`ai/` は依存 DAG の層に沿って機能サブパッケージ化されている: `core/` (types/labels/splits/temperature/probabilities = 最下層) → `model/` (registry/_artifacts_nn + NN 実装 net/loss/dataset/preprocess) → `training/` (train_nn) / `inference/` (predict) / `betting/` (odds/strategy) / `simulation/` (engine/persistence) / `evaluation/` (backtest)。NN 専用化で旧 `ai/nn/` は廃止。`features/` は公開オーケストレータ `builder.py` と per-domain 抽出器 `features/extractors/*` に分離。

### 推論パスは bundle 経由 (NN 専用)
- `registry.load_model_full(path)` が `ModelBundle` (`model_type="nn"`) を返す。`bundle.nn_model` が RaceTransformerModel、`nn_preprocessor` / `temperature_scaler` は optional
- **アーキは単一 (target)**: ability エンコーダ = 集約特徴 + per-race 履歴 GRU (odds は含めない)、value = スコア head で標準化済み odds を concat (ability→value 分離)。`history_feat_dim`/`odds_feat_dim` は *次元* で、0 ならその入力なし (`odds_feat_dim=0` は `KEIBA_EXCLUDE_ODDS_FEATURES`)。旧 v1/v2 や gated フラグ (`use_history`/`use_odds_head`/`arch_version`) は全廃済み
- 推論は **bundle-aware の `predict_race(bundle, frame)` / `predict_race_with_combinations(bundle, frame, ...)` / `predict_race_with_shap(bundle, frame, ...)`** を必ず使う (`ai/inference/predict.py`)
- **`session=` を必ず渡す**。`history_feat_dim>0` のモデル (現行 active を含む) は推論時に過去走系列を DB から引くので、`session=None` だと履歴が zero に degrade する。着順精度はほぼ変わらないのに **単勝回収率が 0.912 → 0.823 に落ちる** (実測・test 19ヶ月 5,404レース)。呼び出し側は必ずセッションをループの外で開いたまま保持すること
- combo 確率は NN スコア → 解析的 Plackett-Luce で導出する。外部 isotonic 校正 (`combo_calibrators` / win / place) は全廃済み。連系の校正は `combo_nll` / `multi` 損失で **NN 内部に学習**させる (下記「連系の校正を NN 内部へ」)。combo 確率は解析的 PL の出力をそのまま使う (外部後処理なし)
- SHAP は廃止。`predict_race_with_shap` は `top_features=[]` を返す (ルーター互換のための残置スタブ)

### NN は optional dep
`torch` / `lightning` は `pyproject.toml` の `[project.optional-dependencies].nn`。未インストール環境では `load_model_full` / 予測系が `ModuleNotFoundError`。導入は `uv pip install -e ".[nn]"`。scraper/ingest だけなら torch 不要。

### モデル保存レイアウト
- NN: `data/models/<YYYYMMDDTHHMMSS>-nn/` に `model.pt` + `meta.json` (`model_type="nn"`) + optional `preprocessor.pkl` / `temperature_scaler.pkl` / `history_norm.pkl` / `speed_figure.pkl`
- active は `model_runs.is_active` で管理。`registry._resolve_model_path` が **basename 比較** でパス表記差 (WSL/Windows) を吸収する

### 特徴量とリーク防止
- `features/builder.py` の `FEATURE_COLUMNS` (46 列) が単一の真実
- ID 系 (`horse_id` / `jockey_id` / `trainer_id`) は **絶対に FEATURE_COLUMNS に入れない**
- `build_training_frame` / `build_inference_frame` ともに **race_date より厳密に過去** の情報しか参照しない (`_build_entry_row` 内で SQL 条件)。新規特徴量を足すときも同じ制約を維持すること
- `KEIBA_EXCLUDE_ODDS_FEATURES=1` で `ODDS_FEATURE_COLUMNS` を除外した特徴量リストになる (オッズ未確定時の検証用)
- 実験用の特徴量ノブ (すべて **default-off・inert**、A/B で本番 ROI 改善せず=市場効率の壁。`KEIBA_RELATIVE_FORM` は 2026-08-30 に log-loss で否定: paired 5,390 レースで +0.0022 [−0.0016,+0.0062]): `KEIBA_MISSING_INDICATORS` (欠損flag) / `KEIBA_LOG_FEATURES` (log変換) / `KEIBA_SPEED_FIGURE` (par-time タイム指数, 履歴17次元 + `speed_figure.pkl` artifact) / `KEIBA_PACE_FEATURES` (ペース想定) / `KEIBA_RELATIVE_FORM` (上がり・走破時計のレース内偏差と順位パーセンタイル)。harness `scripts/model_side_ab.py`、詳細 `docs/ai-model.md`「実験ノブ」と `docs/archive/2026.md`。有効化する場合は学習と推論で同じフラグを揃えること (列構成が変わるため)。**併せて `KEIBA_DISABLE_FRAME_CACHE=1` を渡す**: `_frame_cache_key` は DB パス + 行数シグネチャ + 期間だけで作られ**特徴量フラグを含まない**ので、既存の cache を掴むと新しい列が無いまま学習が走る。`train_nn` は frame に無い列を黙って落とすため**エラーにならずベースラインとして完走する** (2026-08-30 に実際に踏んだ)

### DB スキーマの方針
- `race_id` / `horse_id` / `jockey_id` / `trainer_id` は **TEXT**。netkeiba ID は構造化文字列 (年+回+場+日+R) で算術対象ではない
- Alembic は `migrations/versions/0001` ~ `0014` (最新: `0014_bet_record_conditions`)
- FK CASCADE: `entries.race_id` / `payouts.race_id` は CASCADE、`horse_id` は RESTRICT、`jockey_id` / `trainer_id` は SET NULL

### ジョブはインメモリ
- `api/jobs.py` の `JobRegistry` が `asyncio.create_task` でバックグラウンド実行を管理
- **プロセス再起動でジョブ状態は消える**。永続化を増やすときは明示的にユーザー合意を取る
- `/api/scraper/run` と `/api/models/train` は 202 Accepted を即時返却

### スクレイパー停止
`scraper/stop_flag.py` が「環境変数 `KEIBA_SCRAPER_STOP=1` か、プロセス内フラグ」をループのたびに見る。UI / API / 環境変数の 3 経路で止められる。新しいループを書くときも `is_stopped()` を呼ぶこと。robots.txt は fail-closed (取得失敗 = 拒否、10 分後に再試行)。

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
- NN の損失は本番 **`multi` (default)** / `log_growth` / `combo_nll` / `plackett_luce` の **4 種**を `--loss` で選択し `meta.json.loss_type` に記録 (旧 `log_growth_place` / `log_growth_combo` / `listmle` / `time_margin` は廃止・存在しない)。加えて実験用 `kelly_deploy` (デプロイ整合 Kelly: EV>0のみ・棄権・edge比例ステークを微分可能化) が `--loss` に在るが、A/B で本番 tansho ROI は log_growth 未満 (−0.06) と判明・本番非採用 (着順精度は高い。詳細 `docs/ai-model.md`「実験ノブ」と `docs/archive/2026.md`)。既定は ROI 志向 (decision-focused): `log_growth` は実オッズの単勝回収率を log-growth で直接最適化し、モデル選択は `--monitor valid_tansho_roi`。`log_growth` の `cash_fraction` (旧 `kelly_fraction`, 0.25 固定) は **賭け金の Kelly ではなく odds を勾配に残す cash 項**で、1.0 にすると勾配が勝ち馬クロスエントロピーと一致し ROI 志向でなくなる。**賭け金・評価側の Kelly は全廃済み** (`backtest.py` は 1 点定額のみ。`kelly_bet_size` / `--bet-sizing` / `--bankroll` は存在しない)。複勝専用の回収率損失 `place_growth` (+ `--place-temp`) と、`plackett_luce` の top-k 打ち切り (`--pl-top-k`) を 2026-08-27 に追加したが、**どちらも回収率は改善しない**。place_growth は狙いどおりモデルを堅い馬寄りに変える (選ぶ馬の 2/3 が変化・平均オッズ 7.74→6.09) が、増えた的中が小さくなった配当で相殺される。top-k は log-loss 0.5125→0.5080 / valid NDCG@3 0.5831→0.6030 と確率と順位を改善するが、回収率への効果は検出できない。両方 default-off。最良構成は **二段階** (`plackett_luce` 事前学習 → `--init-from <model_dir>` で `multi` に fine-tune)。OOS で単複ROIは順位損失・市場の人気1番を有意に上回るが依然 <1.0 (詳細 `docs/ai-model.md`)
- **連系の校正を NN 内部へ**: `combo_nll`=連系 combo確率の **NLL (proper scoring rule)** で **外部 isotonic 校正を不要にする** (旧 `combo_calibrators` の代替, `--combo-bet-type` で対象連系・`all` で全連系), `multi`=`log_growth`+`combo_weight`·`combo_nll` の **全馬券対応の本番目的** (`--combo-weight` 既定0.01)。解析ヘルパ `_pl_exacta`/`_pl_trifecta`/`_winning_combo_prob` (`ai/model/loss.py`)。注: 連系は控除率25%で校正しても黒字化はしない
- **温度スケーラは NLL 較正** (`TemperatureScaler.fit_calibration`、勝ち馬の負の対数尤度最小化)。単勝 softmax と複勝 PL に**同じ T** を使うので確率が互いに矛盾しない。旧実装の payback グリッド探索は T をグリッド端に張り付かせ (`T_win=0.133` / `T_place=10.0`)、`win_prob` が 1 位に 0.999999 乗る = 画面に「単勝確率 100.0%」と出る壊れ方をしていた。**賭ける/賭けないは温度ではなく買い方のルールで表現する**
- **EV (期待値) はどの券種でも買う/買わないの判定に使わない** (2026-08-28 に全廃)。買い目は「オッズが取れるものを、券種の優先度 (単勝→複勝→連系) → 的中確率の順に、予算の限り」選ぶ。単勝はオッズ下限 `win_min_odds` のみ。実測 (test 19ヶ月): 単勝 EV条件 0.698 / 本命買い **0.931**、複勝 EV条件 0.654 / 本命買い **0.887**、連系 EV条件 0.849 / 確率順 0.877 (ただし**前進検証 9 fold では差が検出できない**: 0.832 vs 0.839。連系で EV を戻さないのは回収率ではなく設計上の理由 — 根拠の無い閾値・壊れた入力確率・単複との一貫性)。特に複勝は **EV の順序が逆** (EV 0.0-0.9 帯が 0.832 で最良、2.0 以上が 0.573 で最悪) なので温度・冪などの単調変換では直らない。`min_ev` / `win_ev_threshold` / `place_ev_threshold` は設定に存在しない (backtest の `LEGACY_*_EV_THRESHOLD` は旧ルール `--win-bet-rule ev` 専用の分析用途)。backtest の既定は `--win-bet-rule top1 --place-bet-rule topk --place-top-k 1`。backtest は `log_loss` / `market_log_loss` (本命の二値 NLL と市場 1/オッズ の同じ量) と評価窓 `eval_start`/`eval_end` も `metrics_json` に書く。active は 0.574 対 市場 0.483 で**確率としては市場に負けている** (回収率では勝っている)。画面は `metrics_json` に `payback_win` があるかで「実測 / 学習時」を判別し、ラベルを変える (複勝的中率は出所で別の量になるため)
- **運用モデルは 2 つ**。買い目を決めるのは `model_runs.is_active` の active、確からしさを答えるのは `settings.probability_model_path` の確率モデル (`--loss plackett_luce --pl-top-k 5` で学習)。**確率モデルに馬を選ばせない** — 的中率は上がるが人気馬に寄って回収率が落ちる (単勝 0.824 / 複勝 0.881)。用途は (a) 複勝を買うかの判定 (`place_min_confidence`、既定 0.30)、(b) 連系の確率。理由は active の確率が壊れているため (本命の win_prob と勝敗の相関 0.073 / 市場は 0.354。ROI 志向の損失は順序しか最適化しない)。実装は `ai/inference/confidence.py` と `merge_combination_sources` (単複の候補は必ず active 側を使う)。未設定でも動く (複勝は全レース購入・連系の確率は active 由来)
- 確率モデルの割り当ては **Dashboard のモデル一覧**の行アクション (モデル画面は Dashboard に統合済み・旧 `/models` は redirect)。Settings には無い。使用中のモデルは削除できない (409)。シミュレーションの実行条件は `simulation_runs.conditions_json` に残るので、設定を変えて回し直しても後から見分けられる
- **賭け金は EV 順に並べない**。単勝 → 複勝 → 連系の順で、同券種内は的中確率順。較正後は単勝の EV が 0.6 前後で連系 (EV 5〜9) より低く出るため、EV 順だと予算が足りないときに**回収率の推定が最も確かな単複が真っ先に切り捨てられる** (実測: 2,034 レースで単勝 3 点・複勝 1 点)。1 点あたりの金額も券種別 (`stake_units`、既定 単勝500/複勝500/連系100)。定額設定で測り直した実測では連系は 5 券種とも 0.80〜0.88 (ワイド 0.880 / 三連単 0.797) で単複 (0.931 / 0.885) に劣るため(旧記述の「連系は測定不能」は破産する複利設定の産物で誤り)
- ROI系損失・監視・温度スケーラは **標準化前の生オッズ**を使う必要があるため `odds_win_raw`(単勝) と `place_ret_raw`(複勝) を非特徴列として dataset/collate に通す (連系の払戻は通していない) (`odds_win` は特徴量で標準化される)。win_prob は softmax(score / T_win)、place_prob は PL Monte Carlo。combo確率は素の PL Monte Carlo (外部 isotonic 校正は全廃済み。`combo_nll`/`multi` 学習で NN 内部に校正が入る)。新しい損失を足すときも `predict_race` の確率変換は共通なので学習側だけ拡張すれば足りる
