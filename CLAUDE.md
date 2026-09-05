# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

netkeiba スクレイピング + NN (Set Transformer ランキング) による競馬予想ツール (個人研究用)。FastAPI バックエンド + React 管理画面の単一リポジトリで、`scripts/dev.sh` が両方を一発起動する。

詳しい仕様は `docs/` 配下 (`spec.md` / `design.md` / `ai-model.md` / `data-pipeline.md` / `operations.md`) を参照。本ファイルは「コード全体を読まないと掴めない big picture」のみを要約する。環境固有の運用メモは `CLAUDE.local.md` (gitignored) に分離している。

## 日常コマンド

### 起動
```bash
bash scripts/dev.sh   # uv sync + alembic upgrade + (必要なら) pnpm install + uvicorn(:8765) + Vite(:5173)
```
PR 取り込み直後でもこれ一本で動く。`pnpm install` は **lockfile が `node_modules` より新しいときだけ** (毎回走らせると Windows でファイルロックを踏む)、`uv sync` は**失敗しても警告して続行**する。Ctrl-C で `trap 'kill 0' EXIT` が全プロセス停止。

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
取り込み・学習・評価・バックアップ・migration のコマンドは `docs/operations.md`。二段階学習と `--persist` を付ける理由もそこにある。

## アーキテクチャ要点

設計の説明は `docs/design.md`、モデルの中身は `docs/ai-model.md`、API と DB は `docs/spec.md`。ここには**知らないと壊すもの**だけ置く。

- **層をまたぐ向きは `api → jobs → ai / features / scraper → db`。** `ai` は `scraper` を直接呼ばない (循環禁止)、`api/routers/*.py` はロジックを持たない。機械が見張る形は `.claude/policy.yml` の `code.layers`
- **推論は `predict_race` 系 (`ai/inference/predict.py`) を bundle 込みで呼び、`session=` を必ず渡す。** 履歴を DB から引くので、渡さないと zero に degrade する。着順精度はほぼ変わらないのに**単勝回収率が 0.912 → 0.823 に落ちる**ため、テストでは気づけない。セッションはループの外で開いたまま保持すること
- **SHAP は廃止。**`predict_race_with_shap` は `top_features=[]` を返すだけの残置スタブ (ルーター互換のため消していない)
- **`torch` / `lightning` は optional dep** (`pyproject.toml` の `[project.optional-dependencies].nn`)。未インストール環境では `load_model_full` / 予測系が `ModuleNotFoundError`。導入は `uv pip install -e ".[nn]"`。scraper/ingest だけなら不要
- **モデルパスの解決は basename 比較** (`registry._resolve_model_path`)。WSL と Windows でパス表記が違っても同じモデルを指せるようにするため。active は `model_runs.is_active`
- **ID 系 (`horse_id` / `jockey_id` / `trainer_id`) を `FEATURE_COLUMNS` に入れない。**暗記でリークする。`features/builder.py` の 46 列が単一の真実
- **特徴量は race_date より厳密に過去の情報しか参照しない** (`_build_entry_row` 内の SQL 条件)。新しい特徴量を足すときも同じ制約を保つこと
- **実験用の特徴量ノブを有効にするときは `KEIBA_DISABLE_FRAME_CACHE=1` を併せて渡す。**`_frame_cache_key` は特徴量フラグを含まないので、既存 cache を掴むと新しい列が無いまま学習が走り、**エラーにならずベースラインとして完走する** (2026-08-30 に実際に踏んだ)。ノブの一覧と A/B の結果は `docs/ai-model.md`「実験ノブ」
- **`race_id` / `horse_id` / `jockey_id` / `trainer_id` は TEXT。**netkeiba の ID は年+回+場+日+R の構造化文字列で、算術の対象ではない
- **ジョブはインメモリ** (`api/jobs.py` の `JobRegistry` が `asyncio.create_task` で管理)。**プロセス再起動で状態が消える。**永続化を増やすときは明示的に合意を取る
- **新しいスクレイピングのループを書いたら `is_stopped()` を呼ぶ** (`scraper/stop_flag.py`)。呼ばないと UI / API / 環境変数 `KEIBA_SCRAPER_STOP=1` の 3 経路から止められなくなる。robots.txt は fail-closed (取得失敗 = 拒否)
- **uvicorn は `127.0.0.1` のみにバインドし、認証は無い** (ローカル単体起動前提)。CORS 許可は Vite dev と環境変数 `KEIBA_CORS_EXTRA` の追加分だけ
- **`shadcn` CLI は走らせない。**`src/components/ui/` に手書き配置する (`components.json` は設定の記録のみ)。**見た目の規定 (配色・字の尺度・余白・角丸・状態) は [docs/design.md](docs/design.md)「フロントエンド スタイル設計」が正本。**`tailwind.config.ts` は `fontSize` と `borderRadius` を *差し替え* ているので、規定に無いクラス (`text-base` / `rounded-lg` 等) は**エラーにならず無効になる**


## 注意ポイント

- 環境変数 `KEIBA_DATA_DIR` で `data/` の場所を切り替え可能。テストでは `tmp_path` ベースで上書きする (`conftest.py` 参照)
- `core/paths.py` の `data_dir()` を経由してパスを組み立てる。`data/` 配下の直書きは避ける
- NN の損失は本番 **`multi` (default)** / `log_growth` / `combo_nll` / `plackett_luce` の **4 種**を `--loss` で選択し `meta.json.loss_type` に記録 (旧 `log_growth_place` / `log_growth_combo` / `listmle` / `time_margin` は廃止・存在しない)。加えて実験用 `kelly_deploy` (デプロイ整合 Kelly: EV>0のみ・棄権・edge比例ステークを微分可能化) が `--loss` に在るが、A/B で本番 tansho ROI は log_growth 未満 (−0.06) と判明・本番非採用 (着順精度は高い。詳細 `docs/ai-model.md`「実験ノブ」と `docs/archive/2026.md`)。既定は ROI 志向 (decision-focused): `log_growth` は実オッズの単勝回収率を log-growth で直接最適化し、モデル選択は `--monitor valid_tansho_roi`。`log_growth` の `cash_fraction` (旧 `kelly_fraction`, 0.25 固定) は **賭け金の Kelly ではなく odds を勾配に残す cash 項**で、1.0 にすると勾配が勝ち馬クロスエントロピーと一致し ROI 志向でなくなる。**賭け金・評価側の Kelly は全廃済み** (`backtest.py` は 1 点定額のみ。`kelly_bet_size` / `--bet-sizing` / `--bankroll` は存在しない)。複勝専用の回収率損失 `place_growth` (+ `--place-temp`) と、`plackett_luce` の top-k 打ち切り (`--pl-top-k`) を 2026-08-27 に追加したが、**どちらも回収率は改善しない**。place_growth は狙いどおりモデルを堅い馬寄りに変える (選ぶ馬の 2/3 が変化・平均オッズ 7.74→6.09) が、増えた的中が小さくなった配当で相殺される。top-k は log-loss 0.5125→0.5080 / valid NDCG@3 0.5831→0.6030 と確率と順位を改善するが、回収率への効果は検出できない。両方 default-off。最良構成は **二段階** (`plackett_luce` 事前学習 → `--init-from <model_dir>` で `multi` に fine-tune)。OOS で単複ROIは順位損失・市場の人気1番を有意に上回るが依然 <1.0 (詳細 `docs/ai-model.md`)
- **連系の校正を NN 内部へ**: `combo_nll`=連系 combo確率の **NLL (proper scoring rule)** で **外部 isotonic 校正を不要にする** (旧 `combo_calibrators` の代替, `--combo-bet-type` で対象連系・`all` で全連系), `multi`=`log_growth`+`combo_weight`·`combo_nll` の **全馬券対応の本番目的** (`--combo-weight` 既定0.01)。解析ヘルパ `_pl_exacta`/`_pl_trifecta`/`_winning_combo_prob` (`ai/model/loss.py`)。注: 連系は控除率25%で校正しても黒字化はしない
- **`src/main.py` は wheel に *コピー* される** (`[tool.hatch.build.targets.wheel.force-include]`)。パッケージ本体は editable でリンクされるが main.py だけは複製なので、**router を足しても `uv run pytest` からは 404 のまま**になる (テストは `.venv/Lib/site-packages/main.py` を import する)。`uv pip install -e . --no-deps --offline` で入れ直すこと。**`--offline` が要る**のは Norton の TLS 傍受で pypi の証明書検証が落ちるため (`invalid peer certificate: UnknownIssuer`)。依存が揃っている限りこれで通る
- **`payouts.combo` は表記が違う**。netkeiba 由来なので `1 - 10` / `7 → 1` と**区切りの前後に空白が入る**が、買い目 (`bet_records.combo` / 推奨) は空白なしの `1-10` / `7→1`。素の文字列比較では**連系が 1 件も一致しない**ので、突き合わせる側は必ず `core.bet_types.normalize_combo` を通す。`ai/betting/odds.py` は元から正規化していたが `services/bet_settlement.py` は素の `==` で、**記録した連系がすべて外れとして確定する**状態だった (2026-09-01 に修正。既存テストが空白なしの偽データを使っていて検出できていなかった)
- **設定の応答は `_dict_to_response` を通る**。ここでキー名を間違えても pydantic は未知の引数を黙って捨てるので、**保存した値が GET/PUT の応答に出ず既定値が返る** (画面上は「保存したのに戻る」)。旧名 `place_min_confidence=` のままだった `place_min_hit_prob` と `max_points_per_bet_type` が実際にこれで死んでいた
- **温度スケーラは NLL 較正** (`TemperatureScaler.fit_calibration`、勝ち馬の負の対数尤度最小化)。単勝 softmax と複勝 PL に**同じ T** を使うので確率が互いに矛盾しない。旧実装の payback グリッド探索は T をグリッド端に張り付かせ (`T_win=0.133` / `T_place=10.0`)、`win_prob` が 1 位に 0.999999 乗る = 画面に「単勝確率 100.0%」と出る壊れ方をしていた。**賭ける/賭けないは温度ではなく買い方のルールで表現する**
- **EV (期待値) はどの券種でも買う/買わないの判定に使わない** (2026-08-28 に全廃)。買い目は「オッズが取れるものを、券種の優先度 (単勝→複勝→連系) → 的中確率の順に、予算の限り」選ぶ。単勝はオッズ下限 `win_min_odds` のみ。**連系の点数は的中確率の下限 `combo_min_hit_prob` だけで決まり、券種ごとの上限は持たない** (2026-09-01 に廃止) ので、点数はレースごとに変わる。`min_ev` / `win_ev_threshold` / `place_ev_threshold` は設定に存在しない。**戻す提案をする前に `docs/ai-model.md`「推奨ベットルール」を読む** — 券種ごとの実測と、回収率では正当化できないのに戻さない理由がそこにある
- **backtest で `--start` を省くと DB 全期間が窓になり、学習期間を含んだ in-sample の値が出る** (確率モデルが実際にそうなっていて、単勝 0.945 と表示されていたが正しくは 0.821)。いまは `eval_overlaps_train` を検出して `--persist` を弾く (`--allow-in-sample` で明示的にのみ通す)。既定は `--win-bet-rule top1 --place-bet-rule topk --place-top-k 1`、`--bootstrap-iters` は 2000 で 95% 区間を `metrics_json` に書く
- **回収率の数字を CLAUDE.md に写さない。**測り直すたびに動くので、片方だけ古くなる。窓 (期間・レース数) と 95% 区間つきの正本は `docs/ai-model.md`「OOS 実測」。画面は `metrics_json` に `payback_win` があるかで「実測 / 学習時」を判別し、ラベルを変える (複勝的中率は出所で別の量になるため)
- **シミュレーションは RACE 画面と同じ仕組みで回す** (2026-09-01 に統一)。入力は **1 レースに使う上限 (`race_budget`) だけ**で、初期資産・賭け金の決め方 (定額/複利)・戦略プリセット (conservative/balanced/aggressive)・狙い方 (`top_n_horses`)・履歴の無いレースの除外 (`exclude_low_information`) は**すべて廃止**した (`STRATEGY_PRESETS` / `StakingMode` は存在しない)。賭け金は残高に依存しないので破産が起きず、評価が途中で止まらない。結果は資産残高ではなく **0 から始まる累計損益** (`final_profit` / `peak_profit` / `trough_profit` / `profit_timeseries`)。DB は migration 0015 で列名ごと置き換え、旧ルールで走った 8 件は削除した (列名だけ変えても数字の意味が変わらないため)
- **狙い方 (上位何頭で買い目を組むか) は選択肢にしない**。買うかどうかは的中確率の下限が決めるので、頭数を広げても線を超えない候補が増えるだけで買い目は変わらない。`ai.betting.strategy.TOP_N_HORSES` (=3) 固定で、API のクエリからも UI からも外した
- **運用モデルは 2 つ**。買い目を決めるのは `model_runs.is_active` の active、確からしさを答えるのは `settings.probability_model_path` の確率モデル (`--loss plackett_luce --pl-top-k 5` で学習)。**確率モデルに馬を選ばせない** — 的中率は上がるが人気馬に寄って回収率が落ちる (単勝 0.824 / 複勝 0.881)。用途は (a) 複勝を買うかの判定と厚み (`place_min_hit_prob`、既定 0.60 = 3着内率)、(b) 連系の確率。**確信度は券種横断で「その買い目が当たる確率」**に統一 (単勝=1着確率 / 複勝=3着内率 / 連系=組合せの的中確率)。**点数は単複とも確信度で動く**が (式は次項)、**回収率が上がるのは複勝だけ** — 単勝は的中率が 6%→37% と動くのに回収率が動かない (相関 −0.005)。連系も無相関。理由は active の確率が壊れているため (本命の win_prob と勝敗の相関 0.073 / 市場は 0.354。ROI 志向の損失は順序しか最適化しない)。実装は `ai/inference/confidence.py` と `merge_combination_sources` (単複の候補は必ず active 側を使う)。未設定でも動く (複勝は全レース購入・連系の確率は active 由来)
- 確率モデルの割り当ては **Dashboard のモデル一覧**の行アクション (モデル画面は Dashboard に統合済み・旧 `/models` は redirect)。Settings には無い。使用中のモデルは削除できない (409)。シミュレーションの実行条件は `simulation_runs.conditions_json` に残るので、設定を変えて回し直しても後から見分けられる
- **賭け金は「1 点 = 100 円 × 点数」だけ**。券種ごとの 1 点あたり金額 (旧 `stake_units`) とふだん買う券種 (旧 `enabled_bet_types`) は設定から廃止した (2026-09-01)。厚みは金額ではなく**点数**で表し、点数は確信度が決める: 単複は `points_for_confidence` (`base 5 × (確信度/基準)^2` を 1〜15 点、基準は単勝 0.25 / 複勝 0.50)、連系は 1 組合せ 1 点で**何点買うかは的中確率の下限**が決める。単勝を確信度で動かしても回収率はほぼ変わらない(OOF 14,829 レース: 定額 5 点 0.8438 → 確信度連動 0.8483、fold の幅は 0.767〜0.955 → 0.776〜0.913 と狭まる)。**賭け金は EV 順に並べない** — 単勝 → 複勝 → 連系の順で、同券種内は的中確率順。較正後は単勝の EV が 0.6 前後で連系 (EV 5〜9) より低く出るため、EV 順だと予算が足りないときに**回収率の推定が最も確かな単複が真っ先に切り捨てられる** (実測: 2,034 レースで単勝 3 点・複勝 1 点)。定額設定で測り直した実測では連系は 5 券種とも単複に劣る(旧記述の「連系は測定不能」は破産する複利設定の産物で誤り)
- ROI系損失・監視・温度スケーラは **標準化前の生オッズ**を使う必要があるため `odds_win_raw`(単勝) と `place_ret_raw`(複勝) を非特徴列として dataset/collate に通す (連系の払戻は通していない) (`odds_win` は特徴量で標準化される)。win_prob は softmax(score / T_win)、place_prob は PL Monte Carlo。combo確率は素の PL Monte Carlo (外部 isotonic 校正は全廃済み。`combo_nll`/`multi` 学習で NN 内部に校正が入る)。新しい損失を足すときも `predict_race` の確率変換は共通なので学習側だけ拡張すれば足りる

<!-- claude-keeper:generated -->
## 体制

規約は `.claude/policy.yml`。役に読ませるときは**会話の経緯を渡さない** (skill `flat-view`)。「いまは直さない」は `.claude/judgments.yml` に書く。コマンドは `/standup` (全役) / `/docs` / `/code` / `/critique` / `/design` / `/ship` / `/routine`。

| 役 | いつ呼ぶか |
|---|---|
| `docs-auditor` | docs か実装を直したあと。docs と実装の食い違いを見る |
| `duplication-auditor` | docs を切り出したあと。文書どうしの二重管理を見る (実装は開かない) |
| `code-steward` | 実装が一区切りついたとき。伸びた・散った・溜まったを見る |
| `critic` | 実験の結論を出したあと。**窓が短くて出た差ではないか**を疑う |
| `reviewer` | コミットする前。差分を通しで読む |
<!-- /claude-keeper:generated -->
