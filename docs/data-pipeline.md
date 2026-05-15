# KEIBA AI — データパイプライン仕様書

関連ドキュメント: [spec.md](spec.md) / [design.md](design.md) / [operations.md](operations.md)

---

## 対象 URL とパース対象

スクレイピング対象は netkeiba の以下 URL 群。

| 種別 | URL パターン | 取得対象 |
|---|---|---|
| 開催日カレンダー | `https://db.netkeiba.com/race/list/<YYYYMMDD>/` | 開催レース ID 一覧 |
| レース結果 | `https://db.netkeiba.com/race/<race_id>/` | 着順・タイム・払戻金・上がり3F・通過順・馬名・騎手名・調教師名 |
| 馬詳細 | `https://db.netkeiba.com/horse/<horse_id>/` | name / sex / birth_date（新規 horse のみ取得） |
| 馬血統 | `https://db.netkeiba.com/horse/ped/<horse_id>/` | sire（父馬名）/ dam（母馬名）（新規 horse のみ取得） |

騎手・調教師の詳細ページは現状未実装。名前は race_result HTML から取得し COALESCE upsert している。

### 取得対象範囲

デフォルトでは **中央競馬（JRA）のレースのみ** ingest する。race_id の 5-6 桁目がトラックコード `01`〜`10`（札幌〜小倉の JRA 10 場）に一致するものを取得し、地方競馬（NAR）のコード（大井: 44、金沢: 46、高知: 54、佐賀: 55 等）は除外する。

地方を含めたい場合は環境変数 `KEIBA_INCLUDE_NAR=1` を設定して `ingest_range` / `keiba-ingest` を実行する。

| モード | race_id 5-6 桁目の範囲 | 1 開催日あたりのレース数（目安） | 想定 Phase 1 所要時間 |
|---|---|---|---|
| デフォルト（中央のみ） | `01`〜`10` | 約 36 | 3〜5 時間 |
| NAR 含む（opt-in） | `01`〜`10` + 地方コード | 約 92 | 8〜12 時間 |

中央レースが 0 件の日は `ParseError` を送出する。地方をスキップした件数は `INFO` ログに出力される。

### 主なパース対象（実装済みセレクタ）

| ページ | パース対象 | 実装セレクタ | 備考 |
|---|---|---|---|
| 開催日カレンダー | レース ID 一覧 | `<a href>` の `?race_id=<12桁>` または `/race/<12桁>` を正規表現で抽出し、JRA トラックコード（5-6 桁目 `01`〜`10`）でフィルタ。`KEIBA_INCLUDE_NAR=1` のとき地方コードも含める | `db.netkeiba.com/race/list/YYYYMMDD/` から取得 |
| レース結果ヘッダ | 距離・天候・馬場・競馬場 | ページ全文を正規表現スキャン（class 名変動対応）、競馬場は race_id 5-6 桁から導出 | 実 HTML 検証済み |
| レース結果テーブル | 着順・タイム・馬体重・上がり3F・通過順・馬名・騎手名・調教師名 | `class` が `race_table_01` または `RaceTable` に一致する `<table>`、列名ヘッダを辞書で参照 | EUC-JP 強制デコード |
| 払戻金 | 単勝・複勝 | `class="pay_table_01"` の `<table>` 内、`class="txt_r"` の `<td>` から金額を抽出。複勝は 1 セル内 `<br>` 区切り | EUC-JP 強制デコード |
| レース結果ヘッダ（race_class） | レースクラス | `class="RaceData02"` の `<span>` 群 → `class="RaceName"` 要素 → ページ全文 の順で fallback し `_GRADE_RE`（GⅠ/GⅡ/GⅢ/G1/G2/G3/Listed/L/OP/重賞）をマッチ | 複数パスで fallback するため HTML 構造変動に強い |
| 馬詳細 | name / sex / birth_date | `<title>` から馬名（正規表現）、`class="horse_title"` から性別、`class="db_prof_table"` の「生年月日」行から誕生日 | 実 HTML 検証済み |
| 馬血統 | sire / dam | `class` が `blood_table` に一致する `<table>` から最大 rowspan の TD 2 件を父馬・母馬として抽出 | 10 文字英数字対応 |

セレクタが実際の HTML と一致しない場合は `ParseError` を発生させてログに記録するため、破損データの混入は防がれる。

---

## レート制御ポリシー

netkeiba へのアクセスは以下のルールを厳守する。

| 項目 | 値 |
|---|---|
| 最小間隔 | 3 秒 |
| ジッター範囲 | 3〜6 秒（一様乱数で加算） |
| 深夜帯（0〜6 時 JST）| 5 秒以上 |
| 並列リクエスト数 | 1（直列のみ） |

```python
# レート制御の疑似コード
import random, asyncio

async def rate_limited_get(url: str) -> str:
    wait = 3.0 + random.uniform(0, 3.0)
    if is_midnight_jst():
        wait = max(wait, 5.0)
    await asyncio.sleep(wait)
    return await http_client.get(url)
```

### リトライポリシー

| 条件 | 動作 |
|---|---|
| 5xx / タイムアウト | 指数バックオフ: 4s → 8s → 16s → 30s（最大 4 回） |
| 429 Too Many Requests | 60 秒ペナルティ待機後にリトライ |
| 4xx（404 等） | リトライせずエラーとして `scrape_log` に記録 |

---

## robots.txt 遵守ポリシー

1. スクレイピング実行前に対象ドメインの `robots.txt` を取得する
2. 取得結果をドメイン単位でインメモリキャッシュし、24 時間以内は再取得しない
3. `Disallow` に一致するパスへのリクエストは発行しない

`robots.txt` の取得に失敗した場合（ネットワーク障害等）は、警告ログを出力してリクエストを許可する（fail-open 動作）。

```python
# robots.txt フェッチ: fail-open
try:
    rp.read()
except Exception as exc:
    logger.warning("Failed to fetch robots.txt: %s — allowing all requests", exc)
# 失敗してもキャッシュに空の parser を格納し、can_fetch は True を返す
```

`User-Agent` は Settings 画面で設定可能（`PUT /api/settings` で変更できる）。

---

## HTML キャッシュ戦略

### キャッシュパス命名規則

```text
data/raw/<yyyy>/<mm>/<race_id>.html        — レース結果 HTML（race_id の年月から導出）
data/raw/misc/<sha256(url)[:16]>.html      — その他 HTML（馬詳細 / 馬血統 / 開催日カレンダー）
```

`cache.py` の `_cache_path()` は URL 中に 12 桁の race_id が含まれる場合は `<yyyy>/<mm>/` へ、それ以外は `misc/` へ振り分ける。

### 同一性判定

- フェッチ後に `SHA-256(response_body)` を計算し `scrape_log.content_hash` に記録する
- 次回同一 URL を取得する際、既存のキャッシュファイルの content_hash と一致する場合はパースをスキップする（HTTP ステータス 200 でも変更なし扱い）
- `ETag` ヘッダーが返却された場合は `scrape_log.etag` に保存し、次回 `If-None-Match` ヘッダーで送信して 304 を期待する

### キャッシュの保持期間

- `data/raw/<yyyy>/<mm>/`（レース結果 HTML）: 手動削除するまで保持する。parser 修正後に再 parse できるよう意図的に残す
- `data/raw/misc/`（馬詳細 / 馬血統 / カレンダー HTML）: `ingest_range` が各日の ingest 完了直後に `clear_misc_cache()` で自動削除する。これらは一度 DB へ parse されれば再利用の必要がないため、長期 ingest でのディスク肥大を防ぐ
- `KEIBA_KEEP_MISC_CACHE=1` 環境変数を設定するとデバッグ用に misc キャッシュを削除しない（opt-out）
- `data/raw/` は `.gitignore` 対象

---

## 増分取得アルゴリズム

毎回全件を再取得するのではなく、未取得のレース ID のみフェッチする。

```text
1. 取得対象日付リストを決定（--start / --end で指定した範囲）
2. 各日付に対して kaisai_date カレンダーページをフェッチ → レース ID 一覧を取得
3. races テーブルに存在しないレース ID を「未取得」とみなす
4. 未取得レース ID に対して結果ページ・出馬表を順次フェッチ
5. 各ページから horse_id / jockey_id / trainer_id と馬名・騎手名・調教師名を抽出し、
   _ensure_masters で horses / jockeys / trainers に upsert する。
   name は取得できた場合のみ上書きし、既に保存済みの name は COALESCE で保持する
5a. 新規 horse（DB に存在しないか name IS NULL）の場合、追加で以下 2 ページを順次フェッチする。
    - https://db.netkeiba.com/horse/<horse_id>/ → name / sex / birth_date を取得
    - https://db.netkeiba.com/horse/ped/<horse_id>/ → sire / dam を取得
    フェッチ・パースが失敗した場合は warning を記録して続行する（race ingest 全体は止めない）。
    既存 horse（horses.name IS NOT NULL）はスキップし、追加フェッチを行わない。
    COALESCE upsert は 5 列（name / sex / birth_date / sire / dam）に対応している。
6. パース結果を SQLAlchemy Session 経由で DB に保存し、scrape_log に成功ステータスを記録
```

### `ingest_range` CLI

```bash
cd backend

# 指定期間を連続取り込み
uv run python -m keiba_ai.jobs.ingest_range \
    --start 2021-01-01 \
    --end 2025-12-31

# 1 日あたりのレース取り込み上限を指定（動作確認・負荷低減用）
uv run python -m keiba_ai.jobs.ingest_range \
    --start 2021-01-01 \
    --end 2021-01-31 \
    --limit-per-day 3
```

### 大規模取り込み時の運用ノート

- 長期連続稼働（数日〜数週間）になるため、PC が安定して動作できる環境で実行すること
- アクセスが集中する土日の正レース時間帯（午前 10 時〜午後 5 時 JST）はなるべく避け、深夜〜早朝に実行する
- `scrape_log` を定期的に確認し、大量のエラーが発生していないか監視する
- ディスク容量の目安: レース結果 HTML（`data/raw/<yyyy>/<mm>/`）+ SQLite 合計で約 1 GB（5 年分）。misc キャッシュは各日完了後に自動削除されるため、**5 GB 以上**の空きがあれば十分
- 初回取り込み時、新規 horse ごとに詳細ページ・血統ページの 2 ページを追加フェッチするため 1 頭あたり 6〜12 秒（3〜6 秒 × 2 リクエスト）が加算される。5 年分の初回取り込みでは数千頭の新規 horse が登場するため、総所要時間は数時間単位で増加する（同一 horse が複数レースに出走するケースは 2 回目以降スキップされる）

---

## 失敗時のレジューム

- フェッチが失敗したレース ID は `scrape_log` に `status='error'` で記録する
- 次回実行時に `status='error'` かつ `fetched_at < (now - 1h)` のレコードを再試行対象に含める
- `status='ok'` かつ `content_hash` 一致のレコードはスキップする
- `ingest_range` は各日付の ok ログを参照してスキップするため、中断後に同じコマンドを再実行するだけでレジュームできる
- 未取得日数は `GET /api/scraper/status?range=N`（デフォルト 30 日）の `missing_dates_count` で確認できる（ok ログ 0 件の日数をカウント）
- `scrape_log` は UI 監視用途でも参照される。`GET /api/scraper/recent_activity?minutes=N` が直近 N 分のレコードを集計し、status 内訳・rate_per_min・最新 race_id を返す。`ingest_range` を CLI で実行中も UI からリアルタイムに進捗を確認できる（ScraperStatusCard が実行中 5 秒 / アイドル 30 秒間隔でポーリング）
- `scrape_log.fetched_at` には `ix_scrape_log_fetched_at` インデックスが設定されている（migration 0003）。Phase 2 の大規模 ingest で行数が数万に達しても `recent_activity` の `WHERE fetched_at >= cutoff` が full scan にならないよう保護している
- `recent_activity` エンドポイントは直近 N 分を最大 2000 行に制限して取得する（約 3 時間分のピーク fetch 量に相当）。UI 側は集計値と最新の `latest_race_id` のみを参照するため、この上限で実運用上の問題は生じない

---

## 法的・倫理的配慮

本ツールは以下の原則に従う。

1. **個人研究限定**: 取得データ・学習済みモデルを第三者へ提供・公開しない
2. **商用利用禁止**: 本ツールを利用した収益活動は行わない
3. **レート制御徹底**: [レート制御ポリシー](#レート制御ポリシー) に定める間隔を必ず守る
4. **即時停止スイッチ**: 環境変数 `KEIBA_SCRAPER_STOP=1` を設定するとスクレイピングループが次の待機後に停止する（`scraper/stop_flag.py` モジュールが実装）。`POST /api/scraper/stop` でも即時停止可能。スクレイピングの手動実行も `POST /api/scraper/run`（バックグラウンド・202 即時返却）で行える
5. **robots.txt 遵守**: [robots.txt 遵守ポリシー](#robotstxt-遵守ポリシー) を守る
6. **規約変更時の対応**: netkeiba の利用規約変更を検知した場合は即時停止し、対応を検討する（[operations.md 規約上の注意点](operations.md) 参照）

**データの完全性**: horse / jockey / trainer の `name` は race_result HTML から取得し `_ensure_masters` で COALESCE upsert する。新規 horse に限り `birth_date`・`sex`・`sire`・`dam` を馬詳細ページ・血統ページから取得して upsert する。既存 horse（`horses.name IS NOT NULL`）はスキップするため、旧取り込み分で sire/dam が NULL のレコードは引き続き NULL のままとなる。

**特徴量計算とデータ充足の関係**: 特徴量エンジニアリング（`features/` 配下）は horse_id / jockey_id / trainer_id を参照して過去実績を集計する。これらの統計は実データが十分に蓄積されてからでないと意味のある値にならない。

- 馬の直近 5 走平均着順（`recent_avg_finish`）: 対象馬の過去レース記録が 1 件以上ないと `NaN` になる
- 騎手の直近 30 日勝率（`jockey_recent_win_rate`）: 直近 30 日間の騎手出走データが必要。取り込み期間が短いと大半が `NaN` になり、特徴量が機能しない
- 調教師の同競馬場複勝率（`trainer_course_place_rate`）: 同様に取り込み量に依存する
- 直近上がり 3F 平均（`recent_avg_agari_3f`）: entries テーブルの `agari_3f` 列を参照する。旧取り込み分は `agari_3f = NULL` のまま残るため、初期段階では NaN になりやすい
- 血統特徴量（`sire_progeny_win_rate` / `dam_progeny_win_rate`）: horses テーブルの `sire` / `dam` に依存する。旧取り込み分の sire/dam が NULL の馬は NaN になる

初回学習・評価フェーズ（データ取り込み開始から数ヶ月程度）では、これらの特徴量が `NaN` のまま LightGBM に渡されることが多い。モデルは native missing value 処理で対応するが、精度は本番データが蓄積した後の再学習時に向上する。

---

## 運用上の注意

- スクレイピングは netkeiba の利用規約上グレーゾーンであることを認識した上で、節度ある利用を徹底する
- アクセスが集中する土日の正レース時間帯（午前 10 時〜午後 5 時 JST）はなるべく避け、深夜〜早朝に実行する
- `scrape_log` を定期的に確認し、大量のエラーが発生していないか監視する
- netkeiba のサイト構造が変更された場合（HTML セレクタが壊れた場合等）は、スクレイパーを停止して修正する前に全取得を再実行しない

<!-- TODO: サイト構造変更の検知ロジック（パースエラー率の閾値アラート）は未実装 -->
