# keiba-ai backend

netkeiba スクレイパー + LightGBM 予測バックエンド（個人研究用）。

## セットアップ

```bash
cd backend
uv sync
```

## データ取り込み

```bash
uv run python -m keiba_ai.jobs.ingest --date 2024-12-28
uv run python -m keiba_ai.jobs.ingest --date 2024-12-28 --limit 3   # 先頭 3 レースのみ
```

## テスト実行

```bash
uv run pytest
```

## 環境変数

| 変数 | デフォルト | 説明 |
|---|---|---|
| `KEIBA_USER_AGENT` | (研究用 UA 文字列) | HTTP リクエストの User-Agent |
| `KEIBA_RATE_MIN_SECONDS` | `3.0` | リクエスト間隔の最小値（秒） |
| `KEIBA_RATE_MAX_SECONDS` | `6.0` | リクエスト間隔の最大値（秒） |
| `KEIBA_NIGHT_MIN_SECONDS` | `5.0` | 深夜帯（22:00-05:00 JST）の最小待機秒数 |
| `KEIBA_DATA_DIR` | `data/` | データディレクトリ（リポジトリ相対） |
| `KEIBA_SCRAPER_STOP` | - | `1` を設定するとスクレイパーを即時停止 |
