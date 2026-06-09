# KEIBA AI

netkeiba データを NN（PyTorch Set Transformer ランキング）で学習し、単勝・複勝・各種馬券の買目予想と確率を提示する個人研究用ツール。FastAPI バックエンド + React 管理画面構成で、`scripts/dev.sh` で起動した dev サーバにブラウザでアクセスして利用する。

## 概要

| 項目 | 内容 |
|---|---|
| カテゴリ | Tool / AI（競馬予想支援・個人研究用） |
| バックエンド | FastAPI (Python) + PyTorch NN |
| フロントエンド | React + Vite + TypeScript + shadcn/ui + Tailwind |
| データソース | netkeiba スクレイピング（個人研究範囲・レート制御徹底） |
| DB | SQLite（SQLAlchemy + Alembic） |
| 動作形態 | ローカル dev サーバ + ブラウザアクセス |

## ディレクトリ構造

```text
.
├── backend/        # FastAPI + AI + スクレイパー (Python, uv 管理)
├── frontend/       # React 管理画面 (Vite + TypeScript)
├── scripts/        # dev.sh (uvicorn + Vite 一発起動)
├── data/           # SQLite DB / モデル / raw HTML キャッシュ / ログ (gitignored)
├── data-snapshot/  # データ取り込み中の安全用スナップショット (gitignored)
└── docs/           # 仕様・設計・運用ドキュメント
```

## 開発

```bash
# 開発サーバ起動 (FastAPI on :8765 + Vite on :5173)
bash scripts/dev.sh
# → http://localhost:5173 をブラウザで開く
```

詳細は [docs/operations.md](docs/operations.md) を参照。

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/README.md](docs/README.md) | ドキュメント管理ハブ |
| [docs/spec.md](docs/spec.md) | 技術仕様（スタック・DB・API・開発ビルド手順） |
| [docs/design.md](docs/design.md) | 設計方針（アーキテクチャ・AI モジュール・UI 構成） |
| [docs/data-pipeline.md](docs/data-pipeline.md) | スクレイピング・取り込み仕様 |
| [docs/ai-model.md](docs/ai-model.md) | モデル設計（Set Transformer・損失・確率変換・評価） |
| [docs/operations.md](docs/operations.md) | 運用（セットアップ・再学習サイクル・バックアップ・障害対応） |

## 重要な制約

- **個人研究限定**: 取得データ・学習済みモデルの第三者への提供・公開は行わない
- **netkeiba 規約**: スクレイピングは規約上グレーゾーン。レート制御（最低 3 秒 + ジッター）を徹底し、robots.txt を遵守する
- **即時停止スイッチ**: Settings 画面の停止スイッチ、および `/api/scraper/stop` エンドポイントで任意のタイミングでスクレイピングを止められる
