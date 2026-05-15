# KEIBA AI

netkeiba データを LightGBM で学習し、単勝・複勝の買目予想と確率を提示する個人研究用ツール。FastAPI バックエンド + React 管理画面構成。`scripts/dev.sh` で uvicorn + Vite を一発起動し、ブラウザでアクセスする。

## アプリ概要

| 項目 | 内容 |
|---|---|
| タイトル | KEIBA AI |
| カテゴリ | Tool / AI（競馬予想支援・個人研究用） |
| バックエンド | FastAPI (Python) + LightGBM |
| フロントエンド | React + Vite + TypeScript + shadcn/ui + Tailwind |
| データソース | netkeiba スクレイピング（個人研究範囲・レート制御徹底） |
| DB | SQLite（SQLAlchemy + Alembic） |
| 動作形態 | ローカル dev サーバ（uvicorn + Vite）+ ブラウザアクセス |
| バージョン | 0.1.0 |

## アーキテクチャサマリ

```text
ブラウザ (http://localhost:5173)
    │
    ▼  React 管理画面 (Vite dev server)
    │
    │  HTTP → http://127.0.0.1:8765/api/*
    ▼
FastAPI (uvicorn)
    ├─ スクレイパー (netkeiba)
    ├─ AI 推論 (LightGBM)
    └─ SQLite (data/keiba.db)
```

## ディレクトリ構造

```text
docs/
├── README.md          # このファイル（管理ハブ）
├── spec.md            # 技術仕様（スタック・DB・API・開発ビルド手順）
├── design.md          # 設計方針（アーキテクチャ・AI モジュール・UI 構成）
├── data-pipeline.md   # スクレイピング・取り込み仕様
├── ai-model.md        # モデル設計（LightGBM lambdarank・確率変換・評価）
└── operations.md      # 運用（セットアップ・再学習サイクル・バックアップ・障害対応）
```

## ドキュメント一覧

| ファイル | 概要 | 状態 |
|---|---|---|
| [spec.md](spec.md) | 技術仕様（技術スタック・ディレクトリ構成・DBスキーマ・API エンドポイント・開発ビルド手順） | 骨子完成 |
| [design.md](design.md) | 設計方針（非機能要件・アーキテクチャ図・AI モジュール責務分離・UI 画面構成・状態管理・拡張ポイント） | 骨子完成 |
| [data-pipeline.md](data-pipeline.md) | スクレイピング・取り込み仕様（対象 URL・レート制御・robots.txt 遵守・HTML キャッシュ・増分取得・失敗レジューム・法的配慮） | 骨子完成 |
| [ai-model.md](ai-model.md) | モデル設計（問題定義・LightGBM lambdarank・二値分類・確率変換・特徴量・学習評価フロー・ベットルール） | 骨子完成 |
| [operations.md](operations.md) | 運用（ローカル開発セットアップ・データ取り込み運用・再学習サイクル・モデル世代管理・バックアップ・トラブルシューティング） | 骨子完成 |

## 重要な制約

- **個人研究限定**: 本ツールは個人研究目的のみ。取得データ・学習済みモデルの第三者への提供・公開は行わない
- **netkeiba 規約**: スクレイピングは規約上グレーゾーン。レート制御（最低 3 秒 + ジッター）を徹底し、robots.txt を遵守する
- **即時停止スイッチ**: Settings 画面の停止スイッチ、および `/api/scraper/stop` エンドポイントで任意のタイミングでスクレイピングを止められる
- **規約変更時**: netkeiba の利用規約・robots.txt が変更された場合は即座にスクレイピングを停止し、対応を検討する（[operations.md](operations.md) 参照）
