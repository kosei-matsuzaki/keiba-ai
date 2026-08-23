# docs — KEIBA AI

作品としての説明（概要・見どころ・技術解説）はリポジトリ直下の [README.md](../README.md) にあります。
README の `<!-- portfolio:begin -->` 〜 `<!-- portfolio:end -->` の内側はポートフォリオサイト
（`portfolio-site/src/data/projects.ts`）から生成しているので、直接編集しないでください。

ここは開発者向けの入口です。

| ファイル | 内容 |
| --- | --- |
| [spec.md](spec.md) | 技術仕様（技術スタック・ディレクトリ構成・DB スキーマ・API エンドポイント・開発ビルド手順） |
| [design.md](design.md) | 設計方針（非機能要件・アーキテクチャ図・AI モジュール責務分離・UI 画面構成・状態管理・拡張ポイント） |
| [data-pipeline.md](data-pipeline.md) | スクレイピング・取り込み仕様（対象 URL・レート制御・robots.txt 遵守・HTML キャッシュ・増分取得・失敗レジューム・法的配慮） |
| [ai-model.md](ai-model.md) | モデル設計（問題定義・Set Transformer・損失・確率変換・特徴量・学習評価フロー・ベットルール・10 回の改善実験） |
| [operations.md](operations.md) | 運用（セットアップ・データ取り込み・再学習サイクル・モデル世代管理・バックアップ・トラブルシューティング） |
| `model-explainer.py` / `.mp4` | モデルの計算過程を manim で可視化した解説動画とそのソース |
| `images/` | 画面キャプチャ。直下の README とポートフォリオサイトの両方がここを参照する |

## クイックスタート

前提: [uv](https://docs.astral.sh/uv/) / Node.js 20+ / pnpm がインストール済みであること。

```bash
# 開発サーバ起動 (FastAPI on :8765 + Vite on :5173)
bash scripts/dev.sh
# → http://localhost:5173 をブラウザで開く
```

依存同期・DB migration は `dev.sh` が毎回自動で行います。**スクレイピング済みデータと学習済みモデルはリポジトリに含まれない**ため、初回起動時は空の状態から始まります (画面の取込ボタンまたは `uv run keiba-ingest --date YYYY-MM-DD` でデータ取得)。セットアップの詳細は [docs/operations.md](operations.md) を参照。

## ディレクトリ構造

```text
docs/
├── README.md          # このファイル（管理ハブ）
├── spec.md            # 技術仕様（スタック・DB・API・開発ビルド手順）
├── design.md          # 設計方針（アーキテクチャ・AI モジュール・UI 構成）
├── data-pipeline.md   # スクレイピング・取り込み仕様
├── ai-model.md        # モデル設計（Set Transformer・損失・確率変換・評価）
└── operations.md      # 運用（セットアップ・再学習サイクル・バックアップ・障害対応）
```

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
    ├─ AI 推論 (PyTorch NN)
    └─ SQLite (data/keiba.db + data/odds.db)
```

## 重要な制約

- **個人研究限定**: 本ツールは個人研究目的のみ。取得データ・学習済みモデルの第三者への提供・公開は行わない
- **netkeiba 規約**: スクレイピングは規約上グレーゾーン。レート制御（最低 3 秒 + ジッター）を徹底し、robots.txt を遵守する
- **即時停止スイッチ**: Settings 画面の停止スイッチ、および `/api/scraper/stop` エンドポイントで任意のタイミングでスクレイピングを止められる
- **規約変更時**: netkeiba の利用規約・robots.txt が変更された場合は即座にスクレイピングを停止し、対応を検討する（[operations.md](operations.md) 参照）
