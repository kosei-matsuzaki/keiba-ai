# docs — KEIBA AI

作品としての説明（概要・見どころ・技術解説）はリポジトリ直下の [README.md](../README.md) にあります。
README の `<!-- portfolio:begin -->` 〜 `<!-- portfolio:end -->` の内側はポートフォリオサイト
（`portfolio-site/src/data/projects.ts`）から生成しているので、直接編集しないでください。

ここは開発者向けの入口です。

| ファイル | 内容 | いつ読むか |
| --- | --- | --- |
| [spec.md](spec.md) | 技術仕様（技術スタック・ディレクトリ構成・DB スキーマ・API エンドポイント・開発ビルド手順） | API を足す / スキーマを変える前 |
| [design.md](design.md) | 設計方針（非機能要件・アーキテクチャ図・AI モジュール責務分離・UI 画面構成・状態管理・拡張ポイント） | 画面や層をまたぐ変更を入れる前 |
| [data-pipeline.md](data-pipeline.md) | スクレイピング・取り込み仕様（対象 URL・レート制御・robots.txt 遵守・HTML キャッシュ・増分取得・失敗レジューム・法的配慮） | スクレイパーを触る前。**レート制御と robots.txt は必ず** |
| [ai-model.md](ai-model.md) | モデル設計（問題定義・Set Transformer・損失・確率変換・特徴量・学習評価フロー・ベットルール・実験の記録） | 学習・評価・買い方を変える前。**否定済みの仮説を再訪しないため** |
| [operations.md](operations.md) | 運用（セットアップ・データ取り込み・再学習サイクル・モデル世代管理・バックアップ・トラブルシューティング） | 動かすとき・壊れたとき |
| [archive/](archive/README.md) | 記録（打ち切った系統・過去の設計メモ）。[2026.md](archive/2026.md) と [design-review-2026-08-23.md](archive/design-review-2026-08-23.md) | 「なぜこうなっているか」を辿るとき。**いまの仕様として読まない** |
| [explainer/](explainer/README.md) | モデルの計算過程を manim で可視化した解説動画とそのソース。書き出し手順もここ | 人に説明するとき / 動画を撮り直すとき |
| `images/` | 画面キャプチャ。直下の README とポートフォリオサイトの両方がここを参照する | 画面を撮り直したとき |

## クイックスタート

前提: [uv](https://docs.astral.sh/uv/) / Node.js 20+ / pnpm がインストール済みであること。

```bash
# 開発サーバ起動 (FastAPI on :8765 + Vite on :5173)
bash scripts/dev.sh
# → http://localhost:5173 をブラウザで開く
```

依存同期・DB migration は `dev.sh` が毎回自動で行います。**スクレイピング済みデータと学習済みモデルはリポジトリに含まれない**ため、初回起動時は空の状態から始まります (画面の取込ボタンまたは `uv run keiba-ingest --date YYYY-MM-DD` でデータ取得)。セットアップの詳細は [docs/operations.md](operations.md) を参照。

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
- **即時停止スイッチ**: Race 画面の取込パネル（`DayIngestPanel`）、`/api/scraper/stop` エンドポイント、環境変数 `KEIBA_SCRAPER_STOP=1` の 3 経路で、任意のタイミングでスクレイピングを止められる
- **規約変更時**: netkeiba の利用規約・robots.txt が変更された場合は即座にスクレイピングを停止し、対応を検討する（[operations.md](operations.md) 参照）
