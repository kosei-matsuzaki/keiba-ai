# docs — KEIBA AI

作品としての説明（概要・見どころ・技術解説）はリポジトリ直下の [README.md](../README.md) にあります。
README の `<!-- portfolio:begin -->` 〜 `<!-- portfolio:end -->` の内側はポートフォリオサイト
（`portfolio-site/src/data/projects.ts`）から生成しているので、直接編集しないでください。

**README に出ている回収率は生成された時点の数字で、測った窓が書かれていません。**
測り直すたびに動くので、窓（期間・レース数）と 95% 区間つきの正本は
[ai-model.md](ai-model.md) の「OOS 実測」を見てください。

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

## 動かす

`bash scripts/dev.sh` で FastAPI (:8765) と Vite (:5173) が起動します。
**前提ツール・初回の手順・つまずいたときは [operations.md](operations.md)** を見てください。

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

個人研究限定・商用利用禁止・レート制御（最低 3 秒 + ジッター）・robots.txt の fail-closed・
即時停止の 3 経路・規約変更時の対応は、**[data-pipeline.md](data-pipeline.md)「法的・倫理的配慮」が
正本**です。スクレイパーを触る前に必ず読んでください。
