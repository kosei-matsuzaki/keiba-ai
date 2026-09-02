# 解説動画 (manim)

モデルの計算過程を 3Blue1Brown 風に可視化した動画と、その manim ソース。
散文のドキュメントとは性質が違う（コードとビルド生成物）ので、`docs/` 直下から
ここに分けてある。

| ファイル | 中身 |
| --- | --- |
| [model-explainer.py](model-explainer.py) | manim のシーン定義 (`ModelMath`)。能力推定 → self-attention → 確率導出 → 買目提案 |
| `model-explainer.mp4` | 上を 1080p30 で書き出したもの (8.7 MB)。リポジトリ直下の [README.md](../../README.md) からリンクしている |

ポスター画像は `docs/images/model-explainer-poster.png` に置いてある。動画の終盤で
実際の画面を見せる場面が `docs/images/` の画面キャプチャを読むので、**画像ディレクトリは
動かさないこと**（動かすと動画が書き出せなくなる）。

## 書き出し

**リポジトリ直下から**実行する。`docs/images` を相対で読むため。

```bash
manim -ql docs/explainer/model-explainer.py ModelMath                      # プレビュー
manim -r 1920,1080 --fps 30 docs/explainer/model-explainer.py ModelMath    # 1080p30 本番
```

別のディレクトリから回す場合は `KEIBA_IMG_DIR` に画像ディレクトリの絶対パスを渡す。

必要なのは `manim` とその依存だけで、`[nn]` extra (torch / lightning) は要らない。
