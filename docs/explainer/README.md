# 解説動画 (manim)

モデルの計算過程を 3Blue1Brown 風に可視化した動画と、その manim ソース。
散文のドキュメントとは性質が違う（コードとビルド生成物）ので、`docs/` 直下から
ここに分けてある。

| ファイル | 中身 |
| --- | --- |
| [model-explainer.py](model-explainer.py) | manim のシーン定義 (`ModelMath`)。8 幕の中身は同クラスの docstring |
| `model-explainer.mp4` | 上を 1080p30 で書き出したもの (12 MB・2分31秒)。リポジトリ直下の [README.md](../../README.md) からリンクしている |

ポスター画像は `docs/images/model-explainer-poster.png`（能力エンコーダの場面を
書き出しから 1 枚抜いたもの）。

## 書き出し

外部ファイルを読まないので、どのディレクトリからでも実行できる。

```bash
manim -ql docs/explainer/model-explainer.py ModelMath                      # プレビュー
manim -r 1920,1080 --fps 30 docs/explainer/model-explainer.py ModelMath    # 1080p30 本番
```

必要なのは `manim` とその依存に加えて **LaTeX**。数式だけでなく**字幕もラベルも
すべて LaTeX で組んでいる** (`jt()`) ので、LaTeX が無い環境では 1 枚も出ない
(`latex failed but did not produce a log file` で止まる)。`[nn]` extra (torch / lightning) は要らない。

**manim の `Text` (Pango) は使わない。** 1 つの文の中で書体が入れ替わり
(単語ごとに serif と等幅が混ざる)、字間も揃わないため
([manim#2844](https://github.com/ManimCommunity/manim/issues/2844))。`jt()` が
LaTeX の特殊文字を逃がすので呼ぶ側は素の文字列を渡してよく、バッククォートで
囲んだ部分 (`` `odds_win` ``) だけ等幅になる。

手を入れたら、書き出す前に**中身が枠に収まっているか**を見る。各 act は組み終えてから
`fit()` で 1 つの枠に入れる作りなので、要素を足すと**エラーにならずに縮む**。

```bash
ffmpeg -i media/videos/model-explainer/480p15/ModelMath.mp4 -vf 'fps=1/3,scale=640:-1' /tmp/f/%03d.png
ffmpeg -start_number 1 -i /tmp/f/%03d.png -vf tile=4x6 -frames:v 1 /tmp/sheet.png
```

尺を削るときは `--dry_run` で幕ごとの秒数を出してから決める（各 act が終わるたびに
`[timing]` を出す）。**勘で切ると機構の説明から先に痩せる** — 実際に厚かったのは説明では
なく、同じ絵を 1 頭ずつ・1 ステップずつ再生していた時間だった。

```bash
manim -ql --dry_run docs/explainer/model-explainer.py ModelMath 2>&1 | tr '\r' '\n' | grep timing
```
