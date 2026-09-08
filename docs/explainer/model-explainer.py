"""Horse-racing model explainer (manim, 3Blue1Brown style). All on-screen text is English.

Eight acts. Each overview is followed by the acts that open its parts: the encoder and
its three feature blocks, then the race-level pipeline, attention and the scoring head,
then the training loop. The list is in :class:`ModelMath`.

Layout discipline (an earlier cut drifted left and left stray arrows behind):

  * an act builds **all** of its mobjects first, then ``fit()`` scales and centres
    the whole group into one content box -- no per-act nudges, nothing off-frame
  * ``clear_stage()`` fades everything that is not the title, the caption or an
    explicitly carried vector, so no mobject can survive into the next act
  * every string goes through ``jt()``, which typesets with LaTeX rather than
    Pango -- manim's ``Text`` swaps typeface mid-sentence here (ManimCommunity/manim#2844)

Palette (consistent throughout):
  aggregate=emerald / past-run token=aqua / hidden h=blue / race=violet / odds=amber /
  ability=fuchsia / Query=cyan / Key=yellow / Value=lime / score+active model=rose /
  market=slate

Render (any working directory -- nothing outside this file is read):
  manim -ql docs/explainer/model-explainer.py ModelMath                      # preview
  manim -r 1920,1080 --fps 30 docs/explainer/model-explainer.py ModelMath    # 1080p30 final

Needs ``manim``, its dependencies and a LaTeX install (every caption is typeset by LaTeX).
The ``[nn]`` extra (torch / lightning) is not used.
"""
from manim import *
import numpy as np

C_AGG     = "#34d399"   # aggregate features (emerald)
C_PAST    = "#5eead4"   # one past-run token (aqua) -- distinct from the hidden state
C_HIST    = "#60a5fa"   # GRU hidden state / history vector (blue)
C_RACE    = "#a78bfa"   # race-level features (violet)
C_ODDS    = "#fbbf24"   # odds / market price (amber)
C_ABILITY = "#e879f9"   # ability vector (fuchsia)
C_Q       = "#22d3ee"   # Query -- also the probability model (cyan)
C_K       = "#facc15"   # Key (yellow)
C_V       = "#a3e635"   # Value (lime)
C_SCORE   = "#fb7185"   # score / probability / the active model (rose)
C_MARKET  = "#94a3b8"   # the market's own numbers (slate)
C_DIM     = "#9aa4b2"
BG        = "#0c1420"

# 1 つの箱に全部入れて fit() で収める。act ごとの手当ては置かない
# (旧版はここを個別に shift していて、足すたびに片側へ寄っていた)。
CONTENT_W = 12.8
CONTENT_H = 4.9
CONTENT_C = np.array([0.0, -0.05, 0.0])
CAP_W = 12.0
MAX_GROW = 1.25

# 字の大きさはこの 5 段だけ。役割で選ぶ (画面のどこに出るかではなく)
T_END = 26      # 締めのカード
T_CAP = 23      # 字幕
T_HEAD = 20     # 図の見出し
T_LABEL = 16    # 図中のラベル
T_NOTE = 15     # 補足
T_MICRO = 13    # 目盛り・縮小図

# 幕 2-4 は右端にエンコーダの地図を出したままにする。いま開いているブロックが
# どこの話なのかを、言葉ではなく位置で示すため。
#
# 網の部分は "encoder" の箱に畳んで、入力列だけ大きく残す。地図の仕事は
# 「三つのうちどれか」を指すことなので、**指す対象が小さくなっては意味がない**。
# 本文と地図は「左右に振り分ける」のではなく、2 つで 1 つの構図として置く。
# 端に寄せると本文も地図も小さいまま中央が空く
MAP_SCALE = 0.90         # 縮小図の倍率 (層の間だけ別に詰める)
MAP_GAP = 0.75           # 縮小図の段の間隔。倍率だけ上げても幅は伸びない
MAP_C = np.array([3.50, -0.05, 0.0])
MAP_DIM = 0.32
MAP_FREE_W = 6.6
MAP_FREE_C = np.array([-3.05, -0.05, 0.0])
MAP_FREE_GROW = 1.5

POOL = [0.25, 0.82, 0.48, 0.35, 0.9, 0.6, 0.18, 0.72, 0.42, 0.55,
        0.86, 0.3, 0.66, 0.5, 0.22, 0.78, 0.4, 0.7]

#: LaTeX で組むと Pango より一回り小さく出るので、字数の設計をそのままにして倍率で合わせる
TEX_SIZE = 1.45

_TEX_ESCAPE = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape(s):
    """LaTeX の特殊文字を逃がし、素の引用符を LaTeX の開き/閉じに割り振る。"""
    out, opening = [], True
    for ch in s:
        if ch == '"':
            out.append("``" if opening else "''")
            opening = not opening
        else:
            out.append(_TEX_ESCAPE.get(ch, ch))
    return "".join(out)


def _tex_body(s):
    """バッククォートで囲んだ部分を等幅にする (`odds_win` のような識別子用)。"""
    parts = []
    for i, chunk in enumerate(s.split("`")):
        if not chunk:
            continue
        parts.append(r"\texttt{" + _escape(chunk) + "}" if i % 2 else _escape(chunk))
    return "".join(parts)


def jt(s, size=22, color=WHITE, weight=NORMAL, caps=False):
    """On-screen text, typeset by LaTeX rather than Pango.

    manim の `Text` は Pango 経由で、**1 つの文の中で書体が入れ替わる**
    (単語ごとに serif と等幅が混ざる) 上に字間が揃わない
    — ManimCommunity/manim#2844。数式は既に Computer Modern なので、
    文字も LaTeX で組めば書体が 1 つに揃い、この経路ごと無くなる。

    バッククォートで囲んだ部分は等幅になる: ``jt("`odds_win` and popularity")``。
    `--` は LaTeX が en ダッシュに組む (等幅の中では 2 本のハイフンのまま)。
    """
    body = _tex_body(s)
    if caps:
        # 差し込み図の見出し用。素の小文字だと本文の言いかけに見えるので、
        # スモールキャップスでラベルの側に寄せる
        body = r"\textsc{" + body + "}"
    if weight == BOLD:
        body = r"\textbf{" + body + "}"
    # mbox で折り返しを止める。standalone は既定幅で改行するので、そのままだと
    # 長い注記だけ勝手に 2 行になり、1 行を前提にした fit() の見積もりが崩れる
    return Tex(r"\mbox{" + body + "}", color=color, font_size=size * TEX_SIZE)


def mt(tex, color=WHITE, s=0.6):
    return MathTex(tex, color=color).scale(s)


def arr(start, end, color, sw=3.0, buff=0.1):
    return Arrow(start, end, buff=buff, color=color, stroke_width=sw,
                 tip_length=0.16, max_tip_length_to_length_ratio=0.28)


def vs(seed, k=5):
    return [POOL[(seed * 5 + i * 3) % len(POOL)] for i in range(k)]


def valdot(v, color, cell=0.30):
    """One number, drawn as a square.

    値と隠れ層のノードは同じ形にする。片方だけ形が違うと、同じ「1 つの数」が
    別物に見える。角を落とさないのは、角丸は演算の箱 (W / head MLP / encoder)
    の側の形だから。
    """
    sq = Square(cell, stroke_width=1.0, stroke_color=color)
    sq.set_fill(color, opacity=0.12 + 0.72 * float(v))
    return sq


def vvec(vals, color, cell=0.30):
    """Vertical vector: a column of value cells."""
    g = VGroup(*[valdot(v, color, cell) for v in vals])
    g.arrange(DOWN, buff=cell * 0.12)
    return g


def hvec(vals, color, cell=0.24):
    """Horizontal vector -- keeps the per-horse rows of act 6 flat and even."""
    g = VGroup(*[valdot(v, color, cell) for v in vals])
    g.arrange(RIGHT, buff=cell * 0.12)
    return g


def opbox(tex, color, s=0.72):
    b = RoundedRectangle(width=s, height=s, corner_radius=0.09,
                         stroke_color=color, stroke_width=2.5,
                         fill_color=color, fill_opacity=0.14)
    t = MathTex(tex, color=color).scale(0.55).move_to(b)
    return VGroup(b, t)


def chip(s, color, size=19, width=None, pad=0.55, caps=False):
    """Rounded label box. Fixed `width` keeps a column of chips aligned."""
    t = jt(s, size, color=color, caps=caps)
    box = RoundedRectangle(width=width or (t.width + pad), height=t.height + 0.36,
                           corner_radius=0.12, stroke_color=color, stroke_width=2,
                           fill_color=color, fill_opacity=0.10)
    t.move_to(box)
    return VGroup(box, t)


def nlayer(k, color, x, span=1.15, size=0.23):
    """A hidden layer -- same shape as a value, so the whole diagram reads as one."""
    ys = np.linspace(span, -span, k)
    return VGroup(*[Square(size, stroke_color=color, stroke_width=2,
                           fill_color=color, fill_opacity=0.15).move_to([x, y, 0])
                    for y in ys])


def edges(l1, l2, color=C_DIM, faint=False):
    """All-to-all thin lines, drawn edge to edge rather than centre to centre.

    中心から引くとセルの下に線が潜って濁る。外周から引くと、入力ベクトルと
    隠れ層を同じ描き方でつなげるので、**入出力だけ矢印にする必要がなくなる**。
    faint は縮小図用: 同じ本数を小さい面積に引くので、薄くしないと滲む。
    """
    e = VGroup()
    for a in l1:
        for b in l2:
            e.add(Line(a.get_right(), b.get_left(), stroke_color=color,
                       stroke_width=0.6 if faint else 0.8,
                       stroke_opacity=0.22 if faint else 0.28))
    return e


def probbars(vals, labels, color, bw=0.44, gap=0.26, hmax=1.7, fmt="{:.0%}"):
    """Labelled probability bars (value on top, name underneath)."""
    g = VGroup()
    for v, lb in zip(vals, labels):
        bar = Rectangle(width=bw, height=max(0.05, hmax * v), stroke_width=0,
                        fill_color=color, fill_opacity=0.85)
        name = jt(lb, T_MICRO, C_DIM).next_to(bar, DOWN, buff=0.10)
        val = jt(fmt.format(v), T_MICRO, color).next_to(bar, UP, buff=0.08)
        g.add(VGroup(bar, name, val))
    g.arrange(RIGHT, buff=gap, aligned_edge=DOWN)
    return g


def curve(points, color, sw=3.0):
    m = VMobject(stroke_color=color, stroke_width=sw)
    m.set_points_smoothly([np.array([x, y, 0.0]) for x, y in points])
    return m


def fit_beside_map(group):
    """Fit into the space left of the encoder map that acts 2-4 keep on screen."""
    return fit(group, w=MAP_FREE_W, center=MAP_FREE_C, grow=MAP_FREE_GROW)


def fit(group, w=CONTENT_W, h=CONTENT_H, center=None, grow=MAX_GROW):
    """Scale a finished act to fill the content box, then centre it.

    拡大も許すのは、act ごとに絵の大きさがばらつくため (小さく組んだ act だけ
    画面の 1/3 しか使わない)。上限を置くのは、字が主役の act で文字だけ巨大に
    ならないようにするため。
    """
    if group.width > 0 and group.height > 0:
        s = min(w / group.width, h / group.height, grow)
        if abs(s - 1.0) > 0.01:
            group.scale(s)
    group.move_to(CONTENT_C if center is None else center)
    return group


class ModelMath(Scene):
    """Eight acts, and the same move three times: show the whole, then open it.

        1 ability      one horse in, one vector out -- and the three inputs that feed it
        2 aggregate    46 race-day columns, the price among them
        3 history      past runs, one token each, folded by a GRU
        4 race         seven shared columns, copied onto every horse
        5 pipeline     where those ability vectors go: transformer, head, every bet type
        6 attention    one horse reads the field: query, key, value, and the matrix
        7 the price    it enters at the head, and every bet type follows from one score
        8 training     the loop that adjusts it, and what "the money grew" is defined as

    Act 1 shows a whole and acts 2-4 open its three blocks; act 5 shows the next whole
    and acts 6-7 open its two stages; act 8 opens with the training loop before the
    formula and the curve. Every act names itself in the corner, in its own colour.

    While those three acts run, the encoder stays on screen shrunk into a map on the
    right: the block being opened is boxed, and the act's content grows out of that
    block. Which part of the encoder is being explained is then never in doubt.
    """

    # ============================================================ scaffolding
    def construct(self):
        self.camera.background_color = BG
        self._cap = None

        title = jt("Horse-Racing Prediction with a Set Transformer", 34, weight=BOLD)
        sub = jt("Inside the computation -- how a race becomes a score", T_CAP, color=C_DIM)
        VGroup(title, sub).arrange(DOWN, buff=0.30).move_to(ORIGIN)
        self.play(Write(title), FadeIn(sub, shift=UP * 0.2))
        self.wait(1.1)
        self.play(FadeOut(sub),
                  title.animate.scale(0.46).set_opacity(0.55).to_corner(UL, buff=0.38),
                  run_time=0.9)
        self.title = title
        self._act_head = None

        for act in (self.act1_ability, self.act2_aggregate, self.act3_history,
                    self.act4_race, self.act5_pipeline, self.act6_attention,
                    self.act7_price, self.act8_training):
            act()
            # 尺を組み直すときは、勘で削ると機構の説明から先に痩せる。どの幕に
            # 時間が乗っているかを見てから削るための実測 (--dry_run で読める)
            print(f"[timing] {act.__name__:16s} -> {self.renderer.time:7.1f}s")

        self.play(FadeOut(self._cap), FadeOut(self.title), FadeOut(self._act_head),
                  run_time=0.7)
        end = VGroup(
            jt("Separate ability from market price.", T_END),
            jt("Then optimise the money, not the ranking.", T_END, C_DIM),
        ).arrange(DOWN, buff=0.36).move_to(ORIGIN)
        self.play(FadeIn(end[0], shift=UP * 0.2))
        self.play(FadeIn(end[1], shift=UP * 0.2), run_time=0.9)
        self.wait(2.2)
        self.play(FadeOut(end), run_time=1.0)

    def set_act(self, n, name, color):
        """Corner header: which act this is, in that act's colour.

        どの幕も同じ枠なので、外から見て「いまどこか」の手がかりが画面に無い。
        色はその幕が説明している対象の色をそのまま使う。
        """
        num = jt(f"{n:02d}", 30, color)
        of = jt("/ 08", T_MICRO, C_DIM)
        nm = jt(name, T_LABEL + 2, color, caps=True)
        line = VGroup(num, of, nm).arrange(RIGHT, buff=0.24, aligned_edge=DOWN)
        rule = Line(ORIGIN, RIGHT * (line.width + 0.1), stroke_color=color, stroke_width=3)
        block = VGroup(rule, line).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        block.next_to(self.title, DOWN, buff=0.26).align_to(self.title, LEFT)
        if self._act_head is None:
            self._act_head = block
            self.play(FadeIn(block, shift=RIGHT * 0.15), run_time=0.5)
        else:
            self.play(FadeOut(self._act_head, shift=LEFT * 0.12),
                      FadeIn(block, shift=RIGHT * 0.12), run_time=0.45)
            self._act_head = block

    @staticmethod
    def _wrap(s, limit=58):
        """Break one caption into at most two lines.

        区切り記号があればそこで折る。単純に中央で折ると "-- and" のように
        句のつながりを断つ位置に落ちて、二行目が言いかけから始まる。
        """
        if len(s) <= limit:
            return [s]
        for sep in (" -- ", ": ", ". ", "; ", ", "):
            i = s.find(sep, int(len(s) * 0.25), int(len(s) * 0.78))
            if i > 0:
                return [s[:i + len(sep) - 1].strip(), s[i + len(sep):].strip()]
        words, target = s.split(" "), len(s) / 2
        head = words[0]
        for i in range(2, len(words)):
            trial = " ".join(words[:i])
            if abs(len(trial) - target) >= abs(len(head) - target):
                break
            head = trial
        return [head, s[len(head) + 1:]]

    def cap(self, s, color=WHITE, hold=0.5):
        """Bottom caption. Cross-fades -- morphing between unrelated strings smears.

        1 行に詰めると字が小さくなるか端まで届く。2 行に折って、縁からも離す。
        """
        new = VGroup(*[jt(line, T_CAP, color=color) for line in self._wrap(s)])
        new.arrange(DOWN, buff=0.20)
        new.to_edge(DOWN, buff=0.46)
        if new.width > CAP_W:
            new.scale_to_fit_width(CAP_W)
        if self._cap is None:
            self.play(FadeIn(new), run_time=0.6)
        else:
            self.play(FadeOut(self._cap, shift=DOWN * 0.12),
                      FadeIn(new, shift=DOWN * 0.12), run_time=0.5)
        self._cap = new
        self.wait(hold)

    def carry(self, mob):
        """Lift a copy to the top level so `clear_stage` can spare it.

        act をまたいで持ち越すベクトルは、親 VGroup の子のままだと親ごと消える。
        複製を独立に add してから元を消すと、見た目は繋がったまま残る。
        """
        c = mob.copy()
        self.add(c)
        return c

    def clear_stage(self, keep=(), run_time=0.7):
        """Fade everything except title / caption / carried mobjects.

        走査するのは self.mobjects なので、グループに入れ忘れた矢印も必ず消える
        (旧版は isinstance(Arrow) で拾っていて、包んだ矢印が次の act に残っていた)。
        """
        spared = {id(self.title), id(self._cap), id(self._act_head)}
        spared |= {id(m) for m in keep}
        doomed = [m for m in self.mobjects if id(m) not in spared]
        if doomed:
            self.play(*[FadeOut(m, shift=LEFT * 0.18) for m in doomed], run_time=run_time)

    def build_map(self, column, labels, layers, wires, ability, drop):
        """Shrink the encoder into an inset on the right and keep it through acts 2-4.

        描き直した略図ではなく**実物をそのまま小さくする**。層の間だけ詰めて幅を
        落とし、結線は薄くして地の模様にする。枠と見出しを付けるのは、本文の隣に
        置いたときに「余った図」ではなく差し込み図として読ませるため。

        ブロック名は載せない。色と位置で足りるうえ、名前の列に幅を取られると
        **指す対象そのものが小さくなる**。
        """
        k = MAP_SCALE
        small = [column.copy().scale(k), layers[0].copy().scale(k),
                 layers[1].copy().scale(k), ability.copy().scale(k)]
        VGroup(*small).arrange(RIGHT, buff=MAP_GAP)
        col, h1, h2, ab = small
        cells = VGroup(*[c for block in col for c in block])
        wire_targets = [edges(cells, h1, faint=True), edges(h1, h2, faint=True),
                        edges(h2, ab, faint=True)]
        inner = VGroup(col, *wire_targets, h1, h2, ab)
        head = jt("ability encoder", T_NOTE, C_DIM, caps=True).next_to(inner, UP, buff=0.30)
        panel = SurroundingRectangle(VGroup(head, inner), color=C_DIM, buff=0.32,
                                     corner_radius=0.16, stroke_width=1.2)
        panel.set_fill(C_DIM, 0.04).set_stroke(opacity=0.35)
        VGroup(panel, head, inner).move_to(MAP_C)
        col.set_opacity(MAP_DIM)
        VGroup(h1, h2, ab).set_opacity(MAP_DIM + 0.18)

        self.play(FadeOut(drop), FadeIn(panel), FadeIn(head),
                  FadeOut(VGroup(*labels), shift=LEFT * 0.2),
                  *[Transform(part, target) for part, target in zip(column, col)],
                  Transform(layers[0], h1), Transform(layers[1], h2),
                  Transform(ability, ab),
                  *[Transform(part, target) for part, target in zip(wires, wire_targets)],
                  run_time=1.1)
        self._map_blocks = list(column)
        # clear_stage が見るのは最上位の mobject なので、子ではなく親の column を渡す
        self._map_parts = [column, *layers, *wires, ability, panel, head]
        # 枠と見出しは幕 5 で全体図の 1 段目に畳む (差し込み図がそのまま箱になる)
        self._map_frame = (panel, head)
        self._map_box = None

    def focus_map(self, idx):
        """Light up one block of the map and dim the other two.

        この大きさでは濃淡だけだとどこが光っているのか分からないので、囲みも足す。
        """
        anims = [block.animate.set_opacity(1.0 if i == idx else MAP_DIM)
                 for i, block in enumerate(self._map_blocks)]
        target = self._map_blocks[idx]
        box = SurroundingRectangle(target, buff=0.06, corner_radius=0.04, stroke_width=2,
                                   color=target[0].get_stroke_color())
        if self._map_box is None:
            self._map_box = box
            self._map_parts.append(box)
            anims.append(Create(box))
        else:
            anims.append(Transform(self._map_box, box))
        self.play(*anims, run_time=0.5)

    def labelled_column(self, values, color, names, head, cell=0.34, name_color=None):
        """A feature column with one arrow-and-name per cell. Acts 2-4 share this shape."""
        col = vvec(values, color, cell=cell)
        names_g, arrows = VGroup(), VGroup()
        for i, (sq, name) in enumerate(zip(col, names)):
            c = (name_color or [C_DIM] * len(names))[i]
            t = jt(name, T_LABEL, c).next_to(sq, RIGHT, buff=1.1)
            names_g.add(t)
            arrows.add(arr(sq.get_right(), t.get_left(), c, sw=1.6, buff=0.12))
        heading = jt(head, T_HEAD, color)
        block = VGroup(heading, VGroup(col, arrows, names_g)).arrange(DOWN, buff=0.42)
        return block, col, arrows, names_g

    # ============================================================ 1. ability
    def act1_ability(self):
        self.set_act(1, "ability", C_ABILITY)
        self.cap("One horse in, one vector out. The model calls it ability.",
                 C_ABILITY)

        agg = vvec(vs(2, 4), C_AGG, cell=0.30)
        hist = vvec(vs(16, 4), C_HIST, cell=0.30)
        race = vvec(vs(5, 3), C_RACE, cell=0.30)
        concat = VGroup(agg, hist, race).arrange(DOWN, buff=0.07).move_to([-4.6, 0.0, 0])
        la = jt("aggregate", T_LABEL, C_AGG).next_to(agg, LEFT, buff=0.22)
        lh = jt("history", T_LABEL, C_HIST).next_to(hist, LEFT, buff=0.22)
        lr = jt("race", T_LABEL, C_RACE).next_to(race, LEFT, buff=0.22)
        l_h1 = nlayer(5, WHITE, -1.4, span=1.20)
        l_h2 = nlayer(5, WHITE, 0.8, span=1.20)
        ability = vvec(vs(30, 4), C_ABILITY, cell=0.32).move_to([3.3, 0.0, 0])
        # 入力・出力を矢印で結ばない。層と層を結ぶ細線と同じ描き方に揃えると、
        # 全結合という中身も正しく出るし、扇形の矢印が要らなくなる
        # concat のまま渡すと 3 ブロックの右端中央 (=3 点) からしか線が出ない。
        # 全結合なのだから、1 つの値ごとに引く
        in_cells = VGroup(*[cell for block in concat for cell in block])
        e1, e2, e3 = edges(in_cells, l_h1), edges(l_h1, l_h2), edges(l_h2, ability)
        albl = jt("ability, 32-dim", T_LABEL, C_ABILITY).next_to(ability, DOWN, buff=0.20)
        gelu = jt("each layer: linear, then GELU", T_LABEL, C_DIM)
        noodds = jt("no odds in here: ability is judged without the market's opinion",
                    18, C_ODDS)
        body = VGroup(concat, la, lh, lr, l_h1, l_h2, ability, e1, e2, e3, albl)
        gelu.next_to(body, UP, buff=0.34)
        noodds.next_to(body, DOWN, buff=0.38)
        fit(VGroup(body, gelu, noodds))

        self.play(FadeIn(concat), FadeIn(la), FadeIn(lh), FadeIn(lr), run_time=0.8)
        self.play(FadeIn(e1), FadeIn(e2), FadeIn(e3), Create(l_h1), Create(l_h2),
                  FadeIn(gelu), run_time=1.0)
        for le, ln, col in [(e1, l_h1, WHITE), (e2, l_h2, WHITE)]:
            self.play(LaggedStart(*[ShowPassingFlash(ed.copy().set_stroke(C_SCORE, 2.0),
                                                     time_width=0.6)
                                    for ed in le], lag_ratio=0.003, run_time=0.8),
                      LaggedStart(*[n.animate.set_fill(col, 0.85) for n in ln], lag_ratio=0.05))
        self.play(LaggedStart(*[ShowPassingFlash(ed.copy().set_stroke(C_SCORE, 2.0),
                                                 time_width=0.6)
                                for ed in e3], lag_ratio=0.004, run_time=0.8),
                  FadeIn(ability), FadeIn(albl))
        self.play(FadeIn(noodds, shift=UP * 0.15))
        self.wait(1.4)

        self.cap("Three kinds of input feed it. Each one is worth opening.", C_DIM)
        self.build_map(concat, [la, lh, lr], (l_h1, l_h2), (e1, e2, e3), ability,
                       VGroup(gelu, noodds, albl))

    # ============================================================ 2. aggregate
    def act2_aggregate(self):
        self.set_act(2, "aggregate", C_AGG)
        self.focus_map(0)
        self.cap("46 columns: one race-day snapshot of the horse.", C_AGG)
        names = ["`recent_avg_finish`", "`jockey_recent_win_rate`", "`horse_weight`",
                 "`days_since_last_race`", "`sire_progeny_win_rate`", "`odds_win`"]
        block, col, arrows, labels = self.labelled_column(
            vs(1, 6), C_AGG, names, "46 aggregate columns",
            name_color=[C_DIM] * 5 + [C_ODDS])
        fit_beside_map(block)
        # 地図のブロックから中身が育って出てくる。どの部分の話かを言葉でなく動きで示す
        self.play(TransformFromCopy(self._map_blocks[0], col), FadeIn(block[0]), run_time=1.1)
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(f))
                                for a, f in zip(arrows, labels)], lag_ratio=0.14), run_time=1.4)
        self.wait(0.7)
        self.cap("The price is in there -- and the encoder never sees it.",
                 C_ODDS)
        self.play(Indicate(VGroup(col[5], arrows[5], labels[5]), color=C_ODDS, scale_factor=1.08),
                  run_time=0.8)
        self.wait(0.9)
        self.clear_stage(keep=self._map_parts)

    # ============================================================ 3. history
    def act3_history(self):
        self.set_act(3, "history", C_HIST)
        self.focus_map(1)
        self.cap("Every past run, one token each.", C_PAST)
        tnames = ["finish / field size", "beaten margin", "last 3F", "passing position",
                  "weight carried", "distance", "class of the race", "days since"]
        block, tok, ta, tl = self.labelled_column(
            vs(20, 8), C_PAST, tnames, "one past run = 16 numbers", cell=0.30)
        fit_beside_map(block)
        self.play(TransformFromCopy(self._map_blocks[1], tok), FadeIn(block[0]), run_time=1.1)
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(f))
                                for a, f in zip(ta, tl)], lag_ratio=0.12), run_time=1.6)
        self.wait(0.9)
        self.play(FadeOut(block[0]), FadeOut(tl), FadeOut(ta), run_time=0.5)

        self.cap("A GRU folds the sequence into one history vector.",
                 C_HIST)
        # 地図の隣に置くぶん横幅が狭いので、鎖そのものを詰めて組む
        hid_x, cell_x = [-3.5, -1.85, -0.2, 1.6], [-2.68, -1.03, 0.62]
        hvals = [[0.05] * 5, vs(10, 5), vs(13, 5), vs(16, 5)]
        hs = [vvec(hvals[j], C_HIST, cell=0.24).move_to([hid_x[j], 0.45, 0]) for j in range(4)]
        hl = [mt(f"h_{j}", C_HIST, 0.58).next_to(hs[j], UP, buff=0.12) for j in range(4)]
        cells = VGroup()
        for cx in cell_x:
            b = RoundedRectangle(width=0.78, height=1.0, corner_radius=0.1,
                                 stroke_color=C_HIST, stroke_width=2.5,
                                 fill_color=C_HIST, fill_opacity=0.10).move_to([cx, 0.45, 0])
            cells.add(VGroup(b, mt(r"r,z,\tilde{h}", C_HIST, 0.38).move_to(b)))
        xs = []
        for t, cx in enumerate(cell_x):
            xv = vvec(vs(20 + t, 8), C_PAST, cell=0.15).move_to([cx, -1.45, 0])
            xl = mt("x_{t-" + str(3 - t) + "}", C_PAST, 0.46).next_to(xv, DOWN, buff=0.10)
            xs.append(VGroup(xv, xl))
        eq = mt(r"h_t=(1-z_t)\odot h_{t-1}+z_t\odot \tilde{h}_t", WHITE, 0.62)
        box = SurroundingRectangle(hs[3], color=C_HIST, buff=0.12, corner_radius=0.08)
        hlab = jt("history vector", T_LABEL, C_HIST).next_to(box, DOWN, buff=0.20)
        chain = VGroup(*hs, *hl, cells, *xs, box, hlab)
        eq.next_to(chain, UP, buff=0.36)
        fit_beside_map(VGroup(chain, eq))

        self.play(ReplacementTransform(tok, xs[0][0]), FadeIn(xs[0][1]),
                  FadeIn(hs[0]), FadeIn(hl[0]), Write(eq), run_time=1.1)
        for t in range(3):
            step = [FadeIn(cells[t]),
                    GrowArrow(arr(hs[t].get_right(), cells[t][0].get_left(), C_HIST, sw=2.2)),
                    GrowArrow(arr(xs[t][0].get_top(), cells[t][0].get_bottom(), C_PAST, sw=2.2))]
            if t:
                step.append(FadeIn(xs[t], shift=UP * 0.2))
            self.play(*step, run_time=0.6)
            self.play(GrowArrow(arr(cells[t][0].get_right(), hs[t + 1].get_left(), C_HIST, sw=2.2)),
                      TransformFromCopy(hs[t], hs[t + 1]), FadeIn(hl[t + 1]), run_time=0.7)
        self.play(Create(box), FadeIn(hlab), run_time=0.6)
        self.wait(1.2)
        self.clear_stage(keep=self._map_parts)

    # ============================================================ 4. race
    def act4_race(self):
        self.set_act(4, "race", C_RACE)
        self.focus_map(2)
        self.cap("Seven columns describe the race, not the horse.", C_RACE)
        names = ["`course`", "`distance`", "`surface`", "`weather`",
                 "`track_condition`", "`race_class`", "`n_runners`"]
        block, col, arrows, labels = self.labelled_column(
            vs(5, 7), C_RACE, names, "7 race columns", cell=0.28)
        fit_beside_map(block)
        self.play(TransformFromCopy(self._map_blocks[2], col), FadeIn(block[0]), run_time=1.1)
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(f))
                                for a, f in zip(arrows, labels)], lag_ratio=0.12), run_time=1.4)
        self.wait(0.9)

        self.cap("Shared by the field, so every horse gets the same copy.",
                 C_RACE)
        stacks = VGroup()
        for i in range(4):
            a = vvec(vs(2 + i, 4), C_AGG, cell=0.18)
            h = vvec(vs(16 + i, 4), C_HIST, cell=0.18)
            r = vvec(vs(5, 7), C_RACE, cell=0.18)      # 4 頭とも同じ値 = 共有されている
            s = VGroup(a, h, r).arrange(DOWN, buff=0.05)
            lab = jt(f"Horse {i + 1}", T_MICRO, C_DIM).next_to(s, DOWN, buff=0.16)
            stacks.add(VGroup(s, lab))
        stacks.arrange(RIGHT, buff=0.7)
        src = col.copy().next_to(stacks, LEFT, buff=1.3)
        fans = VGroup(*[arr(src.get_right(), s[0][2].get_left(), C_RACE, sw=1.8, buff=0.15)
                        for s in stacks])
        fit_beside_map(VGroup(src, fans, stacks))
        self.play(ReplacementTransform(col, src), FadeOut(arrows), FadeOut(labels),
                  FadeOut(block[0]), run_time=0.9)
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(f), FadeIn(s))
                                for f, s in zip(fans, stacks)], lag_ratio=0.2), run_time=1.5)
        self.wait(1.2)
        self.clear_stage()
    # ============================================================ 5. the pipeline
    def act5_pipeline(self):
        """Where the ability vectors go -- shown before the acts that open the stages.

        幕 1-4 と同じ「全体を先に、部品を後で」を後半にも置く。これが無いと、
        エンコーダの話が終わった直後に attention とスコアが唐突に始まる。

        エンコーダは紫の出どころを示すだけなので、渡し終わったら退場させる。
        残す絵は「集合の中で読み合う」「値段が横から合流する」「確率が出る」の三つ。
        """
        self.set_act(5, "the pipeline", C_DIM)
        self.cap("Every horse now has one. This is where they go.", C_ABILITY)

        # --- 本番の配置 (エンコーダは入らない) -------------------------------
        a_rows = VGroup(*[hvec(vs(30 + i, 4), C_ABILITY, cell=0.20) for i in range(4)])
        a_rows.arrange(DOWN, buff=0.24)
        horse_lbls = VGroup(*[jt(f"Horse {i + 1}", T_MICRO, C_DIM).next_to(a_rows[i], LEFT,
                                                                          buff=0.26)
                              for i in range(4)])
        attn = VGroup(horse_lbls, a_rows)

        merged = VGroup()
        for i in range(4):
            merged.add(VGroup(hvec(vs(60 + i, 4), C_ABILITY, cell=0.20),
                              hvec(vs(9 + i, 2), C_ODDS, cell=0.20)).arrange(RIGHT, buff=0.22))
        merged.arrange(DOWN, buff=0.24)
        odds_brace = Brace(VGroup(*[m[1] for m in merged]), DOWN, buff=0.16, color=C_ODDS)
        odds_lbl = jt("the odds, one price per horse", T_NOTE, C_ODDS)
        odds_lbl.next_to(odds_brace, DOWN, buff=0.14)

        h_mlp = nlayer(3, C_SCORE, 0.0, span=0.46, size=0.20)
        s_col = VGroup(*[valdot(v, C_SCORE, 0.26) for v in [0.9, 0.35, 0.6, 0.2]])
        s_col.arrange(DOWN, buff=0.24)

        bars = VGroup()
        for vals, color, name in [([0.46, 0.14, 0.28, 0.12], C_SCORE, "win"),
                                  ([0.83, 0.42, 0.66, 0.38], C_V, "place"),
                                  ([0.09, 0.05, 0.02], C_ODDS, "combos")]:
            # 券種ごとに自分の最大値で正規化する。連系は絶対値が小さく、共通の
            # 目盛りだと 3 本とも線に潰れる (実数は幕 7 で出す)
            top = max(vals)
            group = VGroup(*[Rectangle(width=0.16, height=0.62 * v / top, stroke_width=0,
                                       fill_color=color, fill_opacity=0.85) for v in vals])
            group.arrange(RIGHT, buff=0.07, aligned_edge=DOWN)
            bars.add(VGroup(group, jt(name, T_MICRO, color).next_to(group, DOWN, buff=0.10)))
        bars.arrange(DOWN, buff=0.30, aligned_edge=LEFT)

        row = VGroup(attn, merged, h_mlp, s_col, bars).arrange(RIGHT, buff=1.15)
        # 前後にいるのは同じ 4 頭。1 頭ぶんの束だけ明るくして、その馬の新しい
        # ベクトルが 4 頭ぜんぶから来ていることを見せる (幕 6 で開く場所)
        cross = VGroup()
        for i in range(4):
            for j in range(4):
                cross.add(Line(a_rows[j].get_right(), merged[i][0].get_left(),
                               stroke_color=C_Q, stroke_width=1.6 if i == 0 else 1.0,
                               stroke_opacity=0.65 if i == 0 else 0.14))
        odds_brace.next_to(VGroup(*[m[1] for m in merged]), DOWN, buff=0.16)
        odds_lbl.next_to(odds_brace, DOWN, buff=0.14)
        w_head = VGroup(edges(VGroup(*[m[1] for m in merged]), h_mlp), edges(h_mlp, s_col))
        w_out = VGroup(*[arr(s_col[i].get_right(), bars.get_left(), C_DIM, sw=1.4, buff=0.14)
                         for i in range(4)])
        names = VGroup(jt("set transformer", T_NOTE, C_Q, caps=True),
                       jt("scoring head", T_NOTE, C_SCORE, caps=True))
        body = VGroup(attn, cross, merged, odds_brace, odds_lbl, w_head, h_mlp, s_col,
                      w_out, bars)
        names[0].next_to(attn, UP, buff=0.34)
        names[1].next_to(VGroup(h_mlp, s_col), UP, buff=0.34)
        note = jt("four rows = four horses in the race", T_NOTE, C_DIM)
        note.next_to(body, DOWN, buff=0.30)
        fit(VGroup(body, names, note))

        # --- 差し込み図をほどいて 4 行を渡し、エンコーダは退場する -----------
        panel, head_lbl = self._map_frame
        col, h1, h2 = self._map_parts[0], self._map_parts[1], self._map_parts[2]
        wires, ability = self._map_parts[3:6], self._map_parts[6]
        enc = VGroup(col, h1, h2, ability, *wires).copy()
        enc.scale(1.15).next_to(a_rows, LEFT, buff=1.5)
        enc_lbl = jt("ability encoder", T_NOTE, C_ABILITY, caps=True).next_to(enc, UP, buff=0.34)
        hand = VGroup(*[arr(enc.get_right(), a_rows[i].get_left(), C_ABILITY, sw=1.6, buff=0.15)
                        for i in range(4)])
        self.play(FadeOut(panel), FadeOut(head_lbl),
                  *[Transform(o, t) for o, t in zip([col, h1, h2, ability, *wires], enc)],
                  FadeIn(enc_lbl), run_time=1.1)
        self.play(LaggedStart(*[GrowArrow(a) for a in hand], lag_ratio=0.12),
                  FadeIn(a_rows), FadeIn(horse_lbls), run_time=1.0)
        self.wait(1.0)
        self.play(FadeOut(VGroup(col, h1, h2, ability, *wires, enc_lbl, hand),
                          shift=LEFT * 0.3), run_time=0.8)

        self.cap("Inside the race, every horse reads every other.", C_Q)
        self.play(FadeIn(cross), FadeIn(names[0]),
                  FadeIn(VGroup(*[m[0] for m in merged])), run_time=1.2)
        self.wait(1.0)

        self.cap("Then each horse's own price joins it.", C_ODDS)
        self.play(FadeIn(VGroup(*[m[1] for m in merged]), shift=LEFT * 0.2),
                  GrowFromCenter(odds_brace), FadeIn(odds_lbl), run_time=0.9)
        self.wait(0.9)

        self.cap("Head, score, and every bet type off the same numbers.", C_SCORE)
        self.play(FadeIn(w_head[0]), FadeIn(h_mlp), FadeIn(names[1]), run_time=0.7)
        self.play(FadeIn(w_head[1]), FadeIn(s_col), run_time=0.6)
        self.play(LaggedStart(*[GrowArrow(a) for a in w_out], lag_ratio=0.1),
                  LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars], lag_ratio=0.2),
                  *[FadeIn(b[1]) for b in bars], FadeIn(note), run_time=1.4)
        self.wait(1.2)

        self.cap("One horse in, one score out -- and the market touches it once.", C_DIM)
        flashes = [ShowPassingFlash(w.copy().set_stroke(WHITE, 3), time_width=0.9)
                   for w in [*cross, *w_head[0], *w_head[1], *w_out]]
        self.play(LaggedStart(*flashes, lag_ratio=0.03), run_time=2.0)
        self.wait(0.8)

        self.cap("Those two are what the next two acts open.", C_DIM)
        self.play(Indicate(VGroup(cross, names[0]), color=C_Q, scale_factor=1.08),
                  Indicate(VGroup(h_mlp, names[1]), color=C_SCORE, scale_factor=1.08),
                  run_time=0.9)
        self.wait(0.9)
        # ability の列は次の幕にそのまま渡す (幕をまたいでも同じ物だと分かる)
        carried = self.carry(a_rows)
        self.clear_stage(keep=[carried])
        self._carry_ability = carried
    # ============================================================ 6. attention
    def act6_attention(self):
        n = 4
        ys = [1.7, 0.55, -0.6, -1.75]
        scores = [2.40, 0.76, 1.16, 0.35]
        weights = [0.62, 0.12, 0.18, 0.08]

        self.set_act(6, "attention", C_Q)
        self.cap("Fast is relative, so each horse reads the others.",
                 C_ABILITY)
        A = VGroup(*[hvec(vs(30 + i, 6), C_ABILITY, cell=0.20).move_to([-4.6, ys[i], 0])
                     for i in range(n)])
        names = VGroup(*[jt(f"Horse {i + 1}", T_LABEL, C_DIM).next_to(A[i], LEFT, buff=0.35)
                         for i in range(n)])
        fit(VGroup(A, names))
        self.play(ReplacementTransform(self._carry_ability, A),
                  LaggedStart(*[FadeIn(nm, shift=RIGHT * 0.2) for nm in names],
                              lag_ratio=0.15), run_time=1.3)
        self.wait(0.7)

        # 1 頭ぶんだけ q,k,v を意味つきで開く。4 頭ぶんを一度に出すと、
        # 12 個の小さなベクトルが同時に現れて何を見ればよいか分からなくなる
        self.cap("One matrix W turns an ability into three vectors.", C_K)
        a1 = A[0].copy()
        a1_lbl = jt("Horse 1", T_LABEL, C_DIM).next_to(a1, LEFT, buff=0.35)
        w_box = opbox("W", C_DIM, 0.62).next_to(a1, RIGHT, buff=0.7)
        a_to_w = arr(a1.get_right(), w_box.get_left(), C_DIM, sw=2.0)
        trio = VGroup()
        for nm, col, meaning in [("q_1", C_Q, "what it is looking for"),
                                 ("k_1", C_K, "what it offers"),
                                 ("v_1", C_V, "what it would contribute")]:
            vec = hvec(vs(40 + len(trio), 5), col, cell=0.20)
            lab = mt(nm, col, 0.5).next_to(vec, LEFT, buff=0.18)
            mean = jt(meaning, T_LABEL, col).next_to(vec, RIGHT, buff=0.35)
            trio.add(VGroup(lab, vec, mean))
        trio.arrange(DOWN, buff=0.34, aligned_edge=LEFT).next_to(w_box, RIGHT, buff=0.8)
        w_to_t = VGroup(*[arr(w_box.get_right(), t[0].get_left(), C_DIM, sw=1.6) for t in trio])
        # 表示済みの A ごと fit し直すと絵が飛ぶので、1 頭だけの構図に組み直して移す
        fit(VGroup(a1_lbl, a1, a_to_w, w_box, trio, w_to_t))
        self.play(ReplacementTransform(A[0], a1), ReplacementTransform(names[0], a1_lbl),
                  FadeOut(VGroup(*[VGroup(A[i], names[i]) for i in range(1, n)])), run_time=0.9)
        self.play(GrowArrow(a_to_w), FadeIn(w_box), run_time=0.6)
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(w_to_t[i]), FadeIn(trio[i]))
                                for i in range(3)], lag_ratio=0.3), run_time=1.6)
        self.wait(1.6)
        self.clear_stage()

        # 「照合 → 数字 → softmax → 重み」を数で見せる。矢印の太さだけで
        # 重みを表すと、何が計算されたのか読み取れない
        self.cap("Match horse 1's query against every key: one number each.", C_Q)
        q = VGroup(mt("q_1", C_Q, 0.55), hvec(vs(40, 5), C_Q, cell=0.20)).arrange(RIGHT, buff=0.18)
        q.move_to([-5.0, 0.0, 0])
        krows, dots, nums = VGroup(), VGroup(), VGroup()
        for i in range(n):
            k = VGroup(mt(f"k_{i + 1}", C_K, 0.5), hvec(vs(45 + i, 5), C_K, cell=0.18))
            k.arrange(RIGHT, buff=0.16).move_to([-1.9, ys[i], 0])
            krows.add(k)
            dots.add(arr(q.get_right(), k.get_left(), C_DIM, sw=1.8, buff=0.15))
            nums.add(jt(f"{scores[i]:.2f}", T_HEAD, WHITE).next_to(k, RIGHT, buff=0.5))
        dot_lbl = mt(r"q_1\cdot k_j", C_DIM, 0.55).next_to(VGroup(*nums), UP, buff=0.45)
        fit(VGroup(q, krows, dots, nums, dot_lbl))
        self.play(FadeIn(q), LaggedStart(*[FadeIn(k) for k in krows], lag_ratio=0.1), run_time=0.9)
        self.play(FadeIn(dot_lbl),
                  LaggedStart(*[AnimationGroup(GrowArrow(d), FadeIn(v))
                                for d, v in zip(dots, nums)], lag_ratio=0.18), run_time=1.6)
        self.wait(1.2)

        self.cap("Softmax turns them into weights that sum to one.", C_SCORE)
        sm = chip("softmax", C_SCORE, T_LABEL).next_to(VGroup(*nums), RIGHT, buff=0.7)
        wnums = VGroup(*[jt(f"{w:.2f}", T_HEAD, C_SCORE).next_to(sm, RIGHT, buff=0.7)
                         .set_y(nums[i].get_y()) for i, w in enumerate(weights)])
        extra = VGroup(sm, wnums)
        shown = VGroup(q, krows, dots, nums, dot_lbl)
        # softmax 側が増えたぶん右に伸びるので、その場で中央へ寄せ直す
        dx = CONTENT_C[0] - VGroup(shown, extra).get_center()[0]
        extra.shift(RIGHT * dx)
        self.play(shown.animate.shift(RIGHT * dx), run_time=0.6)
        sm_arrows = VGroup(*[arr(nums[i].get_right(), wnums[i].get_left(), C_DIM, sw=1.4, buff=0.25)
                             for i in range(n)])
        self.play(FadeIn(sm), run_time=0.4)
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(sm_arrows[i]),
                                               TransformFromCopy(nums[i], wnums[i]))
                                for i in range(n)], lag_ratio=0.15), run_time=1.4)
        self.wait(1.4)
        self.clear_stage()

        self.cap("Its new vector is the Values, mixed in those proportions.",
                 C_V)
        # 係数つきの和は 1 本の数式で出す。v をセル列で描くと項が 4 つ並んだ時点で
        # 横に伸びきり、fit() が全体を縮めて読めなくなる
        rhs = mt(r" + ".join(rf"{weights[i]:.2f}\,v_{{{i + 1}}}" for i in range(n)), WHITE, 0.85)
        out = hvec(vs(60, 6), C_ABILITY, cell=0.26)
        out_lbl = mt("a_1'", C_ABILITY, 0.7).next_to(out, LEFT, buff=0.2)
        eq = mt("=", WHITE, 0.8)
        line = VGroup(VGroup(out_lbl, out), eq, rhs).arrange(RIGHT, buff=0.45)
        note = jt("the horses it attends to are the ones that shape its vector",
                  18, C_DIM).next_to(line, DOWN, buff=0.55)
        fit(VGroup(line, note))
        self.play(Write(rhs), run_time=1.1)
        self.play(FadeIn(eq), FadeIn(out_lbl), FadeIn(out, scale=1.1), run_time=0.7)
        self.play(FadeIn(note, shift=UP * 0.12), run_time=0.5)
        self.wait(1.5)
        self.clear_stage()

        self.cap("Every horse at once: the attention matrix.",
                 C_SCORE)
        atts = [weights,
                [0.14, 0.60, 0.16, 0.10],
                [0.20, 0.14, 0.52, 0.14],
                [0.10, 0.12, 0.16, 0.62]]
        csz = 0.78
        grid, cellnums = VGroup(), VGroup()
        for i in range(n):
            for j in range(n):
                sq = Square(csz, stroke_width=1.1, stroke_color=C_SCORE,
                            fill_color=C_SCORE, fill_opacity=0.08 + 0.85 * atts[i][j])
                sq.move_to([-csz * 1.5 + j * csz, 1.0 - i * csz, 0])
                grid.add(sq)
                if i == 0:
                    # 薄いセルの上では暗い字が沈むので、塗りの濃さで字色を変える
                    ink = BG if atts[i][j] > 0.35 else WHITE
                    cellnums.add(jt(f"{atts[i][j]:.2f}", T_NOTE, ink, weight=BOLD).move_to(sq))
        rlab = VGroup(*[mt(f"i={i + 1}", WHITE, 0.42).next_to(grid[i * n], LEFT, buff=0.14)
                        for i in range(n)])
        clab = VGroup(*[mt(f"j={j + 1}", WHITE, 0.42).next_to(grid[j], UP, buff=0.10)
                        for j in range(n)])
        row1 = SurroundingRectangle(VGroup(*grid[0:n]), color=C_Q, buff=0.04, corner_radius=0.04)
        row1_l = jt("the weights we just computed", T_LABEL, C_Q)
        att_eq = mt(r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}"
                    r"\!\left(\tfrac{QK^\top}{\sqrt d}\right)V", WHITE, 0.68)
        mask = VGroup(
            jt("4 heads, 2 layers; padded slots are masked out", T_LABEL, C_DIM),
            jt("an 8-horse race and an 18-horse race run through the same weights",
               17, C_DIM),
        ).arrange(DOWN, buff=0.16)
        # 式と注記は行列の下ではなく横に置く。下へ積むと縦が伸びて、fit が行列
        # そのものを小さくしてしまう
        core = VGroup(grid, cellnums, rlab, clab, row1)
        row1_l.next_to(VGroup(grid, clab), UP, buff=0.26)
        side = VGroup(att_eq, mask).arrange(DOWN, buff=0.40)
        fit(VGroup(VGroup(row1_l, core), side).arrange(RIGHT, buff=0.9))
        self.play(LaggedStart(*[GrowFromCenter(c) for c in grid], lag_ratio=0.03),
                  FadeIn(rlab), FadeIn(clab), run_time=1.2)
        self.play(Create(row1), FadeIn(cellnums), FadeIn(row1_l), run_time=0.9)
        self.wait(1.0)
        self.play(Write(att_eq), FadeIn(mask, shift=UP * 0.12), run_time=1.0)
        self.wait(1.4)
        self.clear_stage()

    # ============================================================ 7. the price
    def act7_price(self):
        self.set_act(7, "the price", C_ODDS)
        self.cap("Only at the very end does the market get a vote.", C_ODDS)
        n = 4
        svals = [0.9, 0.35, 0.6, 0.2]
        rows = VGroup()
        for i in range(n):
            abl = hvec(vs(30 + i, 8), C_ABILITY, cell=0.24)
            plus = mt(r"\oplus", WHITE, 0.62)
            odds = hvec(vs(9 + i, 2), C_ODDS, cell=0.24)
            head = chip("head MLP", C_SCORE, T_LABEL)
            a1 = mt(r"\rightarrow", WHITE, 0.7)
            sc = valdot(svals[i], C_SCORE, 0.52)
            slb = mt(f"s_{i + 1}", WHITE, 0.5).next_to(sc, RIGHT, buff=0.12)
            rows.add(VGroup(abl, plus, odds, head, a1, VGroup(sc, slb)).arrange(RIGHT, buff=0.32))
        rows.arrange(DOWN, buff=0.40)
        hdr = VGroup(
            jt("ability", T_LABEL, C_ABILITY).next_to(rows[0][0], UP, buff=0.30),
            jt("odds", T_LABEL, C_ODDS).next_to(rows[0][2], UP, buff=0.30),
            jt("score", T_LABEL, C_SCORE).next_to(rows[0][5], UP, buff=0.30),
        )
        eq = mt(r"s_i=\mathrm{MLP}\big(\mathrm{LN}(a_i')\ \oplus\ \mathrm{odds}_i\big)", WHITE, 0.66)
        eq.next_to(rows, DOWN, buff=0.45)
        fit(VGroup(hdr, rows, eq))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows], lag_ratio=0.15),
                  FadeIn(hdr), run_time=1.3)
        self.play(Write(eq), run_time=0.9)
        self.cap("That split -- ability from price -- is the whole architecture.",
                 C_ABILITY)
        self.wait(1.2)
        carried = self.carry(VGroup(*[rows[i][5] for i in range(n)]))
        self.clear_stage(keep=[carried])

        self.cap("One set of scores, one temperature, every bet type.",
                 C_SCORE)
        col = VGroup()
        for i, v in enumerate(svals):
            sc = valdot(v, C_SCORE, 0.52)
            lb = mt(f"s_{i + 1}", WHITE, 0.5).next_to(sc, RIGHT, buff=0.12)
            col.add(VGroup(sc, lb))
        col.arrange(DOWN, buff=0.26).move_to([-5.6, 0.1, 0])
        tbox = chip("divide by T", C_Q, T_LABEL).move_to([-3.5, 0.1, 0])
        tarr = arr(col.get_right(), tbox.get_left(), C_Q, sw=2.4)
        win_lbl = mt(r"\mathrm{softmax}(s_i/T)", C_SCORE, 0.55)
        pl_lbl = mt(r"\mathrm{Plackett\!-\!Luce}(s/T)", C_V, 0.55)
        cmb_lbl = mt(r"P_{\mathrm{PL}}(\text{combination})", C_ODDS, 0.55)
        win_bars = probbars([0.46, 0.14, 0.28, 0.12], ["H1", "H2", "H3", "H4"], C_SCORE, hmax=1.15)
        pl_bars = probbars([0.83, 0.42, 0.66, 0.38], ["H1", "H2", "H3", "H4"], C_V, hmax=1.15)
        cmb = VGroup(
            jt("H1-H3   8.1%", T_LABEL, C_ODDS),
            jt("H1-H2   4.4%", T_LABEL, C_ODDS),
            jt("H1-H3-H2   1.2%", T_LABEL, C_ODDS),
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        rows2 = VGroup()
        for lbl, out, name, c in [(win_lbl, win_bars, "win", C_SCORE),
                                  (pl_lbl, pl_bars, "place, top 3", C_V),
                                  (cmb_lbl, cmb, "quinella, trio, trifecta", C_ODDS)]:
            tag = jt(name, T_NOTE, c)
            body = VGroup(lbl, out).arrange(RIGHT, buff=0.7)
            tag.next_to(body, UP, buff=0.12).align_to(body, LEFT)
            rows2.add(VGroup(tag, body))
        rows2.arrange(DOWN, buff=0.52, aligned_edge=LEFT).next_to(tbox, RIGHT, buff=1.0)
        fan = VGroup(*[arr(tbox.get_right(), r.get_left(), C_DIM, sw=1.8) for r in rows2])
        tnote = jt("one T for all of them, fitted by minimising the winner's NLL", T_LABEL, C_Q)
        stage = VGroup(col, tarr, tbox, fan, rows2)
        tnote.next_to(stage, DOWN, buff=0.34)
        fit(VGroup(stage, tnote))
        self.play(ReplacementTransform(carried, col), run_time=0.9)
        self.play(GrowArrow(tarr), FadeIn(tbox), run_time=0.6)
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(r))
                                for a, r in zip(fan, rows2)], lag_ratio=0.3), run_time=1.8)
        self.play(FadeIn(tnote, shift=UP * 0.12), run_time=0.6)
        self.wait(1.6)
        self.clear_stage()

    # ============================================================ 8. training
    def act8_training(self):
        """What training does to the machine acts 5-7 just built.

        文章のパネルから始めると、何を見せられているのかが最後まで分からない。
        まず「採点 → 賭ける → 増えたか → 直す」の輪を 1 枚で出し、そのあとで
        W の中身・止める場所・二段階に降りる。
        """
        self.set_act(8, "training", C_SCORE)
        self.cap("Score the race, bet it, see what the money did, adjust.",
                 C_SCORE)

        model = chip("the model", C_ABILITY, T_LABEL, caps=True)
        model_sub = jt("encoder + transformer + head", T_MICRO, C_DIM)
        scores = VGroup(*[valdot(v, C_SCORE, 0.26) for v in [0.9, 0.35, 0.6, 0.2]])
        scores.arrange(DOWN, buff=0.10)
        scores = VGroup(scores, mt("s_i", C_SCORE, 0.45).next_to(scores, DOWN, buff=0.14))
        bet = chip("bet the race", C_ODDS, T_LABEL, caps=True)
        bet_sub = jt("at the odds actually paid", T_MICRO, C_ODDS)
        money = mt("W", WHITE, 1.0)
        money_sub = jt("what the money did", T_MICRO, C_DIM)

        row = VGroup(model, scores, bet, money).arrange(RIGHT, buff=0.85)
        model_sub.next_to(model, DOWN, buff=0.16)
        bet_sub.next_to(bet, DOWN, buff=0.16)
        money_sub.next_to(money, DOWN, buff=0.20)
        links = VGroup(*[arr(row[i].get_right(), row[i + 1].get_left(), C_DIM, sw=2.0, buff=0.10)
                         for i in range(3)])
        back = CurvedArrow(money.get_bottom() + DOWN * 0.75, model.get_bottom() + DOWN * 0.75,
                           angle=-TAU / 7, color=C_SCORE, stroke_width=3, tip_length=0.2)
        back_lbl = jt("adjust every weight in the direction that grew the money", T_LABEL, C_SCORE)
        back_lbl.next_to(back, DOWN, buff=0.14)
        fit(VGroup(row, model_sub, bet_sub, money_sub, links, back, back_lbl))

        self.play(FadeIn(model, shift=RIGHT * 0.2), FadeIn(model_sub), run_time=0.7)
        self.play(GrowArrow(links[0]), FadeIn(scores), run_time=0.6)
        self.play(GrowArrow(links[1]), FadeIn(bet), FadeIn(bet_sub), run_time=0.7)
        self.play(GrowArrow(links[2]), FadeIn(money, scale=1.2), FadeIn(money_sub), run_time=0.7)
        self.wait(0.6)
        self.play(Create(back), FadeIn(back_lbl), run_time=1.0)
        self.wait(1.4)

        # W だけ残して上へ送り、その中身に降りる
        self.cap("So everything turns on how W is defined.", C_ODDS)
        keep = VGroup(money, money_sub)
        self.play(FadeOut(VGroup(row[0], row[1], row[2], model_sub, bet_sub, links, back,
                                 back_lbl), shift=LEFT * 0.18),
                  keep.animate.move_to([0.0, 1.85, 0]).scale(0.9), run_time=0.9)
        eq1 = mt(r"W \;=\; 1 + c\,\big(p_{\text{winner}}\cdot o_{\text{winner}} - 1\big)", WHITE, 0.82)
        eq2 = mt(r"\mathcal{L} \;=\; -\,\mathbb{E}\big[\log W\big]", WHITE, 0.82)
        n1 = jt("o = the odds actually paid, not a feature", T_LABEL, C_ODDS)
        n2 = VGroup(jt("c = 0.25 keeps the odds inside the gradient", T_NOTE, C_DIM),
                    jt("at c = 1 it collapses into plain cross-entropy", T_NOTE, C_DIM))
        n2.arrange(DOWN, buff=0.14)
        block = VGroup(eq1, n1, eq2, n2).arrange(DOWN, buff=0.30)
        block.next_to(keep, DOWN, buff=0.55)
        fit(VGroup(keep, block), h=CONTENT_H - 0.2)
        self.play(Write(eq1), FadeIn(n1, shift=UP * 0.1), run_time=1.2)
        self.play(Write(eq2), FadeIn(n2, shift=UP * 0.1), run_time=1.0)
        self.wait(1.6)
        self.clear_stage()

        self.cap("The model that gets kept is the one that pays.",
                 C_SCORE)
        ax_x = Line([-3.4, -1.35, 0], [3.6, -1.35, 0], stroke_color=C_DIM, stroke_width=2)
        ax_y = Line([-3.4, -1.35, 0], [-3.4, 1.45, 0], stroke_color=C_DIM, stroke_width=2)
        xlab = jt("epochs", T_LABEL, C_DIM).next_to(ax_x, DOWN, buff=0.16)
        ndcg = curve([(-3.4, -1.1), (-2.0, -0.25), (-0.5, 0.35), (1.2, 0.7), (3.4, 0.85)], C_MARKET)
        roi = curve([(-3.4, -1.2), (-2.3, -0.1), (-1.3, 0.95), (0.1, 0.4), (1.6, -0.15),
                     (3.4, -0.6)], C_SCORE)
        ndcg_l = jt("ranking accuracy", T_LABEL, C_MARKET).next_to(ndcg.get_end(), RIGHT, buff=0.18)
        roi_l = jt("validation win ROI", T_LABEL, C_SCORE).next_to(roi.get_end(), RIGHT, buff=0.18)
        peak = Dot([-1.3, 0.95, 0], color=C_SCORE, radius=0.09)
        drop = DashedLine([-1.3, 0.95, 0], [-1.3, -1.35, 0], stroke_color=C_SCORE,
                          stroke_width=2, dash_length=0.1).set_stroke(opacity=0.6)
        stop_l = jt("early stop here", T_LABEL, C_SCORE).next_to(peak, UP, buff=0.16)
        plot = VGroup(ax_x, ax_y, xlab, ndcg, roi, ndcg_l, roi_l, peak, drop, stop_l)
        note = jt("`--monitor valid_tansho_roi`", T_LABEL, C_SCORE).next_to(plot, DOWN, buff=0.34)
        fit(VGroup(plot, note))
        self.play(Create(ax_x), Create(ax_y), FadeIn(xlab), run_time=0.5)
        self.play(Create(ndcg), FadeIn(ndcg_l), run_time=1.0)
        self.play(Create(roi), FadeIn(roi_l), run_time=1.0)
        self.play(FadeIn(peak, scale=0.5), Create(drop), FadeIn(stop_l),
                  FadeIn(note, shift=UP * 0.12), run_time=0.9)
        self.wait(1.6)
        self.clear_stage()

        self.cap("The loop runs twice. Only the second is about money.",
                 C_ABILITY)
        s1 = chip("stage 1   plackett-luce", C_V, T_LABEL, caps=True)
        s2 = chip("stage 2   multi", C_SCORE, T_LABEL, caps=True)
        s1_sub = VGroup(jt("learn the finishing order", T_LABEL, C_DIM),
                        jt("a proper scoring rule", T_LABEL, C_DIM)).arrange(DOWN, buff=0.14)
        s2_sub = VGroup(jt("learn the money", T_LABEL, C_DIM),
                        jt("`log_growth` + 0.01 `combo_nll`", T_LABEL, C_DIM)).arrange(DOWN, buff=0.14)
        s1_sub.next_to(s1, DOWN, buff=0.22)
        s2_sub.next_to(s2, DOWN, buff=0.22)
        stages = VGroup(VGroup(s1, s1_sub), VGroup(s2, s2_sub)).arrange(RIGHT, buff=1.6)
        link = arr(stages[0].get_right(), stages[1].get_left(), C_DIM, sw=3.0, buff=0.15)
        link_lbl = jt("`--init-from`", T_NOTE, C_DIM).next_to(link, UP, buff=0.14)
        fit(VGroup(stages, link, link_lbl))
        self.play(FadeIn(stages[0], shift=RIGHT * 0.2), run_time=0.8)
        self.wait(1.0)
        self.play(GrowArrow(link), FadeIn(link_lbl), FadeIn(stages[1], shift=RIGHT * 0.2),
                  run_time=0.9)
        self.wait(1.8)
        self.clear_stage()
