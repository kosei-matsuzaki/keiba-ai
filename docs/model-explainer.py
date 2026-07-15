"""Horse-racing model architecture explainer (3Blue1Brown style / manim), all-English.

Faithful computation blocks:
  vector = column of intensity cells / GRU = recurrent update (gates r,z) /
  MLP = neuron layers + weighted edges (forward pass) /
  attention = build Q,K,V per column -> scores -> softmax -> weighted sum.
Ends with a tour of the actual web app.

Palette (consistent throughout):
  aggregate=emerald / past-race x_t=aqua / hidden h=blue / race=violet / odds=amber /
  ability(a_i)=fuchsia / Query=cyan / Key=yellow / Value=lime / score=rose

Render:
  manim -ql arch_math.py ModelMath                      # preview
  manim -r 1920,1080 --fps 30 arch_math.py ModelMath    # 1080p30 final
"""
from manim import *
import numpy as np
import os

FONT = "Segoe UI"   # 全テキストをタイトルと同じ Segoe UI Bold で統一

C_AGG     = "#34d399"   # aggregate features (emerald)
C_PAST    = "#5eead4"   # past-race result vector x_t (aqua) -- distinct from hidden state
C_HIST    = "#60a5fa"   # hidden state h / history vector (blue)
C_RACE    = "#a78bfa"   # race features (violet)
C_ODDS    = "#fbbf24"   # odds (amber)
C_ABILITY = "#e879f9"   # ability vector / a_i (fuchsia)
C_Q       = "#22d3ee"   # Query (cyan)
C_K       = "#facc15"   # Key (yellow)
C_V       = "#a3e635"   # Value (lime)
C_SCORE   = "#fb7185"   # score / probability / attention weight (rose)
C_DIM     = "#9aa4b2"
BG        = "#0c1420"

POOL = [0.25, 0.82, 0.48, 0.35, 0.9, 0.6, 0.18, 0.72, 0.42, 0.55,
        0.86, 0.3, 0.66, 0.5, 0.22, 0.78, 0.4, 0.7]

# act7 の Web スクリーンショットの場所 (リポジトリルートから実行する想定)
IMG_DIR = os.environ.get("KEIBA_IMG_DIR", "docs/images")


def jt(s, size=24, color=WHITE, weight=BOLD):
    """All text in Segoe UI Bold (same look as the title)."""
    return Text(s, font=FONT, font_size=size, color=color, weight=BOLD)


def arr(start, end, color, sw=3.0, buff=0.1):
    return Arrow(start, end, buff=buff, color=color, stroke_width=sw,
                 tip_length=0.16, max_tip_length_to_length_ratio=0.28)


def vs(seed, k=5):
    return [POOL[(seed * 5 + i * 3) % len(POOL)] for i in range(k)]


def vvec(vals, color, cell=0.32):
    g = VGroup()
    for v in vals:
        sq = Square(cell, stroke_width=1.0, stroke_color=color)
        sq.set_fill(color, opacity=0.10 + 0.72 * float(v))
        g.add(sq)
    g.arrange(DOWN, buff=0.035)
    return g


def opbox(tex, color, s=0.72):
    b = RoundedRectangle(width=s, height=s, corner_radius=0.09,
                         stroke_color=color, stroke_width=2.5,
                         fill_color=color, fill_opacity=0.14)
    t = MathTex(tex, color=color).scale(0.55).move_to(b)
    return VGroup(b, t)


def nlayer(k, color, x, span=1.15):
    ys = np.linspace(span, -span, k)
    return VGroup(*[Circle(radius=0.11, stroke_color=color, stroke_width=2,
                           fill_color=color, fill_opacity=0.15).move_to([x, y, 0])
                    for y in ys])


def edges(l1, l2, color=C_DIM):
    e = VGroup()
    for a in l1:
        for b in l2:
            e.add(Line(a.get_center(), b.get_center(),
                       stroke_color=color, stroke_width=0.8, stroke_opacity=0.28))
    return e


class ModelMath(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = jt("Horse-Racing Prediction with a Set Transformer", 34, weight=BOLD).to_edge(UP, buff=0.5)
        sub = jt("Inside the computation — separating ability from market information",
                 21, color=C_DIM).next_to(title, DOWN, buff=0.22)
        self.play(Write(title), FadeIn(sub, shift=UP*0.2))
        self.wait(1.4)
        self.play(title.animate.scale(0.6).to_corner(UL, buff=0.35), FadeOut(sub))

        self._cap = None

        def cap(s, color=WHITE):
            new = jt(s, 22, color=color).to_edge(DOWN, buff=0.4)
            if new.width > 12.8:
                new.scale_to_fit_width(12.8)
            if self._cap is None:
                self.play(FadeIn(new), run_time=0.8)
            else:
                self.play(ReplacementTransform(self._cap, new), run_time=0.9)
            self._cap = new
            self.wait(0.7)

        self.act1_vectors(cap)
        self.act2_gru(cap)
        self.act3_mlp(cap)
        self.act4_attention(cap)
        self.act5_head(cap)
        self.act6_bet(cap)
        self.act7_screens(cap)

        self.play(FadeOut(self._cap))
        closing = jt("Separate ability from market info, and train by directly optimizing ROI",
                     24).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(closing, shift=UP*0.2))
        self.wait(2.5)

    # ============================================================
    def act1_vectors(self, cap):
        cap("A race is a variable-size set of horses (4 here). Each horse is described by feature vectors.")
        vecs = VGroup()
        for i in range(4):
            v = vvec(vs(i + 1, 8), C_AGG, cell=0.26)   # thin & dense (matches later acts)
            lab = jt(f"Horse {i+1}", 18, color=WHITE)
            grp = VGroup(v, lab)
            lab.next_to(v, DOWN, buff=0.2)
            vecs.add(grp)
        vecs.arrange(RIGHT, buff=1.3).move_to(UP*0.3)
        self.play(LaggedStart(*[FadeIn(g, shift=UP*0.3) for g in vecs], lag_ratio=0.2), run_time=1.6)
        self.wait(1.0)
        brace = Brace(vecs, DOWN, buff=0.3, color=C_DIM)
        btext = jt("N horses = variable length (N = 4 here)", 18, color=C_DIM).next_to(brace, DOWN, buff=0.16)
        self.play(GrowFromCenter(brace), FadeIn(btext))
        self.wait(1.6)

        # pick out horse 1 (like the focus later on)
        keep = vecs[0]
        self.play(FadeOut(brace), FadeOut(btext),
                  FadeOut(vecs[1]), FadeOut(vecs[2]), FadeOut(vecs[3]),
                  keep.animate.scale(1.15).move_to([-5.4, 0.4, 0]), run_time=1.2)
        self.wait(0.6)

        # reveal the three feature groups (with example fields)
        cap("Its features come in three groups (examples shown): aggregate, past-race results, race", C_AGG)
        groups = [
            (C_AGG,  6, "Aggregate",             "Jockey · Pedigree\nImpost · Body wt.\nPost · Age / Sex", -3.6, 11),
            (C_PAST, 4, "Result (per past race)", "Finish · Time\nMargin · Last 3F",                        0.4, 21),
            (C_RACE, 3, "Race",                   "Distance · Going\nClass",                                4.3, 5),
        ]
        gvecs, gnames, gfields = VGroup(), VGroup(), VGroup()
        for col, ncell, name, fields, x, seed in groups:
            gv = vvec(vs(seed, ncell), col, 0.26).move_to([x, 0.55, 0])
            gname = jt(name, 18, col).next_to(gv, UP, buff=0.3)
            gfield = jt(fields, 15, C_DIM, ).next_to(gv, DOWN, buff=0.35)
            gvecs.add(gv); gnames.add(gname); gfields.add(gfield)
        # keep (horse 1) morphs into the aggregate group, others fade in
        self.play(ReplacementTransform(keep, gvecs[0]),
                  FadeIn(gnames[0]), FadeIn(gfields[0]), run_time=1.1)
        self.play(LaggedStart(
            AnimationGroup(FadeIn(gvecs[1], shift=UP*0.2), FadeIn(gnames[1]), FadeIn(gfields[1])),
            AnimationGroup(FadeIn(gvecs[2], shift=UP*0.2), FadeIn(gnames[2]), FadeIn(gfields[2])),
            lag_ratio=0.5), run_time=2.0)
        self.wait(2.2)
        self._act1_keep = None
        self._act1_extra = VGroup(gvecs, gnames, gfields)

    # ============================================================
    def act2_gru(self, cap):
        cap("The past-race results are a sequence — a GRU folds them into one history vector", C_HIST)

        cell_x = [-4.5, -1.5, 1.5]
        hid_x = [-6.2, -3.0, 0.0, 3.0]
        hvals = [[0.05]*5, vs(10, 5), vs(13, 5), vs(16, 5)]
        h_mobs = [vvec(hvals[j], C_HIST, cell=0.24).move_to([hid_x[j], 0.55, 0]) for j in range(4)]
        h_lbls = [MathTex(f"h_{j}", color=C_HIST).scale(0.62).next_to(h_mobs[j], UP, buff=0.14) for j in range(4)]

        cells = VGroup()
        for cx in cell_x:
            b = RoundedRectangle(width=1.0, height=1.1, corner_radius=0.1,
                                 stroke_color=C_HIST, stroke_width=2.5,
                                 fill_color=C_HIST, fill_opacity=0.10).move_to([cx, 0.55, 0])
            gates = MathTex(r"r,z,\tilde{h}", color=C_HIST).scale(0.42).move_to(b)
            cells.add(VGroup(b, gates))

        x_mobs = []
        for t, cx in enumerate(cell_x):
            xv = vvec(vs(20 + t, 5), C_PAST, cell=0.22).move_to([cx, -1.7, 0])
            xl = MathTex(f"x_{{t-{3-t}}}", color=C_PAST).scale(0.5).next_to(xv, DOWN, buff=0.1)
            x_mobs.append(VGroup(xv, xl))

        self.play(FadeOut(self._act1_extra),
                  FadeIn(h_mobs[0]), FadeIn(h_lbls[0]), run_time=1.1)
        h0_note = jt("initial hidden state", 17, color=C_HIST).next_to(h_mobs[0], DOWN, buff=0.35).shift(RIGHT*0.4)
        self.play(FadeIn(h0_note))
        self.wait(1.2)

        eq = MathTex(r"h_t=(1-z_t)\odot h_{t-1}+z_t\odot \tilde{h}_t",
                     color=WHITE).scale(0.7).move_to([-0.3, 2.4, 0])
        self.play(Write(eq), run_time=1.2)
        self.wait(0.6)

        x_note = jt("each x = one past race's result vector", 17, color=C_PAST).move_to([-0.2, -3.05, 0])
        for t in range(3):
            self.play(FadeIn(cells[t]), FadeIn(x_mobs[t], shift=UP*0.2), run_time=0.9)
            if t == 0:
                self.play(FadeIn(x_note))
                self.wait(1.0)
                self.play(FadeOut(h0_note))
            a_h = arr(h_mobs[t].get_right(), cells[t][0].get_left(), C_HIST, sw=2.6)
            a_x = arr(x_mobs[t][0].get_top(), cells[t][0].get_bottom(), C_PAST, sw=2.6)
            self.play(GrowArrow(a_h), GrowArrow(a_x), run_time=0.8)
            self.play(Indicate(cells[t][0], color=C_SCORE, scale_factor=1.1), run_time=0.7)
            a_out = arr(cells[t][0].get_right(), h_mobs[t+1].get_left(), C_HIST, sw=2.6)
            self.play(GrowArrow(a_out), TransformFromCopy(h_mobs[t], h_mobs[t+1]),
                      FadeIn(h_lbls[t+1]), run_time=1.0)
            self.wait(0.4)

        box = SurroundingRectangle(h_mobs[3], color=C_HIST, buff=0.12, corner_radius=0.08)
        hist_lbl = jt("history vector", 18, color=C_HIST).next_to(box, RIGHT, buff=0.3)
        self.play(Create(box), FadeIn(hist_lbl))
        self.wait(1.8)

        junk = VGroup(*h_mobs, *h_lbls, cells, *x_mobs, eq, box, hist_lbl, x_note)
        self.play(FadeOut(junk),
                  *[FadeOut(m) for m in self.mobjects if isinstance(m, Arrow)], run_time=1.0)

    # ============================================================
    def act3_mlp(self, cap):
        cap("Concatenate aggregate + history + race features, then a multi-layer MLP encodes ability", C_ABILITY)
        agg = vvec(vs(2, 4), C_AGG, cell=0.26)
        hist = vvec(vs(13, 4), C_HIST, cell=0.26)
        race = vvec(vs(5, 3), C_RACE, cell=0.26)
        concat = VGroup(agg, hist, race).arrange(DOWN, buff=0.06).to_edge(LEFT, buff=1.2)
        la = jt("Aggregate", 15, C_AGG).next_to(agg, LEFT, buff=0.2)
        lh = jt("History", 15, C_HIST).next_to(hist, LEFT, buff=0.2)
        lr = jt("Race", 15, C_RACE).next_to(race, LEFT, buff=0.2)
        self.play(FadeIn(concat), FadeIn(la), FadeIn(lh), FadeIn(lr))
        self.wait(1.0)

        l_in = nlayer(6, C_DIM, -2.4, span=1.3)
        l_h1 = nlayer(5, WHITE, -0.6, span=1.15)
        l_h2 = nlayer(5, WHITE, 1.2, span=1.15)
        l_out = nlayer(4, C_ABILITY, 3.0, span=0.95)
        e1 = edges(l_in, l_h1); e2 = edges(l_h1, l_h2); e3 = edges(l_h2, l_out)
        self.play(Create(l_in), run_time=0.6)
        in_arr = VGroup(*[arr(concat.get_right(), n.get_left(), C_DIM, sw=2) for n in l_in])
        self.play(*[GrowArrow(a) for a in in_arr], run_time=0.9)
        self.play(FadeIn(e1), FadeIn(e2), FadeIn(e3),
                  Create(l_h1), Create(l_h2), Create(l_out), run_time=1.0)
        gelu = jt("each layer:  linear -> GELU", 18, color=C_DIM).next_to(l_h1, UP, buff=1.2)
        self.play(FadeIn(gelu))
        self.wait(0.5)
        for le, ln, col in [(e1, l_h1, WHITE), (e2, l_h2, WHITE), (e3, l_out, C_ABILITY)]:
            self.play(LaggedStart(*[ShowPassingFlash(ed.copy().set_stroke(C_SCORE, 2.2), time_width=0.6)
                                    for ed in le], lag_ratio=0.004, run_time=1.2),
                      LaggedStart(*[n.animate.set_fill(col, 0.85) for n in ln], lag_ratio=0.05))
        ability = vvec(vs(30, 5), C_ABILITY, cell=0.3).next_to(l_out, RIGHT, buff=0.75)
        abl_lbl = jt("ability vector", 16, C_ABILITY).next_to(ability, DOWN, buff=0.16)
        out_arr = VGroup(*[arr(n.get_right(), ability.get_left(), C_ABILITY, sw=2) for n in l_out])
        self.play(*[GrowArrow(a) for a in out_arr], FadeIn(ability), FadeIn(abl_lbl), run_time=1.0)
        self.wait(1.8)
        junk = VGroup(concat, la, lh, lr, l_in, l_h1, l_h2, l_out, e1, e2, e3,
                      gelu, abl_lbl, ability, in_arr, out_arr)
        self.play(FadeOut(junk),
                  *[FadeOut(m) for m in self.mobjects if isinstance(m, Arrow)], run_time=1.0)

    # ============================================================
    def act4_attention(self, cap):
        n = 4
        ys = [2.0, 0.7, -0.6, -1.9]
        x_a, x_w, x_q, x_k, x_v = -6.3, -5.15, -3.85, -2.7, -1.55
        weights = [0.62, 0.12, 0.18, 0.08]
        cap("Set Transformer: each horse looks at every other horse to update itself", C_ABILITY)

        # ability vectors a_1..a_4 (one per row)
        A = VGroup(*[vvec(vs(30 + i, 4), C_ABILITY, 0.12).move_to([x_a, ys[i], 0]) for i in range(n)])
        a_lbls = VGroup(*[MathTex(f"a_{i+1}", color=C_ABILITY).scale(0.45).next_to(A[i], LEFT, buff=0.14)
                          for i in range(n)])
        self.play(LaggedStart(*[FadeIn(a) for a in A], lag_ratio=0.15), FadeIn(a_lbls), run_time=1.4)
        self.wait(0.6)

        # per horse: a_i -> W box -> q_i, k_i, v_i (side by side)
        cap("Each ability passes through its own W box → Query, Key, Value (side by side)", C_K)
        Wb, Q, K, V, qkv_lbls, in_arrows = VGroup(), VGroup(), VGroup(), VGroup(), VGroup(), VGroup()
        for i in range(n):
            Wb.add(opbox(r"W", C_DIM, 0.5).move_to([x_w, ys[i], 0]))
            Q.add(vvec(vs(40 + i, 4), C_Q, 0.12).move_to([x_q, ys[i], 0]))
            K.add(vvec(vs(45 + i, 4), C_K, 0.12).move_to([x_k, ys[i], 0]))
            V.add(vvec(vs(50 + i, 4), C_V, 0.12).move_to([x_v, ys[i], 0]))
        for i in range(n):
            for mob, col in [(Q, C_Q), (K, C_K), (V, C_V)]:
                nm = {id(Q): "q", id(K): "k", id(V): "v"}[id(mob)]
                qkv_lbls.add(MathTex(f"{nm}_{i+1}", color=col).scale(0.32).next_to(mob[i], UP, buff=0.03))
        for i in range(n):
            aw = arr(A[i].get_right(), Wb[i].get_left(), C_DIM, sw=1.5)
            wt = arr(Wb[i].get_right(), Q[i].get_left(), C_DIM, sw=1.5)
            in_arrows.add(aw, wt)
            trip = VGroup(Q[i], K[i], V[i])
            labs = VGroup(*qkv_lbls[i*3:i*3+3])
            if i == 0:
                self.play(GrowArrow(aw), FadeIn(Wb[i]), run_time=0.6)
                self.play(GrowArrow(wt), TransformFromCopy(A[i], trip), FadeIn(labs), run_time=1.0)
                self.wait(0.4)
            else:
                self.play(GrowArrow(aw), FadeIn(Wb[i]), GrowArrow(wt),
                          TransformFromCopy(A[i], trip), FadeIn(labs), run_time=0.6)
        self.wait(0.8)

        # pick out horse 1 (like the earlier focus)
        cap("Pick out horse 1: match q1 to every key → softmax attention (thicker = higher)", C_Q)
        self.play(FadeOut(A), FadeOut(a_lbls), FadeOut(Wb), FadeOut(in_arrows),
                  *[Q[i].animate.set_opacity(0.2) for i in range(1, n)], run_time=1.0)
        q1box = SurroundingRectangle(VGroup(Q[0], qkv_lbls[0]), color=C_Q, buff=0.05, corner_radius=0.05)
        self.play(Create(q1box), Q[0].animate.scale(1.18), run_time=0.7)
        s_arrows = VGroup()
        for j in range(n):
            a = arr(Q[0].get_right(), K[j].get_left(), C_SCORE, sw=1.2 + 3.6*weights[j], buff=0.06)
            a.set_stroke(opacity=0.3 + 0.7*weights[j])
            s_arrows.add(a)
        self.play(LaggedStart(*[GrowArrow(a) for a in s_arrows], lag_ratio=0.18), run_time=1.8)
        self.wait(1.4)

        # weighted sum of values -> a1'
        cap("Weighted-average the Values by that attention → horse 1's new vector a1'", C_V)
        a1p = vvec(vs(60, 4), C_ABILITY, 0.14).move_to([3.0, 0.05, 0])
        a1p_lbl = MathTex(r"a_1'", color=C_ABILITY).scale(0.5).next_to(a1p, UP, buff=0.1)
        wsum = MathTex(r"a_1'=\sum_j A_{1j}\, v_j", color=WHITE).scale(0.58).next_to(a1p, RIGHT, buff=0.45)
        v_arrows = VGroup()
        for j in range(n):
            a = arr(V[j].get_right(), a1p.get_left(), C_V, sw=1.2 + 3.4*weights[j], buff=0.06)
            a.set_stroke(opacity=0.28 + 0.72*weights[j])
            v_arrows.add(a)
        self.play(LaggedStart(*[GrowArrow(a) for a in v_arrows], lag_ratio=0.15),
                  FadeIn(a1p), FadeIn(a1p_lbl), run_time=1.8)
        self.play(Write(wsum), run_time=1.0)
        self.wait(1.8)

        # generalize -> attention matrix -> formula
        cap("Do this for every horse → the attention matrix; repeat over layers", C_SCORE)
        allmobs = VGroup(Q, K, V, qkv_lbls, q1box, a1p, a1p_lbl, wsum)
        self.play(FadeOut(allmobs), FadeOut(s_arrows), FadeOut(v_arrows),
                  *[FadeOut(m) for m in self.mobjects if isinstance(m, Arrow)], run_time=1.0)
        atts = [[0.62, 0.12, 0.18, 0.08],
                [0.14, 0.60, 0.16, 0.10],
                [0.20, 0.14, 0.52, 0.14],
                [0.10, 0.12, 0.16, 0.62]]
        csz = 0.62
        grid = VGroup()
        for i in range(n):
            for j in range(n):
                sq = Square(csz, stroke_width=1.1, stroke_color=C_SCORE,
                            fill_color=C_SCORE, fill_opacity=0.08 + 0.85*atts[i][j])
                sq.move_to([-csz*1.5 + j*csz, 1.15 - i*csz, 0])
                grid.add(sq)
        rlab = VGroup(*[MathTex(f"i={i+1}").scale(0.42).next_to(grid[i*n], LEFT, buff=0.12) for i in range(n)])
        clab = VGroup(*[MathTex(f"j={j+1}").scale(0.42).next_to(grid[j], UP, buff=0.08) for j in range(n)])
        glab = jt("attention matrix  (row i = horse i's attention)", 18, color=C_SCORE).next_to(VGroup(grid, clab), UP, buff=0.35)
        self.play(LaggedStart(*[GrowFromCenter(c) for c in grid], lag_ratio=0.04),
                  FadeIn(rlab), FadeIn(clab), FadeIn(glab), run_time=1.8)
        att_eq = MathTex(r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\tfrac{QK^\top}{\sqrt d}\right)V",
                         color=WHITE).scale(0.66).next_to(grid, DOWN, buff=0.6)
        self.play(Write(att_eq), run_time=1.2)
        self.wait(2.0)
        self.play(FadeOut(VGroup(grid, rlab, clab, glab, att_eq)), run_time=1.0)

    # ============================================================
    def act5_head(self, cap):
        cap("Scoring head: normalized ability ⊕ standardized odds → score (ability→value separation)", C_ODDS)
        n = 4
        scores_vals = [0.9, 0.35, 0.6, 0.2]
        rows = VGroup()
        for i in range(n):
            abl = vvec(vs(30 + i, 4), C_ABILITY, cell=0.2)
            odds = vvec(vs(9 + i, 2), C_ODDS, cell=0.2)
            plus = MathTex(r"\oplus", color=WHITE).scale(0.6)
            head = opbox(r"\text{head}", C_SCORE, 0.58)
            a1 = MathTex(r"\rightarrow", color=WHITE).scale(0.7)
            sc = Square(0.42, stroke_color=C_SCORE, stroke_width=1.5,
                        fill_color=C_SCORE, fill_opacity=0.12 + 0.8*scores_vals[i])
            sclab = MathTex(f"s_{i+1}", color=WHITE).scale(0.5).next_to(sc, RIGHT, buff=0.1)
            row = VGroup(abl, plus, odds, head, a1, VGroup(sc, sclab)).arrange(RIGHT, buff=0.3)
            rows.add(row)
        rows.arrange(DOWN, buff=0.45).move_to(UP*0.05)
        hdr = VGroup(
            jt("Ability", 15, C_ABILITY).next_to(rows[0][0], UP, buff=0.28),
            jt("Odds", 15, C_ODDS).next_to(rows[0][2], UP, buff=0.28),
            jt("Score", 15, C_SCORE).next_to(rows[0][5], UP, buff=0.28),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT*0.2) for r in rows], lag_ratio=0.2),
                  FadeIn(hdr), run_time=1.8)
        self.wait(1.8)
        self._rows5 = VGroup(rows, hdr)

    # ============================================================
    def act6_bet(self, cap):
        cap("One score set → consistent probabilities for every bet type → bet only when EV > 0", C_SCORE)
        self.play(self._rows5.animate.scale(0.82).to_edge(LEFT, buff=0.3), run_time=1.0)
        probs = [0.46, 0.14, 0.28, 0.12]
        bars = VGroup()
        for i, p in enumerate(probs):
            bar = Rectangle(width=0.5, height=0.2 + 2.6*p, stroke_width=0,
                            fill_color=C_SCORE, fill_opacity=0.85)
            lab = jt(f"H{i+1}", 15).next_to(bar, DOWN, buff=0.1)
            bars.add(VGroup(bar, lab))
        bars.arrange(RIGHT, buff=0.4, aligned_edge=DOWN).move_to([2.4, 0.55, 0])
        winlbl = MathTex(r"p^{\text{win}}_i=\mathrm{softmax}(s_i/T)", color=WHITE).scale(0.62).next_to(bars, UP, buff=0.4)
        self.play(FadeIn(winlbl), LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars], lag_ratio=0.12),
                  *[FadeIn(b[1]) for b in bars], run_time=1.6)
        pl = jt("place / exotic bets = Plackett-Luce (analytic / MC from scores)", 18, color=C_DIM).next_to(bars, DOWN, buff=0.6)
        self.play(FadeIn(pl))
        self.wait(1.4)

        specs = [("EV = p × odds", C_SCORE), ("fractional Kelly", C_ODDS), ("BUY (recommended)", C_AGG)]
        chain = VGroup()
        for txt, col in specs:
            t = jt(txt, 18, color=col)
            box = SurroundingRectangle(t, color=col, buff=0.16, corner_radius=0.1).set_stroke(opacity=0.8)
            chain.add(VGroup(t, box))
        chain.arrange(RIGHT, buff=0.7).to_edge(DOWN, buff=1.25).to_edge(RIGHT, buff=0.5)
        arrs = VGroup(*[MathTex(r"\rightarrow", color=WHITE).scale(0.7).move_to(
            (chain[i].get_right() + chain[i+1].get_left())/2) for i in range(2)])
        self.play(FadeIn(chain[0]))
        self.play(FadeIn(arrs[0]), FadeIn(chain[1]))
        self.play(FadeIn(arrs[1]), FadeIn(chain[2]))
        self.play(Indicate(chain[2], color=C_AGG, scale_factor=1.12))
        self.wait(1.8)
        self.play(FadeOut(self._rows5), FadeOut(bars), FadeOut(winlbl), FadeOut(pl),
                  FadeOut(chain), FadeOut(arrs), run_time=1.0)

    # ============================================================
    def act7_screens(self, cap):
        cap("The actual web app — from model predictions to bet suggestions and P&L")

        def shot(name):
            return ImageMobject(os.path.join(IMG_DIR, name))

        detail = shot("race-detail.png")
        detail.height = 4.6
        detail.move_to([0, 0.4, 0])
        frame = SurroundingRectangle(detail, color=C_DIM, buff=0.0, stroke_width=1.5)
        dlab = jt("Race detail — per-horse AI predictions, win EV, and BUY tags", 19, color=WHITE)
        dlab.next_to(detail, DOWN, buff=0.32)
        self.play(FadeIn(detail, scale=1.03), Create(frame), run_time=1.2)
        self.play(FadeIn(dlab))
        self.wait(2.4)
        self.play(FadeOut(detail), FadeOut(frame), FadeOut(dlab), run_time=0.9)

        names = [("race-list.png", "Past races"),
                 ("ledger.png", "Ledger (P&L by bet type)"),
                 ("models.png", "Models (train / switch / backtest)"),
                 ("dashboard.png", "Dashboard (performance overview)")]
        tiles = Group()
        for fn, _ in names:
            img = shot(fn)
            img.height = 2.5
            tiles.add(img)
        tiles.arrange_in_grid(rows=2, cols=2, buff=(0.9, 0.7)).move_to([0, 0.4, 0])
        labels = VGroup()
        borders = VGroup()
        for img, (_, lb) in zip(tiles, names):
            borders.add(SurroundingRectangle(img, color=C_DIM, buff=0.0, stroke_width=1.2))
            labels.add(jt(lb, 16, color=C_DIM).next_to(img, DOWN, buff=0.14))
        self.play(LaggedStart(*[AnimationGroup(FadeIn(img, scale=1.03), Create(bd), FadeIn(lb))
                                for img, bd, lb in zip(tiles, borders, labels)], lag_ratio=0.25), run_time=2.2)
        self.wait(2.6)
        self.play(FadeOut(tiles), FadeOut(borders), FadeOut(labels), run_time=0.9)
