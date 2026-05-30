"""Race-level odds dictionaries used by predict / simulation / evaluate.

Entry point is :func:`compute_race_odds_with_sources` which merges three
sources (live odds → past payouts → tansho-implied) into a single
``{bet_type: {combo: odds}}`` dict alongside per-combo source markers.
"""

from __future__ import annotations

import json
from itertools import combinations, permutations

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ai.calibrate import (
    compute_all_combination_probs,
    compute_analytical_combo_probs,
)
from db.models.entry import Entry
from db.models.payout import Payout
from db.models.race import Race


def _normalize_combo(combo: str) -> str:
    """payouts テーブルの combo 文字列を predict 側と同じ表記に正規化する。

    netkeiba HTML から来た combo は ``10 - 14`` / ``14 → 10`` のように
    区切り文字の前後に **半角/全角空白** が入っていることがある一方、
    predict_race_with_combinations_gbdt は空白なし (``10-14`` / ``14→10``) を生成する。
    そのままだと _settle_candidates の dict ルックアップで全部 miss して
    連系 (馬連/ワイド/馬単/三連複/三連単) の payback_rate が常に 0% になる。

    全種類に共通で「すべての whitespace を除去」する正規化を適用すれば足りる。
    """
    # str.translate より小規模 string では join + split が読みやすい
    return "".join(combo.split())


def compute_past_race_odds(
    session: Session,
    race_id: str,
) -> dict[str, dict[str, float]]:
    """過去レースの確定オッズを返す（取れた combo のみ）。

    - 単勝: entries.odds_win から全馬の確定オッズ（締切時オッズ）
    - 複勝: races.payout_place JSON の 1〜3 着馬のみ payout/100 = 確定オッズ
    - 連系（馬連/ワイド/馬単/三連複/三連単）: payouts テーブルの amount/100 のみ（的中 combo のみ）

    取得不能な combo（複勝の 4 着以下、連系の外れ）は dict に含めない。
    呼び出し側は「ない combo は est_odds=None」で扱うこと。

    Args:
        session: SQLAlchemy Session.
        race_id: 対象レースの race_id。

    Returns:
        {bet_type: {combo: odds}} 形式の 2 段ネスト dict。
        compute_race_odds_with_sources と同じ構造。
    """
    result: dict[str, dict[str, float]] = {}

    # ── 単勝: entries.odds_win から全馬 ─────────────────────────────────────
    entry_rows = session.execute(
        select(Entry.post_position, Entry.odds_win)
        .where(Entry.race_id == race_id)
        .where(Entry.odds_win.is_not(None))
        .where(Entry.post_position.is_not(None))
    ).all()

    if entry_rows:
        result["単勝"] = {
            str(row.post_position): row.odds_win
            for row in entry_rows
        }

    # ── 複勝: races.payout_place JSON の 1〜3 着馬 ─────────────────────────
    race_row = session.execute(
        select(Race.payout_place)
        .where(Race.race_id == race_id)
    ).first()

    if race_row is not None and race_row.payout_place is not None:
        try:
            # payout_place は {"1": 110, "2": 160, "3": 150} 形式の JSON 文字列
            payout_place_map: dict[str, int] = json.loads(race_row.payout_place)
        except (json.JSONDecodeError, TypeError):
            payout_place_map = {}

        if payout_place_map:
            # finish_position ごとの post_position を引く
            # payout_place_map のキーは finish_position（着順）
            finish_positions = [int(k) for k in payout_place_map if k.isdigit()]
            if finish_positions:
                place_entry_rows = session.execute(
                    select(Entry.post_position, Entry.finish_position)
                    .where(Entry.race_id == race_id)
                    .where(Entry.finish_position.in_(finish_positions))
                    .where(Entry.post_position.is_not(None))
                ).all()

                finish_to_post = {
                    row.finish_position: row.post_position
                    for row in place_entry_rows
                }

                fuku_odds: dict[str, float] = {}
                for finish_pos_str, amount in payout_place_map.items():
                    if not finish_pos_str.isdigit():
                        continue
                    finish_pos = int(finish_pos_str)
                    post_pos = finish_to_post.get(finish_pos)
                    if post_pos is not None and amount is not None:
                        fuku_odds[str(post_pos)] = amount / 100.0

                if fuku_odds:
                    result["複勝"] = fuku_odds

    # ── 連系: payouts テーブルから的中 combo のみ ─────────────────────────
    payout_rows = session.execute(
        select(Payout.bet_type, Payout.combo, Payout.amount)
        .where(Payout.race_id == race_id)
        .where(Payout.amount.is_not(None))
    ).all()

    for row in payout_rows:
        if row.bet_type in ("単勝", "複勝"):
            # 単勝・複勝は上記で別途処理済み（payouts の単勝/複勝も入れると重複するが
            # payouts テーブルの方が着順 1 位の確定値なので上書きでも OK）
            continue
        if row.bet_type not in result:
            result[row.bet_type] = {}
        # 連系の combo は payouts テーブルだと "10 - 14" のように空白が入って
        # いるが、predict 側は "10-14" を生成するので空白を除去して合わせる。
        result[row.bet_type][_normalize_combo(row.combo)] = row.amount / 100.0

    return result


# ---------------------------------------------------------------------------
# Tansho-implied combination odds (Plackett-Luce)
#
# 単勝オッズ → 暗黙勝率 → Plackett-Luce score → 各券種の確率 → 控除率込みオッズ。
# 過去レースで連系オッズが取得できない（ハズレ combo は payouts に記録されない）
# 時、全体平均より精度の高い「市場ベース推定オッズ」として使う。
# ---------------------------------------------------------------------------

# JRA 公式の券種別控除率（払戻率は 1 - takeout）
_JRA_TAKEOUT_RATES: dict[str, float] = {
    "単勝": 0.20,
    "複勝": 0.20,
    "枠連": 0.225,
    "馬連": 0.225,
    "ワイド": 0.225,
    "馬単": 0.25,
    "三連複": 0.25,
    "三連単": 0.275,
}


def tansho_to_pl_scores(
    odds_win_by_post: dict[str, float],
) -> tuple[list[str], np.ndarray]:
    """単勝オッズから Plackett-Luce score を導出する。

    手順:
      1. raw_p_i = 1 / odds_i  （ブックメーカー暗黙勝率, overround 含み）
      2. normalized_p_i = raw_p_i / Σ raw_p_j  （overround 補正で合計=1 に正規化）
      3. score_i = log(normalized_p_i)  （PL は softmax(score) なので log で逆算）

    Args:
        odds_win_by_post: {post_position(str): 単勝オッズ(float)}.
            odds が None や 0 以下のエントリは除外される。

    Returns:
        (post_positions, scores):
          - post_positions: post_position 昇順のリスト（数値順）。
          - scores: 同順の np.ndarray (n,)。

    Raises:
        ValueError: 有効なエントリが 0 件のとき。
    """
    valid_items: list[tuple[str, float]] = []
    for post, odds in odds_win_by_post.items():
        if odds is None or odds <= 0:
            continue
        valid_items.append((post, float(odds)))

    if not valid_items:
        raise ValueError("有効な単勝オッズが 0 件です")

    # post_position は string ですが数値順にソート（"10" > "2" にならないように int 経由）。
    valid_items.sort(key=lambda x: int(x[0]))
    post_positions = [p for p, _ in valid_items]

    raw_p = np.array([1.0 / o for _, o in valid_items])
    normalized_p = raw_p / raw_p.sum()
    scores = np.log(normalized_p)
    return post_positions, scores


def compute_implied_combo_odds_from_tansho(
    odds_win_by_post: dict[str, float],
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
    use_analytical: bool = True,
) -> dict[str, dict[str, float]]:
    """単勝オッズから連系券種のオッズを Plackett-Luce で推定する。

    確定オッズが取得できない過去レースの未確定 combo を埋めるための
    「市場ベース推定オッズ」を返す。控除率込みの実オッズ近似値:

      fair_odds  = 1 / P(combo)
      est_odds   = max(1.0, fair_odds × (1 - takeout))   # JRA 最低払戻 100 円

    Returns:
        {bet_type: {combo_str: odds, ...}, ...} の 2 段ネスト dict。
        bet_type と combo 形式は compute_past_race_odds と一致:
          - 複勝:         "3"     (post_position のみ; top-3 入着確率の逆数)
          - 馬連 / ワイド: "3-7"  (post_position 昇順)
          - 馬単:         "3→7"  (1着→2着)
          - 三連複:       "3-5-7" (post_position 昇順)
          - 三連単:       "3→5→7" (1着→2着→3着)

    Args:
        odds_win_by_post: 全馬の単勝オッズ。
        n_samples: MC サンプル数 (use_analytical=False のときのみ意味を持つ)。
        rng: numpy Generator (use_analytical=False のときのみ)。
        use_analytical: True (デフォルト) で閉じた解析式を使用。MC サンプル
            数有限による量子化（10K samples だと低確率 combo が 1/10000,
            2/10000... の離散値しか取れない）を完全に回避する。
            False で従来の MC 経路。

    Raises:
        ValueError: 有効なエントリが 2 件未満のとき（連系券種が組めない）。
    """
    posts, scores = tansho_to_pl_scores(odds_win_by_post)
    n = len(posts)
    if n < 2:
        raise ValueError("連系オッズ推定には 2 頭以上の単勝オッズが必要です")

    if use_analytical:
        probs = compute_analytical_combo_probs(scores)
    else:
        probs = compute_all_combination_probs(
            scores, k=3, n_samples=n_samples, rng=rng
        )

    def _to_odds(prob: float, takeout: float) -> float | None:
        """fair odds × (1-takeout) を返し、JRA 最低払戻 1.0 倍で floor する。"""
        if prob <= 0:
            return None
        est = (1.0 / prob) * (1.0 - takeout)
        return max(1.0, est)

    result: dict[str, dict[str, float]] = {}

    # ── 複勝: place ベクトル (n,), P(horse i in top-3) ─────────────────────
    place_vec = probs.get("place")
    if place_vec is not None:
        fukusho: dict[str, float] = {}
        takeout = _JRA_TAKEOUT_RATES["複勝"]
        for i in range(n):
            prob = float(place_vec[i])
            est = _to_odds(prob, takeout)
            if est is None:
                continue
            fukusho[str(int(posts[i]))] = est
        if fukusho:
            result["複勝"] = fukusho

    # ── 馬連: pair_matrix (n, n) 対称, P({i, j}) ────────────────────────────
    pair_matrix = probs.get("pair")
    if pair_matrix is not None:
        umaren: dict[str, float] = {}
        takeout = _JRA_TAKEOUT_RATES["馬連"]
        for i, j in combinations(range(n), 2):
            prob = float(pair_matrix[i, j])
            est = _to_odds(prob, takeout)
            if est is None:
                continue
            pp_i, pp_j = int(posts[i]), int(posts[j])
            pp_lo, pp_hi = (pp_i, pp_j) if pp_i <= pp_j else (pp_j, pp_i)
            umaren[f"{pp_lo}-{pp_hi}"] = est
        if umaren:
            result["馬連"] = umaren

        # ── ワイド: P(both in top-3, unordered) ─────────────────────────────
        # PL モンテカルロから直接 ワイド 確率を計算: 「両馬 i,j が top-3 に含まれる」
        # samples を再利用したいが compute_all_combination_probs は samples を返さない。
        # 代わりに place 確率の組み合わせから近似は取れないので、ここでは
        # 三連複 (triple) の周辺化で正確に算出する。
        triple_dict: dict[frozenset, float] = probs.get("triple", {})
        if triple_dict:
            wide: dict[str, float] = {}
            takeout_wide = _JRA_TAKEOUT_RATES["ワイド"]
            for i, j in combinations(range(n), 2):
                # P({i,j} ⊂ top-3) = Σ_{k ∉ {i,j}} P({i,j,k} 三連複)
                prob = sum(
                    triple_dict.get(frozenset({i, j, k}), 0.0)
                    for k in range(n) if k != i and k != j
                )
                est = _to_odds(prob, takeout_wide)
                if est is None:
                    continue
                pp_i, pp_j = int(posts[i]), int(posts[j])
                pp_lo, pp_hi = (pp_i, pp_j) if pp_i <= pp_j else (pp_j, pp_i)
                wide[f"{pp_lo}-{pp_hi}"] = est
            if wide:
                result["ワイド"] = wide

    # ── 馬単: ordered_pair_matrix (n, n), P(i 1着 ∩ j 2着) ───────────────
    ordered_pair = probs.get("ordered_pair")
    if ordered_pair is not None:
        umatan: dict[str, float] = {}
        takeout = _JRA_TAKEOUT_RATES["馬単"]
        for i, j in permutations(range(n), 2):
            prob = float(ordered_pair[i, j])
            est = _to_odds(prob, takeout)
            if est is None:
                continue
            umatan[f"{int(posts[i])}→{int(posts[j])}"] = est
        if umatan:
            result["馬単"] = umatan

    # ── 三連複: triple_dict, P({i,j,k}) unordered ─────────────────────────
    triple_dict = probs.get("triple", {})
    if triple_dict:
        sanrenpuku: dict[str, float] = {}
        takeout = _JRA_TAKEOUT_RATES["三連複"]
        for i, j, k in combinations(range(n), 3):
            prob = float(triple_dict.get(frozenset({i, j, k}), 0.0))
            est = _to_odds(prob, takeout)
            if est is None:
                continue
            pps = sorted([int(posts[i]), int(posts[j]), int(posts[k])])
            sanrenpuku[f"{pps[0]}-{pps[1]}-{pps[2]}"] = est
        if sanrenpuku:
            result["三連複"] = sanrenpuku

    # ── 三連単: ordered_triple (n, n, n), P(i→j→k) ─────────────────────
    ordered_triple = probs.get("ordered_triple")
    if ordered_triple is not None:
        sanrentan: dict[str, float] = {}
        takeout = _JRA_TAKEOUT_RATES["三連単"]
        for i, j, k in permutations(range(n), 3):
            prob = float(ordered_triple[i, j, k])
            est = _to_odds(prob, takeout)
            if est is None:
                continue
            sanrentan[f"{int(posts[i])}→{int(posts[j])}→{int(posts[k])}"] = est
        if sanrentan:
            result["三連単"] = sanrentan

    return result


def compute_race_odds_with_sources(
    session: Session,
    race_id: str,
    n_samples: int = 10_000,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, str]]]:
    """確定オッズ + 単勝由来の推定オッズ統合版。レース時期を問わず使える。

    優先順位:
      1. payouts / entries.odds_win / payout_place（過去）   → "confirmed"
      2. 残りの combo は単勝由来 Plackett-Luce 推定で補完     → "implied"

    両方の odds と source は同一構造の 2 段ネスト dict で返す:

    Returns:
        odds:    {bet_type: {combo: odds}}
        sources: {bet_type: {combo: "confirmed" | "implied"}}

    odds が空（単勝も取れない）→ ({}, {}) を返す。
    """
    # Step 1: 過去レースの payout / entries.odds_win 系の確定オッズ
    confirmed = compute_past_race_odds(session, race_id)

    # confirmed 由来の combo に "confirmed" マーカーを付与
    sources: dict[str, dict[str, str]] = {
        bt: {c: "confirmed" for c in combos}
        for bt, combos in confirmed.items()
    }

    # Step 2: 単勝オッズから連系券種の推定オッズを生成
    tansho_odds = confirmed.get("単勝", {})
    if not tansho_odds:
        # confirmed に "単勝" が無い場合（live で連系のみ取得済み等）、
        # entries.odds_win から直接拾う
        entry_rows = session.execute(
            select(Entry.post_position, Entry.odds_win)
            .where(Entry.race_id == race_id)
            .where(Entry.odds_win.is_not(None))
            .where(Entry.post_position.is_not(None))
        ).all()
        tansho_odds = {
            str(row.post_position): row.odds_win for row in entry_rows
        }

    if len(tansho_odds) < 2:
        return confirmed, sources

    try:
        implied = compute_implied_combo_odds_from_tansho(
            tansho_odds, n_samples=n_samples
        )
    except ValueError:
        return confirmed, sources

    # 未確定 combo のみ補完しつつ source も同時記録
    for bet_type, combos in implied.items():
        if bet_type not in confirmed:
            confirmed[bet_type] = {}
            sources[bet_type] = {}
        for combo, odds in combos.items():
            if combo not in confirmed[bet_type]:
                confirmed[bet_type][combo] = odds
                sources[bet_type][combo] = "implied"

    return confirmed, sources
