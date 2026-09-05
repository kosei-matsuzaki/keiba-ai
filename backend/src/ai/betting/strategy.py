"""Bet pattern generation and flat stake allocation.

This module is pure-function: no database access, no I/O.  All functions
operate on CombinationPrediction lists produced by predict_race_with_combinations
and return BetCandidate lists.

Supported buy patterns:
  nagashi    — one axis horse vs. all others (single-axis wheel)
  box        — top-N horses in all combinations
  formation  — first/second/third legs specified independently

賭け金の決め方 (定額):
  `assign_flat_stakes` の docstring を見る。**EV 順には並べない** — その理由
  (較正後は単勝の EV が連系の後ろに回り、予算が足りないと単複から切り捨てられる)
  もそこにある。

  以前は資金比率の fractional Kelly で決めていたが、利用者が扱うのは
  「このレースに何円使うか」であって「資金の何%か」ではないため、
  賭け金の決定からは Kelly を外した (学習の目的関数側は別。ai/model/loss.py)。

Notes on three-leg ordered bets (三連単):
  nagashi and formation for 三連単 are deferred to a future issue.  The
  combinatorial explosion (up to 18 * 17 * 16 = 4896 permutations) and the
  asymmetric axis semantics require dedicated UX; box is sufficient for this
  sprint.
"""

from __future__ import annotations

import pandas as pd

from ai.core.types import BetCandidate, CombinationPrediction, RecommendationResult
from core.bet_types import DEFAULT_COMBO_MIN_HIT_PROB

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Bet types for which nagashi / formation produce meaningful output
_NAGASHI_SUPPORTED = frozenset(["馬連", "ワイド", "馬単", "三連複"])
_FORMATION_SUPPORTED = frozenset(["馬単", "三連単", "三連複"])


def _make_candidate(
    cp: CombinationPrediction,
    bet_type: str,
    pattern: str,
) -> BetCandidate:
    return BetCandidate(
        bet_type=bet_type,
        combo=cp.combo,
        pattern=pattern,  # type: ignore[arg-type]
        prob=cp.prob,
        est_odds=cp.est_odds,
        est_odds_source=cp.est_odds_source,
        ev=cp.ev,
        stake=0,
        post_positions=cp.post_positions,
    )


# ---------------------------------------------------------------------------
# Pattern generators (pure functions, stake=0 on output)
# ---------------------------------------------------------------------------

def generate_nagashi(
    combinations_list: list[CombinationPrediction],
    axis_post_position: int,
    bet_type: str,
    axis_position: int = 1,
) -> list[BetCandidate]:
    """Generate nagashi (axis-wheel) candidates.

    For bet types where nagashi is undefined (e.g. 単勝, 複勝, 三連単), returns
    an empty list.

    Args:
        combinations_list: CombinationPrediction list for the target bet_type.
        axis_post_position: Post position of the axis horse.
        bet_type: 馬券種 string.
        axis_position: For 馬単, which finishing position the axis occupies
            (1 = axis finishes 1st, 2 = axis finishes 2nd).  Ignored for
            bet types other than 馬単.

    Returns:
        list[BetCandidate] with stake=0, pattern='nagashi'.
    """
    if bet_type not in _NAGASHI_SUPPORTED:
        return []

    candidates: list[BetCandidate] = []

    if bet_type in ("馬連", "ワイド"):
        for cp in combinations_list:
            if axis_post_position in cp.post_positions:
                candidates.append(_make_candidate(cp, bet_type, "nagashi"))

    elif bet_type == "馬単":
        for cp in combinations_list:
            if axis_position == 1 and cp.post_positions[0] == axis_post_position or axis_position == 2 and cp.post_positions[1] == axis_post_position:
                candidates.append(_make_candidate(cp, bet_type, "nagashi"))

    elif bet_type == "三連複":
        for cp in combinations_list:
            if axis_post_position in cp.post_positions:
                candidates.append(_make_candidate(cp, bet_type, "nagashi"))

    return candidates


def generate_box(
    combinations_list: list[CombinationPrediction],
    horse_post_positions: list[int],
    bet_type: str,
) -> list[BetCandidate]:
    """Generate box candidates for the given set of horses.

    Counts per bet type:
      馬連:  C(n, 2)  — combinations of 2
      ワイド: C(n, 2)
      馬単:  P(n, 2)  — permutations of 2
      三連複: C(n, 3)
      三連単: P(n, 3)
      単勝 / 複勝: single-horse bets; returns all listed horses

    Args:
        combinations_list: CombinationPrediction list for the target bet_type.
        horse_post_positions: Post positions of horses to include in the box.
        bet_type: 馬券種 string.

    Returns:
        list[BetCandidate] with stake=0, pattern='box'.
    """
    pps = horse_post_positions
    candidates: list[BetCandidate] = []

    if bet_type in ("単勝", "複勝"):
        for cp in combinations_list:
            if cp.post_positions[0] in pps:
                candidates.append(_make_candidate(cp, bet_type, "box"))

    elif bet_type in ("馬連", "ワイド"):
        pp_set = set(pps)
        for cp in combinations_list:
            if pp_set.issuperset(cp.post_positions):
                candidates.append(_make_candidate(cp, bet_type, "box"))

    elif bet_type == "馬単":
        pp_set = set(pps)
        for cp in combinations_list:
            # 馬単: ordered pair, both must be in the box set
            if pp_set.issuperset(cp.post_positions) and cp.post_positions[0] != cp.post_positions[1]:
                candidates.append(_make_candidate(cp, bet_type, "box"))

    elif bet_type == "三連複":
        pp_set = set(pps)
        for cp in combinations_list:
            if pp_set.issuperset(cp.post_positions):
                candidates.append(_make_candidate(cp, bet_type, "box"))

    elif bet_type == "三連単":
        pp_set = set(pps)
        for cp in combinations_list:
            # 三連単: all three positions must be distinct and in the box set
            if (
                pp_set.issuperset(cp.post_positions)
                and len(set(cp.post_positions)) == 3
            ):
                candidates.append(_make_candidate(cp, bet_type, "box"))

    return candidates


def generate_formation(
    combinations_list: list[CombinationPrediction],
    first_post_positions: list[int],
    second_post_positions: list[int],
    third_post_positions: list[int] | None,
    bet_type: str,
) -> list[BetCandidate]:
    """Generate formation candidates.

    For 馬連 / ワイド formation is semantically identical to box (order has no
    meaning), so this function returns [] for those types — callers should use
    generate_box instead.

    Args:
        combinations_list: CombinationPrediction list for the target bet_type.
        first_post_positions:  Horses allowed in 1st finishing position.
        second_post_positions: Horses allowed in 2nd finishing position.
        third_post_positions:  Horses allowed in 3rd position (used only for
            三連単; ignored / can be None for 馬単).
        bet_type: 馬券種 string.

    Returns:
        list[BetCandidate] with stake=0, pattern='formation'.
        Returns [] for unsupported bet types (馬連, ワイド, 単勝, 複勝).
    """
    if bet_type not in _FORMATION_SUPPORTED:
        return []

    candidates: list[BetCandidate] = []
    first_set = set(first_post_positions)
    second_set = set(second_post_positions)

    if bet_type == "馬単":
        for cp in combinations_list:
            a, b = cp.post_positions
            if a in first_set and b in second_set and a != b:
                candidates.append(_make_candidate(cp, bet_type, "formation"))

    elif bet_type == "三連単":
        if not third_post_positions:
            return []
        third_set = set(third_post_positions)
        for cp in combinations_list:
            a, b, c = cp.post_positions
            if (
                a in first_set
                and b in second_set
                and c in third_set
                and len({a, b, c}) == 3
            ):
                candidates.append(_make_candidate(cp, bet_type, "formation"))

    elif bet_type == "三連複":
        # Formation for 三連複 is treated as a box over the union of all legs.
        # Ordering has no meaning in 三連複.
        all_pps = list(first_set | second_set | (set(third_post_positions) if third_post_positions else set()))
        return generate_box(combinations_list, all_pps, bet_type)

    return candidates


# ---------------------------------------------------------------------------
# Flat stake assignment
# ---------------------------------------------------------------------------

# 買う順番。単勝・複勝は回収率の推定が安定していて市場を上回ることが確認できて
# いる (5,376 点で 0.931 / 5,402 点で 0.887) 一方、連系は同じ窓で 1,587 点しか
# 出ず信頼区間が 0.01〜2.6 と測定不能。予算が足りないときに切るべきは連系なので、
# 単複を先に確保する。
_BET_TYPE_PRIORITY: dict[str, int] = {"単勝": 0, "複勝": 1}
_DEFAULT_PRIORITY = 9

#: 連系の買い目を組む上位何頭か。**選択肢にはしない。**
#:
#: 以前は「狙い方 (本命中心 / 標準 / 穴も拾う)」として 3 / 5 / 8 を選ばせていたが、
#: 買うかどうかは的中確率の下限 (`combo_min_hit_prob`) が決めるので、頭数を広げても
#: 線を超えない買い目が候補に増えるだけで**買い目は変わらない**。選べるのに何も
#: 起きない選択肢を残さない。
TOP_N_HORSES = 3

#: 1 レースに使う上限 (円) の既定値。**使い切る目標ではない。**
DEFAULT_RACE_BUDGET = 5_000

#: 1 点あたりの賭け金 (円)。**設定ではない。**
#:
#: 券種ごとに 1 点の額を変える仕組み (旧 `stake_units`) は廃止した。厚みは
#: 「1 点いくらか」ではなく **何点買うか** で表す方が、買い方としても素直で、
#: 確信度との対応も 1 つの式で書ける (`ai.inference.confidence.points_for_confidence`)。
STAKE_UNIT = 100


def assign_flat_stakes(
    candidates: list[BetCandidate],
    race_budget: int,
    points_by_bet_type: dict[str, int] | None = None,
    keep_zero_stake: bool = False,
    min_hit_prob_by_bet_type: dict[str, float] | None = None,
) -> list[BetCandidate]:
    """買い目に賭け金を割り当てる。**1 点 = 100 円**。

      1. 券種ごとの買い条件を満たす買い目だけを対象にする
      2. **単勝 → 複勝 → 連系**の順に並べ、同じ券種内では的中確率の高い順
      3. 券種ごとの点数 × 100 円を、``race_budget`` を超えない範囲で賭ける

    **予算は使い切らなくてよい。** 対象が少なければ賭け金の合計も少なくなる。

    厚みは **点数** で表す。単勝・複勝は 1 頭に対する点数を確信度から決め
    (`points_for_confidence`)、連系は 1 組合せ = 1 点で、**何点買うかは的中確率の
    下限を超えた買い目の数**が決める。券種ごとの点数上限は持たない — ワイドが効く
    レース・三連単が効くレースを一律の点数で潰さないため。

    **EV の高い順には並べない。** 確率を正直に較正すると単勝の EV は 0.6 前後に
    なり、連系 (EV 5〜9) の後ろに回る。その状態で予算が足りないと、回収率の推定が
    最も確かな単複が真っ先に切り捨てられ、測定不能な連系だけが残る (実測: 2,034
    レースで単勝 3 点・複勝 1 点)。EV は券種をまたいで比較できる指標ではない。

    Args:
        candidates: BetCandidate list (stake field is ignored on input).
        race_budget: このレースに使ってよい上限 (円)。
        points_by_bet_type: 券種ごとの点数 (1 買い目あたり)。未指定の券種は 1 点。
            単複はここに確信度から出した点数が入る。
        keep_zero_stake: True なら賭けない買い目も stake=0 で返す。
        min_hit_prob_by_bet_type: 券種ごとの **的中確率の下限**。この線を下回る
            買い目は買わない。**連系の点数はこれだけで決まる** (確信度の高い
            レースほど深く買い、低いレースでは 0 点)。
            既定は `DEFAULT_COMBO_MIN_HIT_PROB` (連系のみ・単複には掛けない)。

    Returns:
        New list of BetCandidate (copies) with updated stake values。
        keep_zero_stake=False なら stake>0 のものだけ。
    """
    points = points_by_bet_type or {}
    floors = (
        DEFAULT_COMBO_MIN_HIT_PROB
        if min_hit_prob_by_bet_type is None
        else min_hit_prob_by_bet_type
    )

    def _stake(c: BetCandidate) -> int:
        return max(1, int(points.get(c.bet_type, 1))) * STAKE_UNIT

    if race_budget < STAKE_UNIT:
        # 1 点分にも満たないなら何も買えない
        return [c.model_copy(update={"stake": 0}) for c in candidates] if keep_zero_stake else []

    def _passes(c: BetCandidate) -> bool:
        # **EV 閾値は使わない。** est_odds が取れない買い目だけを落とす (値段が
        # 分からないものは買えないため)。
        return c.est_odds is not None

    def _sort_key(c: BetCandidate) -> tuple[int, float]:
        # 単勝 → 複勝 → 連系。同じ券種内は的中確率の高い順 (EV 順にはしない)
        return (_BET_TYPE_PRIORITY.get(c.bet_type, _DEFAULT_PRIORITY), -c.prob)

    eligible = sorted((c for c in candidates if _passes(c)), key=_sort_key)
    ineligible = [c for c in candidates if not _passes(c)]

    out: list[BetCandidate] = []
    budget_left = race_budget
    for c in eligible:
        # **確信度が足りない買い目は買わない。** ここが「点数がレースごとに
        # 変わる」仕組み: 線を超えた買い目の数だけ買う。
        floor = floors.get(c.bet_type)
        if floor is not None and c.prob < floor:
            if keep_zero_stake:
                out.append(c.model_copy(update={"stake": 0}))
            continue
        stake = _stake(c)
        if budget_left >= stake:
            out.append(c.model_copy(update={"stake": stake}))
            budget_left -= stake
        elif keep_zero_stake:
            out.append(c.model_copy(update={"stake": 0}))

    if keep_zero_stake:
        out.extend(c.model_copy(update={"stake": 0}) for c in ineligible)
    return out


# ---------------------------------------------------------------------------
# Recommendation orchestration
# ---------------------------------------------------------------------------

def recommend_for_race(
    predictions: pd.DataFrame,
    combinations_by_type: dict[str, list[CombinationPrediction]],
    race_id: str,
    race_budget: int,
    points_by_bet_type: dict[str, int] | None = None,
    win_min_odds: float = 1.1,
    top_n_horses: int = 3,
    enabled_bet_types: list[str] | None = None,
    min_hit_prob_by_bet_type: dict[str, float] | None = None,
) -> RecommendationResult:
    """Generate a bet recommendation for one race.

    For each enabled bet type the function tries three patterns (nagashi, box,
    formation) and picks the one with the highest combined score
    (total_stake * mean_ev).  The winning pattern's candidates are collected
    across all bet types, then assign_stakes is called once on the full set.

    Args:
        predictions: DataFrame with columns [horse_id, score, win_prob,
            post_position].  Horses are sorted by score descending (output of
            predict_race).  The 'post_position' column must be present; it is
            not always included in raw predict_race output, so callers are
            expected to join it from the feature frame.
        combinations_by_type: Dict mapping bet_type to list[CombinationPrediction]
            (output of predict_race_with_combinations).
        race_id: Identifier for the race.
        race_budget: このレースに使ってよい上限 (円)。使い切らなくてよい。
        points_by_bet_type: 券種ごとの点数 (1 買い目あたり)。1 点 = 100 円。
            単複は確信度から決めた点数を入れる。未指定の券種は 1 点。
        win_min_odds: **単勝・複勝は EV 条件を使わず「モデルの本命 (1 位) を買う」**。
            単勝はこのオッズ下限を下回る場合だけ見送る (複勝に下限は無い)。
            理由は較正済み確率での実測 (test 19ヶ月・5,404 レース):

            | 券種 | EV 条件 | 本命買い |
            |---|---|---|
            | 単勝 | 45,001 点 / 0.698 | 5,376 点 / **0.931** |
            | 複勝 | 43,464 点 / 0.654 | 5,402 点 / **0.887** |

            複勝の EV は狂っているだけでなく **順序が逆**で、EV 帯別の回収率は
            0.0-0.9 帯 0.832 → 2.0 以上 0.573 と単調減少する。高 EV = 推定オッズの高い
            穴馬 = 確率もオッズも最大に過大評価される帯、という構造なので、温度や冪
            (Harville 補正) のような単調変換では直らない (実測: 冪 0.85 で 0.665 止まり)。
            **1 位買いでも回収率 1.0 未満**なので「+EV だから買う」ではなく
            「本命はこれ」という意味の推奨である点に注意 (docs/ai-model.md)。
        top_n_horses: Number of top horses (by win_prob) to include in box /
            formation candidates (default 3).
        enabled_bet_types: Bet types to consider.  None means all types
            present in combinations_by_type.

    Returns:
        RecommendationResult。stake=0 の候補も含めて返す (UI 側で「基準を
        超えなかった買い目」も見せられるようにするため)。
    """
    if enabled_bet_types is not None:
        active_types = [bt for bt in enabled_bet_types if bt in combinations_by_type]
    else:
        active_types = list(combinations_by_type.keys())

    # 単勝・複勝は EV ではなく「モデルの本命 (1 位)」で選ぶ。候補自体を 1 点に絞り、
    # EV 閾値は素通しにする (win_min_odds の docstring に理由と実測値)。
    combinations_by_type = dict(combinations_by_type)
    tansho = combinations_by_type.get("単勝")
    if tansho:
        best = max(tansho, key=lambda c: c.prob)
        keep = best.est_odds is not None and best.est_odds > win_min_odds
        combinations_by_type["単勝"] = [best] if keep else []
    fukusho = combinations_by_type.get("複勝")
    if fukusho:
        combinations_by_type["複勝"] = [max(fukusho, key=lambda c: c.prob)]

    # Rank horses by win_prob descending
    sorted_preds = predictions.sort_values("win_prob", ascending=False).reset_index(drop=True)

    # Derive post_positions for top-N horses
    top_pps: list[int] = []
    for _, row in sorted_preds.iterrows():
        pp = row.get("post_position")
        if pp is not None and not pd.isna(pp):
            top_pps.append(int(pp))
        if len(top_pps) >= top_n_horses:
            break

    axis_pp: int | None = top_pps[0] if top_pps else None

    # Formation legs: 1st = rank-1, 2nd-3rd = top-(top_n_horses+1)
    formation_second: list[int] = top_pps  # includes the axis
    formation_third: list[int] = top_pps

    all_candidates: list[BetCandidate] = []

    for bet_type in active_types:
        combos = combinations_by_type[bet_type]
        if not combos:
            continue

        pattern_candidates: dict[str, list[BetCandidate]] = {}

        # nagashi
        if axis_pp is not None:
            ng = generate_nagashi(combos, axis_pp, bet_type)
            if ng:
                pattern_candidates["nagashi"] = ng

        # box
        bx = generate_box(combos, top_pps, bet_type)
        if bx:
            pattern_candidates["box"] = bx

        # formation
        if axis_pp is not None:
            fm = generate_formation(
                combos,
                first_post_positions=[axis_pp],
                second_post_positions=formation_second,
                third_post_positions=formation_third,
                bet_type=bet_type,
            )
            if fm:
                pattern_candidates["formation"] = fm

        if not pattern_candidates:
            continue

        # Select the pattern with the highest total_stake_proxy * mean_ev.
        #
        # **ここの EV は消し忘れではない。** 買う/買わないの判定から EV を外した
        # のは、較正後の EV が券種をまたいで比較できないため (単勝 0.6 に対し
        # 連系 5〜9)。ここは**同じ券種の中で** nagashi / box / formation を
        # 選ぶだけなので、その比較は成り立つ。買うかどうかは下の
        # `assign_flat_stakes` が的中確率の下限で決める。
        # Since stake is not yet assigned, use prob * est_odds (= ev) as a
        # proxy for quality and len(candidates) as coverage.
        # Candidates with ev=None count as 0 for scoring purposes.
        best_pattern: str | None = None
        best_score = -1.0
        for pat, cands in pattern_candidates.items():
            if not cands:
                continue
            mean_ev = sum(c.ev for c in cands if c.ev is not None) / len(cands)
            score = len(cands) * mean_ev
            if score > best_score:
                best_score = score
                best_pattern = pat

        if best_pattern is not None:
            all_candidates.extend(pattern_candidates[best_pattern])

    # Deduplicate by (bet_type, combo) — keep highest EV (None < any float)。
    # 同じ券種・同じ組合せどうしの比較なので、上と同じ理由で EV を使ってよい。
    seen: dict[tuple[str, str], BetCandidate] = {}
    for c in all_candidates:
        key = (c.bet_type, c.combo)
        if key not in seen:
            seen[key] = c
        else:
            existing_ev = seen[key].ev
            new_ev = c.ev
            # None is treated as lower priority; otherwise keep the higher ev
            if existing_ev is None or (new_ev is not None and new_ev > existing_ev):
                seen[key] = c
    deduped = list(seen.values())

    final_candidates = assign_flat_stakes(
        deduped,
        race_budget=race_budget,
        points_by_bet_type=points_by_bet_type,
        keep_zero_stake=True,
        min_hit_prob_by_bet_type=min_hit_prob_by_bet_type,
    )

    return RecommendationResult(
        race_id=race_id,
        race_budget=race_budget,
        candidates=final_candidates,
    )
