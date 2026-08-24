"""Bet pattern generation and flat stake allocation.

This module is pure-function: no database access, no I/O.  All functions
operate on CombinationPrediction lists produced by predict_race_with_combinations
and return BetCandidate lists.

Supported buy patterns:
  nagashi    — one axis horse vs. all others (single-axis wheel)
  box        — top-N horses in all combinations
  formation  — first/second/third legs specified independently

賭け金の決め方 (定額):
  期待値が基準を超える買い目を ev の高い順に並べ、1 点 stake_unit 円ずつ、
  race_budget を上限に賭ける。予算は使い切らなくてよい。

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

def assign_flat_stakes(
    candidates: list[BetCandidate],
    race_budget: int,
    stake_unit: int = 100,
    min_ev: float = 1.0,
    min_ev_by_bet_type: dict[str, float] | None = None,
    keep_zero_stake: bool = False,
) -> list[BetCandidate]:
    """1 点あたり定額で、期待値の高い買い目から順に予算の範囲で賭ける。

    Kelly（資金の何%を賭けるか）ではなく、人が実際にやる買い方に合わせた配分:

      1. 期待値 (ev) が ``min_ev`` を超える買い目だけを対象にする
      2. ev の高い順に並べる
      3. 1 点あたり ``stake_unit`` 円ずつ、``race_budget`` を超えない範囲で賭ける

    **予算は使い切らなくてよい。** 対象が少なければ賭け金の合計も少なくなる
    （3 点しか基準を超えなければ 300 円で終わる）。基準を超える買い目が
    無ければ 1 円も賭けない。

    Args:
        candidates: BetCandidate list (stake field is ignored on input).
        race_budget: このレースに使ってよい上限 (円)。
        stake_unit: 1 点あたりの賭け金 (円)。100 円単位が実際の購入単位。
        min_ev: この値を超える ev の買い目だけを対象にする (1.0 = 収支トントン)。
        min_ev_by_bet_type: 券種ごとに min_ev を上書きする dict。単勝・複勝は
            EV 条件を使わない (本命買い) ので -inf を入れて素通しにするのに使う。
            無い券種は ``min_ev`` にフォールバック。
        keep_zero_stake: True なら賭けない買い目も stake=0 で返す。

    Returns:
        New list of BetCandidate (copies) with updated stake values,
        ev の高い順。keep_zero_stake=False なら stake>0 のものだけ。
    """
    if stake_unit <= 0 or race_budget < stake_unit:
        # 予算が 1 点分にも満たないなら何も買えない
        return [c.model_copy(update={"stake": 0}) for c in candidates] if keep_zero_stake else []

    by_type = min_ev_by_bet_type or {}

    def _threshold(c: BetCandidate) -> float:
        return by_type.get(c.bet_type, min_ev)

    def _passes(c: BetCandidate) -> bool:
        return c.ev is not None and c.ev > _threshold(c)

    def _sort_key(c: BetCandidate) -> tuple[float, float]:
        # ev 降順 → 同点は prob 降順 (決定的にするため)
        return (-(c.ev or 0.0), -c.prob)

    eligible = sorted((c for c in candidates if _passes(c)), key=_sort_key)
    ineligible = [c for c in candidates if not _passes(c)]

    out: list[BetCandidate] = []
    budget_left = race_budget
    for c in eligible:
        if budget_left >= stake_unit:
            out.append(c.model_copy(update={"stake": stake_unit}))
            budget_left -= stake_unit
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
    stake_unit: int = 100,
    min_ev: float = 1.0,
    min_ev_by_bet_type: dict[str, float] | None = None,
    win_min_odds: float = 1.1,
    top_n_horses: int = 3,
    enabled_bet_types: list[str] | None = None,
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
        stake_unit: 1 点あたりの賭け金 (円、既定 100)。
        min_ev: この値を超える期待値の買い目だけを賭ける (1.0 = 収支トントン)。
        min_ev_by_bet_type: 券種ごとの min_ev 上書き (単複は -inf = 素通し)。
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

    # Deduplicate by (bet_type, combo) — keep highest EV (None < any float)
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
        stake_unit=stake_unit,
        min_ev=min_ev,
        # 単勝は上で 1 位 1 点に絞り済みなので EV 閾値は通す (負の EV でも「本命」として出す)
        min_ev_by_bet_type={
            **(min_ev_by_bet_type or {}),
            "単勝": float("-inf"),
            "複勝": float("-inf"),
        },
        keep_zero_stake=True,
    )

    return RecommendationResult(
        race_id=race_id,
        race_budget=race_budget,
        candidates=final_candidates,
    )
