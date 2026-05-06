"""Bet pattern generation and Kelly fractional stake calculation.

This module is pure-function: no database access, no I/O.  All functions
operate on CombinationPrediction lists produced by predict_race_with_combinations
and return BetCandidate lists.

Supported buy patterns:
  nagashi    — one axis horse vs. all others (single-axis wheel)
  box        — top-N horses in all combinations
  formation  — first/second/third legs specified independently

Kelly fractional stake formula:
  edge = prob * odds - 1
  if edge <= 0: stake = 0
  fraction = kelly_fraction * edge / (odds - 1)
  raw_stake = bankroll * fraction
  stake = floor(raw_stake / round_to) * round_to

Notes on three-leg ordered bets (三連単):
  nagashi and formation for 三連単 are deferred to a future issue.  The
  combinatorial explosion (up to 18 * 17 * 16 = 4896 permutations) and the
  asymmetric axis semantics require dedicated UX; box is sufficient for this
  sprint.
"""

from __future__ import annotations

import math

import pandas as pd

from keiba_ai.ai.types import BetCandidate, CombinationPrediction, RecommendationResult

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_BET_TYPES = frozenset(["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"])

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
            if axis_position == 1 and cp.post_positions[0] == axis_post_position:
                candidates.append(_make_candidate(cp, bet_type, "nagashi"))
            elif axis_position == 2 and cp.post_positions[1] == axis_post_position:
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
# Kelly fractional stake
# ---------------------------------------------------------------------------

def kelly_stake(
    prob: float,
    odds: float,
    bankroll: int,
    kelly_fraction: float,
    round_to: int = 100,
) -> int:
    """Kelly fractional stake rounded down to the nearest round_to multiple.

    Formula:
        edge = prob * odds - 1
        if edge <= 0: return 0
        fraction = kelly_fraction * edge / (odds - 1)
        raw_stake = bankroll * fraction
        return floor(raw_stake / round_to) * round_to

    Args:
        prob: Estimated win probability in [0, 1].
        odds: Payout odds (e.g. 5.0 means 5x return on the stake).
        bankroll: Available bankroll in yen.
        kelly_fraction: Fractional Kelly multiplier in (0, 1].  0.25 = quarter Kelly.
        round_to: Stake is always a multiple of this value (default 100 yen).

    Returns:
        Integer stake in yen (>= 0, multiple of round_to).
    """
    edge = prob * odds - 1.0
    # `odds <= 1.0` ガードは数学的に edge > 0 なら不要だが、後段の (odds - 1.0)
    # による ZeroDivisionError を防ぐ防御として明示しておく
    if edge <= 0.0 or odds <= 1.0:
        return 0
    fraction = kelly_fraction * edge / (odds - 1.0)
    raw_stake = bankroll * fraction
    return int(math.floor(raw_stake / round_to)) * round_to


def assign_stakes(
    candidates: list[BetCandidate],
    bankroll: int,
    kelly_fraction: float,
    max_stake_per_race_pct: float,
    round_to: int = 100,
    keep_zero_stake: bool = False,
) -> list[BetCandidate]:
    """Assign Kelly stakes to candidates and apply per-race cap.

    Processing steps:
    1. ev <= 1.0  → stake = 0 (excluded from output unless keep_zero_stake=True).
    2. Apply kelly_stake to each remaining candidate.
    3. If the total stake exceeds bankroll * max_stake_per_race_pct, scale all
       stakes down proportionally (floor to round_to after scaling).
    4. Remove candidates with stake == 0 from the returned list (unless
       keep_zero_stake=True, in which case all candidates are returned with
       their computed stake, including zeros).

    Args:
        candidates: BetCandidate list (stake field is ignored on input).
        bankroll: Current bankroll in yen.
        kelly_fraction: Fractional Kelly multiplier.
        max_stake_per_race_pct: Maximum fraction of bankroll to spend per race
            (e.g. 0.05 = 5%).
        round_to: Stake rounding granularity in yen.
        keep_zero_stake: When True, candidates with stake=0 (due to ev<=1.0 or
            Kelly returning 0 or cap-scaling to 0) are included in the output.
            Defaults to False for backward compatibility.

    Returns:
        New list of BetCandidate (copies) with updated stake values.
        When keep_zero_stake=False (default), stake=0 items are excluded.
        When keep_zero_stake=True, all input candidates are returned with
        their computed stake (0 for ineligible / below-EV candidates).
    """
    cap = bankroll * max_stake_per_race_pct

    # Step 1 + 2: compute raw Kelly stakes
    # ev is None or ev <= 1.0 candidates get stake=0 and are tracked separately
    zero_stake_candidates: list[BetCandidate] = []
    staked: list[tuple[BetCandidate, int]] = []
    for c in candidates:
        if c.ev is None or c.ev <= 1.0:
            if keep_zero_stake:
                zero_stake_candidates.append(c.model_copy(update={"stake": 0}))
            continue
        # est_odds is guaranteed non-None here since ev = prob * est_odds
        s = kelly_stake(c.prob, c.est_odds, bankroll, kelly_fraction, round_to)  # type: ignore[arg-type]
        if s > 0:
            staked.append((c, s))
        elif keep_zero_stake:
            zero_stake_candidates.append(c.model_copy(update={"stake": 0}))

    if not staked:
        return zero_stake_candidates if keep_zero_stake else []

    # Step 3: proportional cap
    total = sum(s for _, s in staked)
    if total > cap:
        scale = cap / total
        staked = [
            (c, int(math.floor(s * scale / round_to)) * round_to)
            for c, s in staked
        ]

    # Step 4: build result list
    result: list[BetCandidate] = []
    for c, s in staked:
        if s > 0:
            result.append(c.model_copy(update={"stake": s}))
        elif keep_zero_stake:
            zero_stake_candidates.append(c.model_copy(update={"stake": 0}))

    if keep_zero_stake:
        return result + zero_stake_candidates
    return result


# ---------------------------------------------------------------------------
# Recommendation orchestration
# ---------------------------------------------------------------------------

def recommend_for_race(
    predictions: pd.DataFrame,
    combinations_by_type: dict[str, list[CombinationPrediction]],
    race_id: str,
    bankroll: int,
    kelly_fraction: float,
    max_stake_per_race_pct: float,
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
        bankroll: Current bankroll in yen.
        kelly_fraction: Fractional Kelly multiplier.
        max_stake_per_race_pct: Maximum fraction of bankroll per race.
        top_n_horses: Number of top horses (by win_prob) to include in box /
            formation candidates (default 3).
        enabled_bet_types: Bet types to consider.  None means all types
            present in combinations_by_type.

    Returns:
        RecommendationResult with candidates that have stake > 0.
    """
    if enabled_bet_types is not None:
        active_types = [bt for bt in enabled_bet_types if bt in combinations_by_type]
    else:
        active_types = list(combinations_by_type.keys())

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

    final_candidates = assign_stakes(
        deduped, bankroll, kelly_fraction, max_stake_per_race_pct,
        keep_zero_stake=True,
    )

    return RecommendationResult(
        race_id=race_id,
        bankroll_at_decision=bankroll,
        candidates=final_candidates,
    )
