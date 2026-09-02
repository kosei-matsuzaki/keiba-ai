import { describe, it, expect } from 'vitest';
import { buildPurchaseGroups } from '../lib/purchaseGroups';
import type { RecommendationCandidate } from '../types/api';

function cand(
  betType: string,
  posts: number[],
  stake = 100,
  prob = 0.1
): RecommendationCandidate {
  return {
    bet_type: betType,
    combo: posts.join('-'),
    pattern: 'box',
    prob,
    est_odds: 10,
    est_odds_source: 'implied',
    ev: 1.0,
    stake,
    post_positions: posts,
  };
}

describe('buildPurchaseGroups', () => {
  it('全点に共通する 1 頭があれば流しにする', () => {
    // 馬連 3-1 / 3-5 / 3-8 → 「3 から 1,5,8 へ流し」
    const groups = buildPurchaseGroups([
      cand('馬連', [1, 3]),
      cand('馬連', [3, 5]),
      cand('馬連', [3, 8]),
    ]);
    expect(groups).toHaveLength(1);
    expect(groups[0].shape).toEqual({ kind: 'nagashi', axes: [3], partners: [1, 5, 8] });
    expect(groups[0].formula).toBe('3 → 1, 5, 8');
    expect(groups[0].points).toBe(3);
    expect(groups[0].totalStake).toBe(300);
  });

  it('出ている馬の全通りが揃っていればボックスにする', () => {
    // 馬連 C(3,2) = 3 点すべて
    const groups = buildPurchaseGroups([
      cand('馬連', [1, 3]),
      cand('馬連', [1, 7]),
      cand('馬連', [3, 7]),
    ]);
    expect(groups).toHaveLength(1);
    expect(groups[0].shape).toEqual({ kind: 'box', horses: [1, 3, 7] });
    expect(groups[0].formula).toBe('BOX 1, 3, 7');
  });

  it('三連複は 2 頭軸の流しにまとめる', () => {
    // 5-7-a / 5-7-b / 5-7-c → 「5,7 から a,b,c へ流し」
    const groups = buildPurchaseGroups([
      cand('三連複', [5, 7, 1]),
      cand('三連複', [5, 7, 2]),
      cand('三連複', [5, 7, 3]),
    ]);
    expect(groups).toHaveLength(1);
    expect(groups[0].shape).toEqual({ kind: 'nagashi', axes: [5, 7], partners: [1, 2, 3] });
    expect(groups[0].patternLabel).toBe('軸2頭流し');
    expect(groups[0].formula).toBe('5-7 → 1, 2, 3');
  });

  it('1 つの形に畳めなくても、複数の流しに分ける', () => {
    // 5-7-{1,2,3} と 5-11-{1,4} — **これを全部「個別」に落とさない**。
    // 覆える点数の多い軸から順に切り出す。
    const groups = buildPurchaseGroups([
      cand('三連複', [5, 7, 1]),
      cand('三連複', [5, 7, 2]),
      cand('三連複', [5, 7, 3]),
      cand('三連複', [5, 11, 1]),
      cand('三連複', [5, 11, 4]),
    ]);
    expect(groups).toHaveLength(2);
    expect(groups[0].shape).toEqual({ kind: 'nagashi', axes: [5, 7], partners: [1, 2, 3] });
    expect(groups[1].shape).toEqual({ kind: 'nagashi', axes: [5, 11], partners: [1, 4] });
    // 点数の合計は元の買い目と一致する (畳んで増減しない)
    expect(groups.reduce((n, g) => n + g.points, 0)).toBe(5);
  });

  it('順序券は先頭の着順だけを軸にする（1着と3着だけ固定は買えない）', () => {
    // 馬単 3→1 / 3→5 → 「3 (1着) から 1,5 へ流し」
    const groups = buildPurchaseGroups([cand('馬単', [3, 1]), cand('馬単', [3, 5])]);
    expect(groups[0].shape).toEqual({ kind: 'nagashi', axes: [3], partners: [1, 5] });

    // 三連単 5→7→1 / 5→7→2 は 1・2 着固定
    const trifecta = buildPurchaseGroups([
      cand('三連単', [5, 7, 1]),
      cand('三連単', [5, 7, 2]),
    ]);
    expect(trifecta[0].shape).toEqual({ kind: 'nagashi', axes: [5, 7], partners: [1, 2] });
  });

  it('本当に独立した買い目だけが単独の行になる', () => {
    // 1-3 と 5-7 は共通の馬が無いので、まとめようがない
    const groups = buildPurchaseGroups([cand('馬連', [1, 3]), cand('馬連', [5, 7])]);
    expect(groups).toHaveLength(2);
    expect(groups.every((g) => g.shape.kind === 'list')).toBe(true);
    expect(groups.every((g) => g.patternLabel === '単独')).toBe(true);
  });

  it('畳んでも買う点数は変わらない', () => {
    // 3 点だが軸も無く、BOX(1,3,5,7) = 6 点にも足りない構成
    const input = [cand('馬連', [1, 3]), cand('馬連', [5, 7]), cand('馬連', [1, 5])];
    const groups = buildPurchaseGroups(input);
    expect(groups.reduce((n, g) => n + g.points, 0)).toBe(input.length);
    expect(groups.reduce((n, g) => n + g.totalStake, 0)).toBe(300);
    // 元の買い目がすべてどれかのグループに入っている
    const combos = groups.flatMap((g) => g.candidates.map((c) => c.combo)).sort();
    expect(combos).toEqual(['1-3', '1-5', '5-7']);
  });

  it('単勝・複勝はまとめず単体として出す', () => {
    const groups = buildPurchaseGroups([cand('単勝', [4], 500), cand('複勝', [4], 500)]);
    expect(groups.map((g) => g.betType)).toEqual(['単勝', '複勝']);
    expect(groups[0].shape).toEqual({ kind: 'single', horses: [4] });
    expect(groups[0].formula).toBe('4');
  });

  it('賭けない候補 (stake=0) は購入対象に入れない', () => {
    const groups = buildPurchaseGroups([
      cand('馬連', [1, 3], 100),
      cand('馬連', [3, 5], 0),
      cand('馬連', [3, 8], 0),
    ]);
    expect(groups).toHaveLength(1);
    expect(groups[0].points).toBe(1);
    expect(groups[0].totalStake).toBe(100);
  });

  it('買う点が無ければ空を返す', () => {
    expect(buildPurchaseGroups([cand('馬連', [1, 3], 0)])).toEqual([]);
  });
});
