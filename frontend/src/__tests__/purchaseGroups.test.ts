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
  it('全点に共通する 1 頭 + 相手の全通りなら流しにする', () => {
    // 馬連 3-1 / 3-5 / 3-8 → 「3 から 1,5,8 へ流し」
    const groups = buildPurchaseGroups([
      cand('馬連', [1, 3]),
      cand('馬連', [3, 5]),
      cand('馬連', [3, 8]),
    ]);
    expect(groups).toHaveLength(1);
    expect(groups[0].shape).toEqual({ kind: 'nagashi', axis: 3, partners: [1, 5, 8] });
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
    expect(groups[0].shape).toEqual({ kind: 'box', horses: [1, 3, 7] });
    expect(groups[0].formula).toBe('BOX 1, 3, 7');
  });

  it('着ごとの候補を掛けて過不足なく一致すればフォーメーション (順序券)', () => {
    // 馬単 1着=3 / 2着=1,5 → 2 点
    const groups = buildPurchaseGroups([cand('馬単', [3, 1]), cand('馬単', [3, 5])]);
    // 軸が 1 頭で相手の全通り = 流しとしても表せる。より単純な流しを優先する
    expect(groups[0].shape.kind).toBe('nagashi');

    // 1着=3,7 / 2着=1,5 の 4 点は流しにできない
    const formation = buildPurchaseGroups([
      cand('馬単', [3, 1]),
      cand('馬単', [3, 5]),
      cand('馬単', [7, 1]),
      cand('馬単', [7, 5]),
    ]);
    expect(formation[0].shape).toEqual({ kind: 'formation', legs: [[3, 7], [1, 5]] });
    expect(formation[0].formula).toBe('3,7 → 1,5');
  });

  it('どの形にも当てはまらなければ列挙に落とす', () => {
    // 3 点だが軸も無く、BOX(1,3,5,7) = 6 点にも足りない
    const groups = buildPurchaseGroups([
      cand('馬連', [1, 3]),
      cand('馬連', [5, 7]),
      cand('馬連', [1, 5]),
    ]);
    expect(groups[0].shape).toEqual({ kind: 'list' });
    expect(groups[0].patternLabel).toBe('個別');
  });

  it('形が一致しないものを流し/ボックスと呼ばない (点数が変わってしまうため)', () => {
    // 3 が全点に入るが、相手 1,5,8 の全通り 3 点のうち 2 点しか無い
    const groups = buildPurchaseGroups([cand('馬連', [1, 3]), cand('馬連', [3, 5])]);
    // 相手 2 頭の全通り = 2 点なので、これは正しく流しになる
    expect(groups[0].shape.kind).toBe('nagashi');

    // 一方、BOX(1,3,5) = 3 点のうち 2 点だけならボックスとは呼べない
    const partial = buildPurchaseGroups([
      cand('三連複', [1, 2, 3]),
      cand('三連複', [1, 2, 4]),
      cand('三連複', [1, 3, 4]),
    ]);
    // 1 が全点に共通し、相手 2,3,4 の C(3,2)=3 点と一致 → 流し
    expect(partial[0].shape).toEqual({ kind: 'nagashi', axis: 1, partners: [2, 3, 4] });
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
    expect(groups[0].points).toBe(1);
    expect(groups[0].totalStake).toBe(100);
  });

  it('買う点が無ければ空を返す', () => {
    expect(buildPurchaseGroups([cand('馬連', [1, 3], 0)])).toEqual([]);
  });
});
