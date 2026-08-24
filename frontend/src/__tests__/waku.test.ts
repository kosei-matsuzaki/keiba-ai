import { describe, it, expect } from 'vitest';
import { wakuOf, wakuColor, WAKU } from '../lib/waku';

/** 各枠に入る馬番を集める。 */
function bracketsFor(runners: number): number[][] {
  const out: number[][] = Array.from({ length: 8 }, () => []);
  for (let n = 1; n <= runners; n++) {
    out[wakuOf(n, runners) - 1].push(n);
  }
  return out;
}

describe('wakuOf', () => {
  it('8 頭以下は馬番 = 枠番', () => {
    for (const runners of [5, 8]) {
      for (let n = 1; n <= runners; n++) {
        expect(wakuOf(n, runners)).toBe(n);
      }
    }
  });

  it('18 頭立ては JRA の実際の割り当てと一致する', () => {
    const brackets = bracketsFor(18);
    expect(brackets[6]).toEqual([13, 14, 15]); // 枠7
    expect(brackets[7]).toEqual([16, 17, 18]); // 枠8
    expect(brackets[0]).toEqual([1, 2]); // 枠1
  });

  it('16 頭立ては各枠 2 頭ずつ', () => {
    for (const b of bracketsFor(16)) {
      expect(b).toHaveLength(2);
    }
  });

  it('余りは外枠から 1 頭ずつ増える（枠ごとの頭数差は 1 以内）', () => {
    for (const runners of [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]) {
      const brackets = bracketsFor(runners);
      const sizes = brackets.map((b) => b.length);
      expect(Math.max(...sizes) - Math.min(...sizes)).toBeLessThanOrEqual(1);
      // 内枠より外枠のほうが多い（等しいことはあっても逆転しない）
      for (let i = 1; i < sizes.length; i++) {
        expect(sizes[i]).toBeGreaterThanOrEqual(sizes[i - 1]);
      }
    }
  });

  it('全馬がいずれかの枠 (1-8) に入り、重複しない', () => {
    for (const runners of [9, 12, 17, 18]) {
      const seen = bracketsFor(runners).flat();
      expect(seen).toHaveLength(runners);
      expect(new Set(seen).size).toBe(runners);
      for (let n = 1; n <= runners; n++) {
        const w = wakuOf(n, runners);
        expect(w).toBeGreaterThanOrEqual(1);
        expect(w).toBeLessThanOrEqual(8);
      }
    }
  });

  it('枠番は馬番に対して単調非減少', () => {
    for (const runners of [9, 11, 14, 18]) {
      for (let n = 2; n <= runners; n++) {
        expect(wakuOf(n, runners)).toBeGreaterThanOrEqual(wakuOf(n - 1, runners));
      }
    }
  });

  it('不正な馬番は 0 を返す', () => {
    expect(wakuOf(0, 18)).toBe(0);
    expect(wakuOf(NaN, 18)).toBe(0);
  });
});

describe('wakuColor', () => {
  it('1-8 枠に色がある', () => {
    for (let w = 1; w <= 8; w++) {
      expect(wakuColor(w)).toBe(WAKU[w - 1]);
    }
  });

  it('範囲外は null', () => {
    expect(wakuColor(0)).toBeNull();
    expect(wakuColor(9)).toBeNull();
  });
});
