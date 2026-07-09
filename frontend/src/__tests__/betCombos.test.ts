import { describe, it, expect } from 'vitest';
import { expandCombos, betK, isOrdered, formatCombo } from '../lib/betCombos';

describe('betCombos helpers', () => {
  it('betK / isOrdered per bet type', () => {
    expect(betK('単勝')).toBe(1);
    expect(betK('馬連')).toBe(2);
    expect(betK('三連単')).toBe(3);
    expect(isOrdered('馬連')).toBe(false);
    expect(isOrdered('馬単')).toBe(true);
    expect(isOrdered('三連単')).toBe(true);
  });

  it('formatCombo: unordered sorts ascending, ordered keeps order', () => {
    expect(formatCombo([5], '単勝')).toBe('5');
    expect(formatCombo([3, 1], '馬連')).toBe('1-3');
    expect(formatCombo([3, 1, 2], '三連複')).toBe('1-2-3');
    expect(formatCombo([3, 1], '馬単')).toBe('3→1');
    expect(formatCombo([3, 1, 2], '三連単')).toBe('3→1→2');
  });
});

describe('expandCombos — box', () => {
  it('馬連ボックス 3頭 → 3点', () => {
    const c = expandCombos({ betType: '馬連', method: 'box', box: [1, 2, 3] });
    expect(c.sort()).toEqual(['1-2', '1-3', '2-3']);
  });

  it('三連単ボックス 3頭 → 6点（順列）', () => {
    const c = expandCombos({ betType: '三連単', method: 'box', box: [1, 2, 3] });
    expect(c).toHaveLength(6);
    expect(c).toContain('1→2→3');
    expect(c).toContain('3→2→1');
  });

  it('単勝ボックス → 各馬の単勝', () => {
    expect(expandCombos({ betType: '単勝', method: 'box', box: [4, 7] }).sort()).toEqual(['4', '7']);
  });

  it('頭数不足は空', () => {
    expect(expandCombos({ betType: '三連複', method: 'box', box: [1, 2] })).toEqual([]);
  });
});

describe('expandCombos — nagashi (1頭軸)', () => {
  it('馬連ながし（軸1 + 相手3） → 3点', () => {
    const c = expandCombos({ betType: '馬連', method: 'nagashi', axes: [1], opponents: [2, 3, 4] });
    expect(c.sort()).toEqual(['1-2', '1-3', '1-4']);
  });

  it('馬単 1着ながし', () => {
    const c = expandCombos({
      betType: '馬単', method: 'nagashi', axes: [1], axisPositions: [1], opponents: [2, 3],
    });
    expect(c.sort()).toEqual(['1→2', '1→3']);
  });

  it('馬単 2着ながし', () => {
    const c = expandCombos({
      betType: '馬単', method: 'nagashi', axes: [1], axisPositions: [2], opponents: [2, 3],
    });
    expect(c.sort()).toEqual(['2→1', '3→1']);
  });

  it('三連複 軸1頭ながし（軸1 + 相手3 → C(3,2)=3点）', () => {
    const c = expandCombos({ betType: '三連複', method: 'nagashi', axes: [1], opponents: [2, 3, 4] });
    expect(c.sort()).toEqual(['1-2-3', '1-2-4', '1-3-4']);
  });

  it('三連単 1着ながし（軸→相手2頭の順列）', () => {
    const c = expandCombos({
      betType: '三連単', method: 'nagashi', axes: [1], axisPositions: [1], opponents: [2, 3],
    });
    expect(c.sort()).toEqual(['1→2→3', '1→3→2']);
  });
});

describe('expandCombos — nagashi (2頭軸)', () => {
  it('三連複 軸2頭ながし（軸1,2 + 相手3 → 各相手で1点ずつ）', () => {
    const c = expandCombos({
      betType: '三連複', method: 'nagashi', axes: [1, 2], opponents: [3, 4, 5],
    });
    expect(c.sort()).toEqual(['1-2-3', '1-2-4', '1-2-5']);
  });

  it('三連単 軸2頭ながし（1着=1,2着=2固定、相手が3着）', () => {
    const c = expandCombos({
      betType: '三連単', method: 'nagashi', axes: [1, 2], axisPositions: [1, 2], opponents: [3, 4],
    });
    expect(c.sort()).toEqual(['1→2→3', '1→2→4']);
  });

  it('三連単 軸2頭ながし（1着=1,3着=2固定、相手が2着）', () => {
    const c = expandCombos({
      betType: '三連単', method: 'nagashi', axes: [1, 2], axisPositions: [1, 3], opponents: [3, 4],
    });
    expect(c.sort()).toEqual(['1→3→2', '1→4→2']);
  });

  it('軸の着順が重複していたら空', () => {
    const c = expandCombos({
      betType: '三連単', method: 'nagashi', axes: [1, 2], axisPositions: [1, 1], opponents: [3, 4],
    });
    expect(c).toEqual([]);
  });
});

describe('expandCombos — formation', () => {
  it('三連単フォーメーション 1着[1] 2着[2,3] 3着[2,3,4]', () => {
    const c = expandCombos({
      betType: '三連単', method: 'formation', columns: [[1], [2, 3], [2, 3, 4]],
    });
    // 1→2→3, 1→2→4, 1→3→2, 1→3→4 (同一馬は除外)
    expect(c.sort()).toEqual(['1→2→3', '1→2→4', '1→3→2', '1→3→4']);
  });

  it('馬連フォーメーションは順序なしで重複排除', () => {
    const c = expandCombos({ betType: '馬連', method: 'formation', columns: [[1, 2], [2, 3]] });
    // (1,2),(1,3),(2,3) — (2,2)は同一馬除外
    expect(c.sort()).toEqual(['1-2', '1-3', '2-3']);
  });
});

describe('expandCombos — single', () => {
  it('三連単 単一点は着順どおり', () => {
    expect(expandCombos({ betType: '三連単', method: 'single', single: [3, 1, 2] })).toEqual(['3→1→2']);
  });
  it('重複馬や頭数不足は空', () => {
    expect(expandCombos({ betType: '三連複', method: 'single', single: [1, 1, 2] })).toEqual([]);
    expect(expandCombos({ betType: '馬連', method: 'single', single: [1] })).toEqual([]);
  });
});
