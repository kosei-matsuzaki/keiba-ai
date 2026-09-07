import { describe, it, expect } from 'vitest';

import { visibleProfitPoints } from '../lib/profitSeries';
import type { BetTimeseriesPoint } from '../types/api';

function pt(date: string, bets: number, cumulative_profit: number): BetTimeseriesPoint {
  return { date, invested: 0, payout: 0, profit: 0, cumulative_profit, bets };
}

describe('visibleProfitPoints', () => {
  it('買っていない日を落とす', () => {
    // 競馬は土日開催なので、API の 0 埋めをそのまま描くと 8 割以上が横ばいの
    // 平日になり、線が階段状に潰れる (実データで 12ヶ月 372 点中 53 点)。
    const points = [
      pt('2024-06-01', 3, 2000),
      pt('2024-06-02', 0, 2000),
      pt('2024-06-03', 0, 2000),
      pt('2024-06-08', 2, 0),
    ];
    expect(visibleProfitPoints(points).map((p) => p.date)).toEqual([
      '2024-06-01',
      '2024-06-08',
    ]);
  });

  it('累計損益はそのまま持ち越す (落とした日のぶんを足し直さない)', () => {
    // 落とすのは点だけで、値は API が計算した累計をそのまま使う。
    const points = [pt('2024-06-01', 3, 2000), pt('2024-06-02', 0, 2000), pt('2024-06-08', 2, -500)];
    expect(visibleProfitPoints(points).map((p) => p.cumulative_profit)).toEqual([2000, -500]);
  });

  it('1 日も買っていなければ空にする (呼び出し側が「記録すると出ます」を出す)', () => {
    expect(visibleProfitPoints([pt('2024-06-01', 0, 0), pt('2024-06-02', 0, 0)])).toEqual([]);
  });
});
