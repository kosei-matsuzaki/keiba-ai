import { describe, it, expect } from 'vitest';

import { gradeClass, groupByCourse, isGraded, raceNumber, splitGrade } from '../lib/races';
import type { RaceSummary } from '../types/api';

function race(race_id: string, course: string): RaceSummary {
  return {
    race_id,
    date: '2024-06-01',
    course,
    surface: '芝',
    distance: 2000,
    race_class: null,
    n_runners: 16,
    name: null,
  };
}

describe('raceNumber', () => {
  it('末尾 2 桁を数として返す', () => {
    expect(raceNumber('202406010101')).toBe(1);
    expect(raceNumber('202406010112')).toBe(12);
  });

  it('番号として読めない ID は null', () => {
    // 呼び出し側が NaNR / 0R を画面に出さずに済むように
    expect(raceNumber('RACE-XX')).toBeNull();
    expect(raceNumber('202406010100')).toBeNull();
    // 末尾 2 桁が数字なら受ける。テスト用の短い ID (CAL0501) もここを通る
    expect(raceNumber('CAL0501')).toBe(1);
  });
});

describe('splitGrade', () => {
  it('格は名前の末尾から取り、名前からは落とす', () => {
    expect(splitGrade('第56回アルゼンチン共和国杯(GII)', 'G2')).toEqual({
      grade: 'GII',
      label: '第56回アルゼンチン共和国杯',
    });
  });

  it('障害の重賞は race_class ではなく名前が正しい', () => {
    // race_class は平地と同じ "G1"。DB に JG* という race_class は 1 件も無い
    expect(splitGrade('第22回中山グランドジャンプ(JGI)', 'G1').grade).toBe('JGI');
  });

  it('末尾が取れないときは race_class で代替する', () => {
    // 「重賞」7 件は名前も (重賞) 止まりで格が読めない
    expect(splitGrade('第3回葵ステークス(重賞)', '重賞')).toEqual({
      grade: '重賞',
      label: '第3回葵ステークス(重賞)',
    });
  });
});

describe('isGraded', () => {
  it('名前由来と race_class 由来の両方の語彙を受ける', () => {
    expect(isGraded('JGIII')).toBe(true);
    expect(isGraded('GI')).toBe(true);
    expect(isGraded('G3')).toBe(true);
    expect(isGraded('重賞')).toBe(true);
  });

  it('Listed と OP は重賞ではない', () => {
    expect(isGraded('Listed')).toBe(false);
    expect(isGraded('OP')).toBe(false);
    expect(isGraded(null)).toBe(false);
  });
});

describe('gradeClass', () => {
  it('G1 級だけアクセント', () => {
    expect(gradeClass('G1')).toBe('text-primary');
    expect(gradeClass('GI')).toBe('text-primary');
    expect(gradeClass('JGI')).toBe('text-primary');
  });

  it('G2 以下は淡い', () => {
    expect(gradeClass('G2')).toBe('text-subtle-foreground');
    expect(gradeClass('JGII')).toBe('text-subtle-foreground');
    expect(gradeClass(null)).toBe('text-subtle-foreground');
  });
});

describe('groupByCourse', () => {
  it('並べ替えない — /api/races/by_date が race_id 昇順を返す', () => {
    const sections = groupByCourse([
      race('202405010101', '東京'),
      race('202405010111', '東京'),
      race('202408010209', '京都'),
    ]);

    expect(sections.map((s) => s.course)).toEqual(['東京', '京都']);
    expect(sections[0].races.map((r) => r.race_id)).toEqual([
      '202405010101',
      '202405010111',
    ]);
  });
});
