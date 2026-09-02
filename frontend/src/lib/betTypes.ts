import type { BetType } from '@/types/api';

/**
 * 対象になりうる馬券種の全リスト（表示順）。
 *
 * Settings（全レース共通の既定値）と、レース詳細の「この予想の条件」
 * （1 レースだけの上書き）の両方から参照する。
 *
 * **枠連は含めない。** オッズと払戻は取得しているが、AI は枠連の買い目を
 * 生成しない（backend の `COMBINATION_BET_TYPES` に無い）。選べるのに何も
 * 起きない選択肢になるため、UI からは外している。
 */
export const ALL_BET_TYPES = [
  '単勝',
  '複勝',
  '馬連',
  'ワイド',
  '馬単',
  '三連複',
  '三連単',
] as const satisfies readonly BetType[];

/** UI で選べる馬券種。`BetType` は枠連を含むが、こちらは含まない。 */
export type SelectableBetType = (typeof ALL_BET_TYPES)[number];

/**
 * 連系（2 頭以上の組合せ）。単複と扱いが違う場所で使う。
 *
 * 単複は「AI の本命を買う」ルールで点数が固定だが、連系は**的中確率の下限**で
 * 買う点数がレースごとに変わる（combo_min_hit_prob）。
 */
export const COMBO_BET_TYPES = [
  '馬連',
  'ワイド',
  '馬単',
  '三連複',
  '三連単',
] as const satisfies readonly BetType[];
