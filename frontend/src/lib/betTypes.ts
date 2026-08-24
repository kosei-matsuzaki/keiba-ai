import type { BetType } from '@/types/api';

/**
 * 対象になりうる馬券種の全リスト（表示順）。
 *
 * Settings（全レース共通の既定値）と、レース詳細の「この予想の条件」
 * （1 レースだけの上書き）の両方から参照する。
 */
export const ALL_BET_TYPES: BetType[] = [
  '単勝',
  '複勝',
  '枠連',
  '馬連',
  'ワイド',
  '馬単',
  '三連複',
  '三連単',
];
