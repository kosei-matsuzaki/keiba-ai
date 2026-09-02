import type { EntrySummary, HorsePrediction } from '@/types/api';

/**
 * 出走馬テーブルの並べ替え規則。**JSX を含まない純粋な計算**なので lib に置く。
 *
 * RaceDetail.tsx が 958 行に伸びていたので、表 (components/EntryPredictionTable)
 * と画面から切り出した。ここが変わる理由は「列を足す / 並べ替えの決まりを変える」
 * だけで、画面の都合とは独立している。
 */
/**
 * 単勝期待値 (EV) = 単勝確率 × 単勝オッズ。
 * 現行モデルは decision-focused（ROI 直接最適化）なので、score より EV が
 * 「買うべきか」の主指標。オッズ未確定 (odds_win=null) のときは null。
 */
export function winEv(pred: HorsePrediction | null, entry: EntrySummary | undefined): number | null {
  if (!pred || entry?.odds_win == null) return null;
  return pred.win_prob * entry.odds_win;
}

export interface EntryRow {
  entry: EntrySummary;
  pred: HorsePrediction | null;
}

export type SortKey =
  | 'post_position'
  | 'horse_name'
  | 'odds_win'
  | 'popularity'
  | 'finish_position'
  | 'score'
  | 'win_prob'
  | 'place_prob'
  | 'win_ev';

export type SortDir = 'asc' | 'desc';

export interface SortState {
  key: SortKey;
  dir: SortDir;
}

// Columns that use asc as the initial direction when first clicked
export const ASC_FIRST_KEYS: ReadonlySet<SortKey> = new Set(['post_position', 'popularity']);

/**
 * Compare two nullable numbers for sort purposes, returning the raw numeric
 * difference (positive = a after b, negative = a before b, 0 = equal).
 * null / NaN comparisons are NOT handled here — handled in sortRows to keep
 * null-last logic independent of sort direction.
 */
function compareNumbers(a: number, b: number): number {
  return a - b;
}

function compareStrings(a: string, b: string): number {
  return a.localeCompare(b, 'ja');
}

/**
 * Extract the raw numeric value for a given sort key from a row.
 * Returns undefined when the value is null / NaN / undefined.
 */
function numericValue(row: EntryRow, key: SortKey): number | undefined {
  let v: number | null | undefined;
  switch (key) {
    case 'post_position': v = row.entry.post_position; break;
    case 'odds_win':      v = row.entry.odds_win; break;
    case 'popularity':    v = row.entry.popularity; break;
    case 'finish_position': v = row.entry.finish_position; break;
    case 'score':         v = row.pred?.score; break;
    case 'win_prob':      v = row.pred?.win_prob; break;
    case 'place_prob':    v = row.pred?.place_prob; break;
    case 'win_ev':        v = winEv(row.pred, row.entry); break;
    default: return undefined;
  }
  if (v == null || isNaN(v as number)) return undefined;
  return v as number;
}

export function sortRows(rows: EntryRow[], sort: SortState): EntryRow[] {
  const multiplier = sort.dir === 'asc' ? 1 : -1;

  return [...rows].sort((a, b) => {
    if (sort.key === 'horse_name') {
      const aNull = a.entry.horse_name == null;
      const bNull = b.entry.horse_name == null;
      if (aNull && bNull) return 0;
      // null is always last regardless of direction
      if (aNull) return 1;
      if (bNull) return -1;
      return compareStrings(a.entry.horse_name!, b.entry.horse_name!) * multiplier;
    }

    const av = numericValue(a, sort.key);
    const bv = numericValue(b, sort.key);

    if (av === undefined && bv === undefined) return 0;
    // null / NaN is always last regardless of direction
    if (av === undefined) return 1;
    if (bv === undefined) return -1;

    return compareNumbers(av, bv) * multiplier;
  });
}
