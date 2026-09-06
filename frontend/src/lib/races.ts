/**
 * レースの ID・格・開催場をどう読むか。
 *
 * 画面ごとに書くと、同じ問いに違う答えを持つ部品が並ぶ。実際に「race_class を
 * 信じる / 信じない」でカレンダーと日別ピッカーが反対を向いていた。
 */

import type { RaceSummary } from '@/types/api';

/**
 * race_id からレース番号。netkeiba の ID は 年+場+回+日+R で、末尾 2 桁が番号。
 *
 * 0 埋めするかは画面ごとに違うので数だけ返す。取れない ID は null —
 * 呼び出し側が `01R` や `NaNR` を出さずに済むように。
 */
export function raceNumber(raceId: string): number | null {
  const n = Number(raceId.slice(-2));
  return Number.isFinite(n) && n > 0 ? n : null;
}

/** 名前の末尾に付く格。「第69回日刊スポ賞中山金杯(GIII)」→ "GIII"。 */
const GRADE_SUFFIX = /[(（](J?G(?:III|II|I))[)）]\s*$/;

/**
 * 表示用に「格」と「名前」へ分ける。
 *
 * **格は race_class ではなく名前の末尾から取る。**race_class は障害の重賞も
 * "G1"/"G2"/"G3" として持っており (JG* という値は DB に 1 件も無い)、
 * それだけを出すと中山グランドジャンプが平地の G1 と同じ顔になる。
 * 名前側は netkeiba のまま JGI/JGIII を保っているので、そちらが正確。
 *
 * 末尾が取れない 12 件 (「重賞」7 件と括弧なし 5 件) は race_class で代替する。
 * 名前からは格を落とす — 隣に出すので二重になる。
 */
export function splitGrade(
  name: string,
  raceClass: string
): { grade: string; label: string } {
  const m = name.match(GRADE_SUFFIX);
  if (m) return { grade: m[1], label: name.slice(0, m.index).trim() };
  return { grade: raceClass, label: name };
}

/**
 * 名前由来の JGI/JGII/JGIII と、名前から取れなかったときの race_class の
 * 両方の語彙を受ける。**`splitGrade` が返した格を渡すこと** — race_class を
 * 直接渡すと、カレンダーと違う読み方で「重賞かどうか」を決めることになる。
 */
const GRADED = new Set([
  'GI',
  'GII',
  'GIII',
  'JGI',
  'JGII',
  'JGIII',
  'G1',
  'G2',
  'G3',
  '重賞',
]);

/** G3 以上か。backend の `_GRADED_CLASSES` と同じ判断。 */
export function isGraded(grade: string | null): boolean {
  return grade !== null && GRADED.has(grade);
}

/** G1 級だけアクセント。残りは淡くする — 月を眺めて「大きい日」を先に見つけるため。 */
export function gradeClass(grade: string | null): string {
  return grade === 'GI' || grade === 'JGI' || grade === 'G1'
    ? 'text-primary'
    : 'text-subtle-foreground';
}

export interface CourseSection {
  course: string;
  races: RaceSummary[];
}

/**
 * 開催場ごとに束ねる。**並べ替えない** — `/api/races/by_date` は race_id 昇順を
 * 返すと docstring で約束しており、race_id は 年+場+回+日+R なので場ごとに
 * 固まったうえで番号順になっている。
 */
export function groupByCourse(races: RaceSummary[]): CourseSection[] {
  const map = new Map<string, RaceSummary[]>();
  for (const race of races) {
    const list = map.get(race.course);
    if (list) list.push(race);
    else map.set(race.course, [race]);
  }
  return [...map.entries()].map(([course, rs]) => ({ course, races: rs }));
}
