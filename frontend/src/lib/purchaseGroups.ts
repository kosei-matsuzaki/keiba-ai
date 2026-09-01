import type { RecommendationCandidate } from '@/types/api';

/**
 * 買い目を **窓口で買う形** にまとめる。
 *
 * 推奨買目は 1 点ずつ出てくるが、実際に買うときは「馬連 3 から 1,5,8 へ流し」の
 * ように 1 回の操作で複数点を買う。1 点ずつの表をそのまま写すのは手間が多く、
 * 買い間違いも起きる。**同じ集合を最小の操作数で表現し直す**のがこの関数の役目。
 *
 * 判定は「その形で買うと、いま推奨されている点と過不足なく一致するか」で行う。
 * 一致しないものを流しやボックスと呼ぶと、**買った点数が推奨と変わってしまう**ので、
 * 当てはまらなければ素直に列挙 (`list`) に落とす。
 */
export type PurchaseShape =
  | { kind: 'single'; horses: number[] }
  | { kind: 'nagashi'; axis: number; partners: number[] }
  | { kind: 'box'; horses: number[] }
  | { kind: 'formation'; legs: number[][] }
  | { kind: 'list' };

export interface PurchaseGroup {
  key: string;
  betType: string;
  shape: PurchaseShape;
  /** 「軸1頭流し」「ボックス」など、買い方の名前。 */
  patternLabel: string;
  /** 「3 → 1, 5, 8」のような式。窓口・ネット投票にそのまま写せる形。 */
  formula: string;
  points: number;
  totalStake: number;
  candidates: RecommendationCandidate[];
}

/** 着順が意味を持つ券種。ここだけ combo の並び順を情報として扱える。 */
const ORDERED_TYPES = new Set(['馬単', '三連単']);
/** 単複はまとめる対象ではない (1 頭 1 点)。 */
const SINGLE_TYPES = new Set(['単勝', '複勝']);

function uniqueSorted(values: number[]): number[] {
  return [...new Set(values)].sort((a, b) => a - b);
}

function combinationCount(n: number, k: number): number {
  if (k > n) return 0;
  let out = 1;
  for (let i = 0; i < k; i += 1) out = (out * (n - i)) / (i + 1);
  return Math.round(out);
}

function permutationCount(n: number, k: number): number {
  if (k > n) return 0;
  let out = 1;
  for (let i = 0; i < k; i += 1) out *= n - i;
  return out;
}

/** 全通りの積 (フォーメーションの点数)。同じ馬が複数の着に入る重複は除く。 */
function formationPoints(legs: number[][], ordered: boolean): number {
  if (!ordered) return 0; // 順序なしのフォーメーションは扱わない (下で list に落ちる)
  const count = (idx: number, used: Set<number>): number => {
    if (idx >= legs.length) return 1;
    let total = 0;
    for (const horse of legs[idx]) {
      if (used.has(horse)) continue;
      used.add(horse);
      total += count(idx + 1, used);
      used.delete(horse);
    }
    return total;
  };
  return count(0, new Set());
}

function detectShape(betType: string, candidates: RecommendationCandidate[]): PurchaseShape {
  const sets = candidates.map((c) => c.post_positions);
  const size = sets[0]?.length ?? 0;
  if (size === 0) return { kind: 'list' };

  if (SINGLE_TYPES.has(betType)) {
    return { kind: 'single', horses: uniqueSorted(sets.flat()) };
  }

  const union = uniqueSorted(sets.flat());
  const ordered = ORDERED_TYPES.has(betType);

  // ボックス: 出ている馬の全通りが揃っているか
  const boxPoints = ordered
    ? permutationCount(union.length, size)
    : combinationCount(union.length, size);
  if (boxPoints === sets.length && union.length > size) {
    return { kind: 'box', horses: union };
  }

  // 流し: 全点に共通して入っている馬が 1 頭だけあり、残りが相手の全通り
  const common = union.filter((h) => sets.every((s) => s.includes(h)));
  if (common.length === 1) {
    const axis = common[0];
    const partners = union.filter((h) => h !== axis);
    const expected = ordered
      ? permutationCount(partners.length, size - 1)
      : combinationCount(partners.length, size - 1);
    if (expected === sets.length) {
      return { kind: 'nagashi', axis, partners };
    }
  }

  // フォーメーション: 着ごとの候補を掛け合わせて過不足なく一致するか (順序券のみ)
  if (ordered) {
    const legs: number[][] = [];
    for (let i = 0; i < size; i += 1) legs.push(uniqueSorted(sets.map((s) => s[i])));
    if (formationPoints(legs, true) === sets.length) {
      return { kind: 'formation', legs };
    }
  }

  return { kind: 'list' };
}

function shapeLabel(shape: PurchaseShape): string {
  switch (shape.kind) {
    case 'single':
      return '単体';
    case 'nagashi':
      return '軸1頭流し';
    case 'box':
      return 'ボックス';
    case 'formation':
      return 'フォーメーション';
    default:
      return '個別';
  }
}

function shapeFormula(shape: PurchaseShape, candidates: RecommendationCandidate[]): string {
  switch (shape.kind) {
    case 'single':
      return shape.horses.join(', ');
    case 'nagashi':
      return `${shape.axis} → ${shape.partners.join(', ')}`;
    case 'box':
      return `BOX ${shape.horses.join(', ')}`;
    case 'formation':
      return shape.legs.map((leg) => leg.join(',')).join(' → ');
    default:
      return candidates.map((c) => c.combo).join(' / ');
  }
}

/**
 * 賭ける買い目を券種ごとにまとめ、買い方の形に変換する。
 *
 * `stake === 0` の候補 (予算に入らなかったもの) は購入対象ではないので外す。
 * 券種の並びは推奨と同じ「単勝 → 複勝 → 連系」で、同券種内は的中確率の高い順。
 */
export function buildPurchaseGroups(candidates: RecommendationCandidate[]): PurchaseGroup[] {
  const buying = candidates.filter((c) => c.stake > 0);
  const byType = new Map<string, RecommendationCandidate[]>();
  for (const c of buying) {
    const list = byType.get(c.bet_type) ?? [];
    list.push(c);
    byType.set(c.bet_type, list);
  }

  const groups: PurchaseGroup[] = [];
  for (const [betType, list] of byType) {
    const sorted = [...list].sort((a, b) => b.prob - a.prob);
    const shape = detectShape(betType, sorted);
    groups.push({
      key: betType,
      betType,
      shape,
      patternLabel: shapeLabel(shape),
      formula: shapeFormula(shape, sorted),
      points: sorted.length,
      totalStake: sorted.reduce((n, c) => n + c.stake, 0),
      candidates: sorted,
    });
  }
  return groups;
}
