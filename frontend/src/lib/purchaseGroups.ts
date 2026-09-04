import type { RecommendationCandidate } from '@/types/api';

/**
 * 買い目を **窓口で買う形** にまとめる。
 *
 * 推奨買目は 1 点ずつ出てくるが、実際に買うときは「馬連 3 から 1,5,8 へ流し」の
 * ように 1 回の操作で複数点を買う。1 点ずつの表をそのまま写すのは手間が多く、
 * 買い間違いも起きる。**同じ集合を最小の操作数で表現し直す**のがこの関数の役目。
 *
 * **1 券種を 1 つの形に畳めなくても諦めない。** 以前は畳めなければ全部「個別」に
 * 落としていたが、実際には「5-7-a,b,c」と「5-11-a,e」のように**複数の流し**に
 * 分ければほとんど畳める。券種ごとに、覆える点数の多い軸から順に切り出していく
 * (集合被覆の貪欲法)。本当に独立している点だけが単独の行として残る。
 *
 * 判定は「その形で買うと、いま推奨されている点と過不足なく一致するか」で行う。
 * 一致しないものを流しやボックスと呼ぶと、**買った点数が推奨と変わってしまう**。
 */
export type PurchaseShape =
  | { kind: 'single'; horses: number[] }
  /** 軸 (1〜2 頭) から相手へ流す。`axes` の並びは順序券なら着順そのまま。 */
  | { kind: 'nagashi'; axes: number[]; partners: number[] }
  | { kind: 'box'; horses: number[] }
  | { kind: 'formation'; legs: number[][] }
  /** 畳めなかった 1 点。 */
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

/**
 * 軸の候補を数え上げる。
 *
 * - 順序券: **先頭から連続する着順**だけを軸にする（1着固定、1・2着固定）。
 *   「1着と3着だけ固定」は窓口で買えないため。
 * - 順序なし: 買い目から `size - 1` 頭を選ぶ組合せすべて。
 */
function anchorsOf(posts: number[], ordered: boolean): number[][] {
  const size = posts.length;
  if (size < 2) return [];
  if (ordered) {
    const out: number[][] = [];
    for (let len = size - 1; len >= 1; len -= 1) out.push(posts.slice(0, len));
    return out;
  }
  const sorted = [...posts].sort((a, b) => a - b);
  const out: number[][] = [];
  // size-1 頭を軸にする組合せ (三連複なら 2 頭軸、馬連なら 1 頭軸)
  for (let skip = 0; skip < size; skip += 1) {
    out.push(sorted.filter((_, i) => i !== skip));
  }
  return out;
}

function anchorKey(anchor: number[]): string {
  return anchor.join('-');
}

/** 軸を除いた残りの馬。順序券は後ろ側、順序なしは差集合。 */
function partnerOf(posts: number[], anchor: number[], ordered: boolean): number[] {
  if (ordered) return posts.slice(anchor.length);
  const rest = [...posts];
  for (const h of anchor) {
    const i = rest.indexOf(h);
    if (i >= 0) rest.splice(i, 1);
  }
  return rest;
}

/**
 * 券種ごとの買い目を、買える形の集まりに分ける（貪欲法）。
 *
 * 1. まず全体がボックスで表せるならそれ 1 つ
 * 2. そうでなければ「最も多くの点を覆う軸」を選んで流しとして切り出し、残りで繰り返す
 * 3. 2 点以上を覆う軸が無くなったら、残りは 1 点ずつ独立した行にする
 */
function partition(
  betType: string,
  candidates: RecommendationCandidate[]
): { shape: PurchaseShape; candidates: RecommendationCandidate[] }[] {
  const ordered = ORDERED_TYPES.has(betType);
  const size = candidates[0]?.post_positions.length ?? 0;

  if (SINGLE_TYPES.has(betType) || size < 2) {
    return [
      {
        shape: { kind: 'single', horses: uniqueSorted(candidates.flatMap((c) => c.post_positions)) },
        candidates,
      },
    ];
  }

  // 1. 全体がボックスか (出ている馬の全通りが揃っているか)
  const union = uniqueSorted(candidates.flatMap((c) => c.post_positions));
  const boxPoints = ordered
    ? permutationCount(union.length, size)
    : combinationCount(union.length, size);
  if (boxPoints === candidates.length && union.length > size) {
    return [{ shape: { kind: 'box', horses: union }, candidates }];
  }

  // 2. 覆える点数の多い軸から切り出す
  const groups: { shape: PurchaseShape; candidates: RecommendationCandidate[] }[] = [];
  let remaining = [...candidates];

  while (remaining.length > 0) {
    const buckets = new Map<string, { anchor: number[]; items: RecommendationCandidate[] }>();
    for (const c of remaining) {
      for (const anchor of anchorsOf(c.post_positions, ordered)) {
        const key = anchorKey(anchor);
        const bucket = buckets.get(key) ?? { anchor, items: [] };
        bucket.items.push(c);
        buckets.set(key, bucket);
      }
    }

    let best: { anchor: number[]; items: RecommendationCandidate[] } | null = null;
    for (const bucket of buckets.values()) {
      if (bucket.items.length < 2) continue;
      // 覆える点数が多い方を優先。同点なら軸が多い (= 買い目が絞れている) 方
      if (
        best == null ||
        bucket.items.length > best.items.length ||
        (bucket.items.length === best.items.length && bucket.anchor.length > best.anchor.length)
      ) {
        best = bucket;
      }
    }

    if (best == null) {
      // 2 点以上を覆う軸が無い = ここから先は本当に独立した買い目
      for (const c of remaining) groups.push({ shape: { kind: 'list' }, candidates: [c] });
      break;
    }

    const partners = uniqueSorted(
      best.items.flatMap((c) => partnerOf(c.post_positions, best!.anchor, ordered))
    );
    groups.push({
      shape: { kind: 'nagashi', axes: best.anchor, partners },
      candidates: best.items,
    });
    const taken = new Set(best.items);
    remaining = remaining.filter((c) => !taken.has(c));
  }

  return groups;
}

function shapeLabel(shape: PurchaseShape): string {
  switch (shape.kind) {
    case 'single':
      return '単体';
    case 'nagashi':
      return `軸${shape.axes.length}頭流し`;
    case 'box':
      return 'ボックス';
    case 'formation':
      return 'フォーメーション';
    default:
      return '単独';
  }
}

function shapeFormula(shape: PurchaseShape, candidates: RecommendationCandidate[]): string {
  switch (shape.kind) {
    case 'single':
      return shape.horses.join(', ');
    case 'nagashi':
      return `${shape.axes.join('-')} → ${shape.partners.join(', ')}`;
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
 * 券種の並びは推奨と同じ「単勝 → 複勝 → 連系」で、同券種内は確信度の高い順。
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
    partition(betType, sorted).forEach((part, i) => {
      const items = [...part.candidates].sort((a, b) => b.prob - a.prob);
      groups.push({
        key: `${betType}#${i}`,
        betType,
        shape: part.shape,
        patternLabel: shapeLabel(part.shape),
        formula: shapeFormula(part.shape, items),
        points: items.length,
        totalStake: items.reduce((n, c) => n + c.stake, 0),
        candidates: items,
      });
    });
  }
  return groups;
}
