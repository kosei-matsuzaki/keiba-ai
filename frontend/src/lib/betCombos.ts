// 買い方（流し / ボックス / フォーメーション）から馬券の組合せ（点）を展開する純粋関数群。
// 馬番(number)の配列で組合せを作り、formatCombo で券種ごとの表記文字列にする。
// 表記はバックエンド scraper/parsers/odds.py の _format_combo と一致させる:
//   - 単勝/複勝:        "5"
//   - 馬連/ワイド/三連複: 昇順 "-" 区切り  ("1-2", "1-2-3")
//   - 馬単/三連単:       着順そのまま "→" 区切り ("1→2", "1→2→3")

import type { BetType } from '@/types/api';

export type BetMethod = 'single' | 'box' | 'nagashi' | 'formation';

/** 券種の 1 点に含まれる頭数（単複=1 / 馬連系=2 / 三連系=3）。 */
export function betK(betType: BetType): number {
  if (betType === '単勝' || betType === '複勝') return 1;
  if (betType === '馬連' || betType === 'ワイド' || betType === '馬単') return 2;
  return 3; // 三連複 / 三連単
}

/** 着順が意味を持つ（順序あり）券種か。 */
export function isOrdered(betType: BetType): boolean {
  return betType === '馬単' || betType === '三連単';
}

/** 馬番配列を券種ごとの combo 文字列にする。 */
export function formatCombo(nums: number[], betType: BetType): string {
  if (nums.length === 1) return String(nums[0]);
  if (isOrdered(betType)) return nums.join('→');
  return [...nums].sort((a, b) => a - b).join('-');
}

function uniq(nums: number[]): number[] {
  return [...new Set(nums)];
}

/** k 個の組合せ（順序なし）。 */
export function combinations<T>(arr: T[], k: number): T[][] {
  if (k <= 0) return [[]];
  if (k > arr.length) return [];
  const res: T[][] = [];
  const rec = (start: number, cur: T[]) => {
    if (cur.length === k) {
      res.push([...cur]);
      return;
    }
    for (let i = start; i < arr.length; i++) {
      cur.push(arr[i]);
      rec(i + 1, cur);
      cur.pop();
    }
  };
  rec(0, []);
  return res;
}

/** k 個の順列（順序あり）。 */
export function permutations<T>(arr: T[], k: number): T[][] {
  if (k <= 0) return [[]];
  if (k > arr.length) return [];
  const res: T[][] = [];
  const used = new Array(arr.length).fill(false);
  const rec = (cur: T[]) => {
    if (cur.length === k) {
      res.push([...cur]);
      return;
    }
    for (let i = 0; i < arr.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(arr[i]);
      rec(cur);
      cur.pop();
      used[i] = false;
    }
  };
  rec([]);
  return res;
}

export interface ComboGenInput {
  betType: BetType;
  method: BetMethod;
  /** single: 1 点を構成する馬番（順序あり券種は着順どおり、長さ k）。 */
  single?: number[];
  /** box: 選んだ頭数の集合。 */
  box?: number[];
  /** nagashi: 軸（1 頭軸なら 1 つ、2 頭軸なら 2 つ。三連系のみ 2 頭可）。 */
  axes?: number[];
  /** nagashi(順序あり): 各軸の着順 1..k（axes と同じ長さ・相異）。相手は残り着を埋める。 */
  axisPositions?: number[];
  /** nagashi: 相手の集合。 */
  opponents?: number[];
  /** formation: k 列、各列の馬番集合。 */
  columns?: number[][];
}

/**
 * 買い方から馬番配列の組合せリストを展開する。無効/不足な入力は [] を返す。
 * 重複は呼び出し側で formatCombo 後にまとめて除去する想定。
 */
export function generateCombos(input: ComboGenInput): number[][] {
  const k = betK(input.betType);
  const ordered = isOrdered(input.betType);

  switch (input.method) {
    case 'single': {
      const h = (input.single ?? []).filter((x) => Number.isFinite(x));
      if (h.length !== k || new Set(h).size !== k) return [];
      return [h];
    }
    case 'box': {
      const s = uniq(input.box ?? []);
      if (s.length < k) return [];
      return ordered ? permutations(s, k) : combinations(s, k);
    }
    case 'nagashi': {
      const axes = uniq((input.axes ?? []).filter((x) => Number.isFinite(x)));
      // 軸は 1 頭以上 k 頭未満（相手で残りを埋めるため）。
      if (axes.length < 1 || axes.length >= k) return [];
      const opp = uniq(input.opponents ?? []).filter((o) => !axes.includes(o));
      const need = k - axes.length;
      if (opp.length < need) return [];

      if (!ordered) {
        // 順序なし: 軸 ∪ 相手から (k-軸数) 頭の組合せ
        return combinations(opp, need).map((c) => [...axes, ...c]);
      }

      // 順序あり: 各軸を指定着順に固定し、相手が残り着を順に埋める
      const positions = input.axisPositions ?? [];
      if (positions.length !== axes.length) return [];
      const posSet = new Set(positions);
      if (posSet.size !== positions.length) return []; // 着順重複は無効
      if (positions.some((p) => p < 1 || p > k)) return [];
      const remaining: number[] = [];
      for (let p = 1; p <= k; p++) if (!posSet.has(p)) remaining.push(p);

      return permutations(opp, need).map((q) => {
        const arr = new Array<number>(k);
        axes.forEach((a, idx) => {
          arr[positions[idx] - 1] = a;
        });
        remaining.forEach((p, idx) => {
          arr[p - 1] = q[idx];
        });
        return arr;
      });
    }
    case 'formation': {
      const cols = input.columns ?? [];
      if (cols.length !== k || cols.some((c) => c.length === 0)) return [];
      // 各列から 1 頭ずつ取り、全頭が相異なる組合せを作る（直積 + distinct）。
      let acc: number[][] = [[]];
      for (const col of cols) {
        const next: number[][] = [];
        for (const partial of acc) {
          for (const h of uniq(col)) {
            if (partial.includes(h)) continue;
            next.push([...partial, h]);
          }
        }
        acc = next;
      }
      if (!ordered) {
        // 順序なし券種はソート集合で重複排除
        const seen = new Set<string>();
        const out: number[][] = [];
        for (const combo of acc) {
          const key = [...combo].sort((x, y) => x - y).join('-');
          if (seen.has(key)) continue;
          seen.add(key);
          out.push(combo);
        }
        return out;
      }
      return acc;
    }
    default:
      return [];
  }
}

/** 買い方を combo 文字列の配列（重複排除済み）に展開する。 */
export function expandCombos(input: ComboGenInput): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const c of generateCombos(input)) {
    const s = formatCombo(c, input.betType);
    if (!seen.has(s)) {
      seen.add(s);
      out.push(s);
    }
  }
  return out;
}
