/**
 * JRA の枠番と枠色。
 *
 * 枠色は競馬にしかない語彙で、利用者は枠で馬を識別する。馬番を枠色のチップで
 * 出すと一目で何のアプリか分かり、しかも実用的になる。
 *
 * **枠色は馬番専用の語彙にすること。** 他の用途 (ステータス・グラフ等) に
 * 流用すると、色が意味を持つという規則が壊れる。
 */
export const WAKU = [
  { bg: '#f5f5f5', fg: '#111111', name: '白' }, // 1
  { bg: '#1a1a1a', fg: '#f5f5f5', name: '黒' }, // 2
  { bg: '#d43b3b', fg: '#ffffff', name: '赤' }, // 3
  { bg: '#2f6fd0', fg: '#ffffff', name: '青' }, // 4
  { bg: '#e8c33a', fg: '#111111', name: '黄' }, // 5
  { bg: '#3fa860', fg: '#ffffff', name: '緑' }, // 6
  { bg: '#e08a2e', fg: '#111111', name: '橙' }, // 7
  { bg: '#e39ab5', fg: '#111111', name: '桃' }, // 8
] as const;

/**
 * 馬番 -> 枠番。JRA の標準的な割り当て（余りは外枠から 1 頭ずつ増える）。
 *
 * 8 頭以下は馬番 = 枠番。9 頭以上は 8 枠に均等配分し、余りは外枠 (8 枠側) から
 * 1 頭ずつ足す。18 頭立てなら 枠7 = 13・14・15 / 枠8 = 16・17・18 になる。
 *
 * NOTE: これは**暫定の導出**。本来は取得元に枠番があるので、スクレイパーで
 * 取って `EntrySummary` に載せるのが正しい (馬券種に枠連がある以上いずれ必要で、
 * 除外馬が出たときは馬番と頭数からの導出が実際とずれる)。
 */
export function wakuOf(umaban: number, runners: number): number {
  if (!Number.isFinite(umaban) || umaban < 1) return 0;
  if (runners <= 8) return umaban;
  const base = Math.floor(runners / 8);
  const extra = runners % 8; // 後ろ extra 枠が base+1 頭
  const boundary = (8 - extra) * base; // ここまでが base 頭ずつの枠
  return umaban <= boundary
    ? Math.ceil(umaban / base)
    : 8 - extra + Math.ceil((umaban - boundary) / (base + 1));
}

/** 枠番 (1-8) の色。範囲外は null。 */
export function wakuColor(waku: number): (typeof WAKU)[number] | null {
  return WAKU[waku - 1] ?? null;
}
