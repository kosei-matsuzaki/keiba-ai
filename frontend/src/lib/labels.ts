/**
 * ラベル用のタイポグラフィ classname を選ぶヘルパ。
 *
 * `.text-label` は uppercase + letter-spacing 0.18em を掛ける欧文用のスタイル。
 * これを和文 (複勝的中率 / 単勝回収率 など) に当てると、uppercase は無意味なうえ
 * 字間が空いて「詰めていない組版」に見える。和文を含むラベルは `.text-label-ja`
 * (letter-spacing 0 / 12px / sans) を使う。
 *
 *   labelClass('NDCG@3')    → 'text-label'
 *   labelClass('単勝回収率') → 'text-label-ja'
 */
export function labelClass(label: string): string {
  for (const ch of label) {
    if ((ch.codePointAt(0) ?? 0) > 127) return 'text-label-ja';
  }
  return 'text-label';
}
