/**
 * 指標カードの見た目を 1 箇所に置く。
 *
 * `MetricCard` と `MetricBand` の両方が使うので、コンポーネントのファイルから
 * 出すと Fast Refresh が効かなくなる (コンポーネント以外を export するため)。
 *
 * **数字が主役**なので飾りは 3 つに絞る:
 *   - ラベルは小さく淡く（読むのは値のあと）
 *   - 値は等幅 + tabular-nums で大きく。桁が揃うことで「計測した値」に見える
 *   - 損益のように向きがある値だけ、左の細い帯に色を出す
 */
export type MetricTone = 'default' | 'positive' | 'negative' | 'muted';

export const METRIC_CARD_CLASS =
  'relative flex min-w-[9.5rem] flex-col gap-1 overflow-hidden rounded-sm ' +
  'border border-border bg-card px-4 py-3 ' +
  'before:absolute before:inset-y-0 before:left-0 before:w-0.5 before:content-[""]';

/** 左の帯の色。向きのある値 (損益・回収率) だけ色を持たせる。 */
export const METRIC_ACCENT_CLASS: Record<MetricTone, string> = {
  default: 'before:bg-border',
  positive: 'before:bg-success',
  negative: 'before:bg-destructive',
  muted: 'before:bg-border',
};

/** 値の色。 */
export const METRIC_VALUE_CLASS: Record<MetricTone, string> = {
  default: 'text-foreground',
  positive: 'text-success',
  negative: 'text-destructive',
  muted: 'text-muted-foreground',
};
