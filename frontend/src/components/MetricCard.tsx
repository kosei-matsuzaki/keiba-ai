import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';
import {
  METRIC_ACCENT_CLASS,
  METRIC_CARD_CLASS,
  METRIC_VALUE_CLASS,
  type MetricTone,
} from '@/lib/metricStyles';

/**
 * 指標 1 つ分のカード。
 *
 * **数字が主役**なので、飾りは 3 つに絞る:
 *   - ラベルは小さく淡く（読むのは値のあと）
 *   - 値は等幅 + tabular-nums で大きく。桁が揃うことで「計測した値」に見える
 *   - 損益のように向きがある値だけ、左の細い帯に色を出す
 *
 * 表やグラフは囲わない（それ自体でまとまって見えるので枠が二重になる）。
 * 囲うのは指標と注意喚起だけ。
 */

interface MetricCardProps {
  label: string;
  /** 整形済みの値。未算出は '未算出' のような文字列で渡す。 */
  value: string;
  /** 値の下の 1 行。基準（1.00 = トントン）や出所を書く。 */
  note?: ReactNode;
  tone?: MetricTone;
  /** マウスオーバーで出す補足。読まなくても困らないことはこちらへ。 */
  hint?: string;
  className?: string;
}

export function MetricCard({
  label,
  value,
  note,
  tone = 'default',
  hint,
  className,
}: MetricCardProps) {
  return (
    <div
      title={hint}
      className={cn(METRIC_CARD_CLASS, METRIC_ACCENT_CLASS[tone], className)}
    >
      <span className="text-xs text-muted-foreground">
        {label}
        {hint && <span className="ml-1 text-subtle-foreground/60">?</span>}
      </span>
      <span
        className={cn(
          'text-kpi',
          METRIC_VALUE_CLASS[tone]
        )}
      >
        {value}
      </span>
      {note && (
        <span className="text-2xs leading-snug text-subtle-foreground">{note}</span>
      )}
    </div>
  );
}
