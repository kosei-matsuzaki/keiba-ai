import type { ReactNode } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/cn';
import { formatPercent, formatRatio, formatScore, formatYen } from '@/lib/formatters';

interface MetricCardProps {
  title: string;
  value: number | null;
  format?: 'percent' | 'decimal' | 'ratio' | 'yen';
  description?: string;
  /** description の色を上書きするヒント (損益のプラスマイナスなど)。 */
  tone?: 'default' | 'positive' | 'negative' | 'muted';
  /** タイトル右に小さく置くアイコンや trend chip 等。 */
  trailing?: ReactNode;
}

function formatValue(value: number | null, format: MetricCardProps['format']): string {
  switch (format) {
    case 'percent':
      return formatPercent(value);
    case 'ratio':
      return formatRatio(value);
    case 'yen':
      return formatYen(value);
    case 'decimal':
    default:
      return formatScore(value);
  }
}

const _TONE_CLASS: Record<NonNullable<MetricCardProps['tone']>, string> = {
  default: 'text-muted-foreground',
  positive: 'text-success',
  negative: 'text-destructive',
  muted: 'text-subtle-foreground',
};

export function MetricCard({
  title,
  value,
  format = 'decimal',
  description,
  tone = 'default',
  trailing,
}: MetricCardProps) {
  return (
    <Card className="p-5">
      <CardContent className="!p-0 flex flex-col gap-3">
        <div className="flex items-start justify-between gap-2">
          <span className="text-label">{title}</span>
          {trailing && <div className="shrink-0">{trailing}</div>}
        </div>
        <p className="text-kpi text-foreground">
          {formatValue(value, format)}
        </p>
        {description && (
          <p className={cn('text-xs', _TONE_CLASS[tone])}>{description}</p>
        )}
      </CardContent>
    </Card>
  );
}
