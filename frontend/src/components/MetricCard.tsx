import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { formatPercent, formatRatio, formatScore } from '@/lib/formatters';

interface MetricCardProps {
  title: string;
  value: number | null;
  format?: 'percent' | 'decimal' | 'ratio';
  description?: string;
}

function formatValue(value: number | null, format: MetricCardProps['format']): string {
  switch (format) {
    case 'percent':
      return formatPercent(value);
    case 'ratio':
      return formatRatio(value);
    case 'decimal':
    default:
      return formatScore(value);
  }
}

export function MetricCard({ title, value, format = 'decimal', description }: MetricCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-3xl font-bold">{formatValue(value, format)}</p>
        {description && <p className="mt-1 text-xs text-muted-foreground">{description}</p>}
      </CardContent>
    </Card>
  );
}
