import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface MetricCardProps {
  title: string;
  value: number | null;
  format?: 'percent' | 'decimal' | 'ratio';
  description?: string;
}

function formatValue(value: number | null, format: MetricCardProps['format']): string {
  if (value === null) return '—';
  switch (format) {
    case 'percent':
      return `${(value * 100).toFixed(1)}%`;
    case 'ratio':
      return value.toFixed(2);
    case 'decimal':
    default:
      return value.toFixed(3);
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
