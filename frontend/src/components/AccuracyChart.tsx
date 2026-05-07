import { memo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { formatScore } from '@/lib/formatters';
import type { TimeseriesPoint } from '@/types/api';

interface AccuracyChartProps {
  points: TimeseriesPoint[];
  metricLabel: string;
}

// memo で wrap: parent (Dashboard) でポーリング由来の re-render が起きても、
// points / metricLabel が同一参照なら recharts の SVG 再構築をスキップ。
function AccuracyChartImpl({ points, metricLabel }: AccuracyChartProps) {
  if (points.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center text-sm text-muted-foreground">
        データがありません
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={points} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="2 4" stroke="hsl(var(--border))" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
          stroke="hsl(var(--border))"
          tickFormatter={(v: string | number) => String(v).slice(5)} // MM-DD
        />
        <YAxis
          tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
          stroke="hsl(var(--border))"
          domain={['auto', 'auto']}
        />
        <Tooltip
          labelFormatter={(label: string) => label}
          formatter={(value: number) => [formatScore(value), metricLabel]}
          contentStyle={{
            background: 'hsl(var(--popover))',
            border: '1px solid hsl(var(--border))',
            borderRadius: 'var(--radius)',
            fontSize: 12,
          }}
        />
        <Line
          type="monotone"
          dataKey="value"
          stroke="hsl(var(--chart-1))"
          strokeWidth={2}
          dot={false}
          connectNulls
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export const AccuracyChart = memo(AccuracyChartImpl);
