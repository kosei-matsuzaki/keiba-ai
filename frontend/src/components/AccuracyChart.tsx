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
        <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 11 }}
          tickFormatter={(v: string | number) => String(v).slice(5)} // MM-DD
        />
        <YAxis tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
        <Tooltip
          labelFormatter={(label: string) => label}
          formatter={(value: number) => [formatScore(value), metricLabel]}
        />
        <Line
          type="monotone"
          dataKey="value"
          stroke="hsl(var(--primary))"
          strokeWidth={2}
          dot={false}
          connectNulls
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export const AccuracyChart = memo(AccuracyChartImpl);
