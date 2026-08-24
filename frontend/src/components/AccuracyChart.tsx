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
  // 点が 1 個だと広い軸の中に点がひとつ浮くだけで「推移」を読み取れないので、
  // 2 点未満はグラフを描かずに空状態にする。高さはグラフと同じ h-60 に固定して
  // カードの高さが跳ねないようにする。
  if (points.length < 2) {
    return (
      <div className="flex h-60 flex-col items-center justify-center gap-1 text-center text-sm text-muted-foreground">
        <p>推移を出すには評価が 2 回以上必要です</p>
        <p className="text-xs text-subtle-foreground">現在 {points.length} 回</p>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={points} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
        {/* 横線だけ。縦グリッドと軸線は情報を足さないので消す */}
        <CartesianGrid vertical={false} stroke="hsl(var(--border))" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 10, fill: 'hsl(var(--subtle-foreground))', fontFamily: 'var(--font-mono)' }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v: string | number) => String(v).slice(5)} // MM-DD
        />
        <YAxis
          tick={{ fontSize: 10, fill: 'hsl(var(--subtle-foreground))', fontFamily: 'var(--font-mono)' }}
          axisLine={false}
          tickLine={false}
          domain={['auto', 'auto']}
        />
        <Tooltip
          labelFormatter={(label: string) => label}
          formatter={(value: number) => [formatScore(value), metricLabel]}
          contentStyle={{
            background: 'hsl(var(--card))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '2px',
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
          }}
        />
        <Line
          type="monotone"
          dataKey="value"
          stroke="hsl(var(--primary))"
          strokeWidth={1.5}
          dot={false}
          connectNulls
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export const AccuracyChart = memo(AccuracyChartImpl);
