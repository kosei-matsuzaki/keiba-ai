import { memo } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { formatSignedYen, formatYen } from '@/lib/formatters';
import type { ProfitPoint } from '@/types/api';

/**
 * 損益推移。**0 から始まる累計損益**を描く。
 *
 * 以前は資産推移（初期資産からの残高）を描いていたが、シミュレーションは
 * 資金運用の再現ではなく「この買い方を続けたらプラスかマイナスか」を見るもの。
 * 元手の額を決めさせると、その額の大小で見た目が変わるだけで情報が増えない。
 * 0 の基準線をまたぐかどうかがそのまま答えになる。
 */
interface ProfitChartProps {
  points: ProfitPoint[];
}

interface TooltipPayloadItem {
  payload: ProfitPoint;
}

interface TooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: string;
}

function _CustomTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const p = payload[0].payload;
  const daily = p.payout - p.invested;
  return (
    <div className="rounded-sm border border-border bg-card px-3 py-2 font-mono text-2xs">
      <div className="font-medium">{label}</div>
      <div className="mt-1 space-y-0.5 text-muted-foreground">
        <div>
          累計{' '}
          <span
            className={`font-medium ${p.profit >= 0 ? 'text-success' : 'text-destructive'}`}
          >
            {formatSignedYen(p.profit)}
          </span>
        </div>
        <div>{p.n_bets} bets</div>
        <div>
          投資 {formatYen(p.invested)} → 払戻 {formatYen(p.payout)}
        </div>
        <div className={daily >= 0 ? 'text-success' : 'text-destructive'}>
          当日収支 {formatSignedYen(daily)}
        </div>
      </div>
    </div>
  );
}

function ProfitChartImpl({ points }: ProfitChartProps) {
  if (points.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        bet データがないため損益推移を描画できません
      </div>
    );
  }

  // 0 を必ず含む範囲にする (プラス側だけ / マイナス側だけでも基準線を見せる)
  const values = points.map((p) => p.profit);
  const top = Math.max(0, ...values);
  const bottom = Math.min(0, ...values);
  const pad = Math.max(1000, Math.round((top - bottom) * 0.05));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={points} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
        <defs>
          <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity={0.5} />
            <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity={0.04} />
          </linearGradient>
        </defs>
        <CartesianGrid vertical={false} stroke="hsl(var(--border))" />
        <XAxis
          dataKey="date"
          tick={{
            fontSize: 10,
            fill: 'hsl(var(--subtle-foreground))',
            fontFamily: 'var(--font-mono)',
          }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v: string | number) => String(v).slice(5)} // MM-DD
        />
        <YAxis
          tick={{
            fontSize: 10,
            fill: 'hsl(var(--subtle-foreground))',
            fontFamily: 'var(--font-mono)',
          }}
          axisLine={false}
          tickLine={false}
          domain={[bottom - pad, top + pad]}
          tickFormatter={(v: number) => {
            const abs = Math.abs(v);
            const sign = v < 0 ? '−' : '';
            if (abs >= 1_000_000) return `${sign}${(abs / 1_000_000).toFixed(1)}M`;
            if (abs >= 1_000) return `${sign}${Math.round(abs / 1_000)}k`;
            return String(v);
          }}
        />
        <Tooltip content={<_CustomTooltip />} />
        {/* 損益分岐ライン (= 0)。ここをまたぐかどうかが結論。 */}
        <ReferenceLine
          y={0}
          stroke="hsl(var(--muted-foreground))"
          strokeDasharray="4 4"
          label={{
            value: '±0',
            fontSize: 10,
            fill: 'hsl(var(--muted-foreground))',
            position: 'insideTopRight',
          }}
        />
        <Area
          type="monotone"
          dataKey="profit"
          stroke="hsl(var(--primary))"
          strokeWidth={1.5}
          fill="url(#profitGradient)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export const ProfitChart = memo(ProfitChartImpl);
