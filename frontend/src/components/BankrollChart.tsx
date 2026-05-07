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

import { formatYen } from '@/lib/formatters';
import type { BankrollPoint } from '@/types/api';

interface BankrollChartProps {
  points: BankrollPoint[];
  /** 初期資産。reference line を引いて損益分岐ラインを示す。 */
  initialBudget: number;
}

interface TooltipPayloadItem {
  payload: BankrollPoint;
}

interface TooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: string;
}

function _CustomTooltip({ active, payload, label }: TooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const p = payload[0].payload;
  const profit = p.payout - p.invested;
  return (
    <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
      <div className="font-medium">{label}</div>
      <div className="mt-1 space-y-0.5 text-muted-foreground">
        <div>
          資産: <span className="font-medium text-foreground">{formatYen(p.bankroll)}</span>
        </div>
        <div>{p.n_bets} bets</div>
        <div>
          投資 {formatYen(p.invested)} → 払戻 {formatYen(p.payout)}
        </div>
        <div className={profit >= 0 ? 'text-green-600' : 'text-red-600'}>
          当日収支 {profit >= 0 ? '+' : '−'}
          {formatYen(Math.abs(profit))}
        </div>
      </div>
    </div>
  );
}

function BankrollChartImpl({ points, initialBudget }: BankrollChartProps) {
  if (points.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        bet データがないため資産推移を描画できません
      </div>
    );
  }

  // domain を初期予算と推移を含む範囲に。底は 0 起点 (破産が見やすい)。
  const maxBankroll = Math.max(initialBudget, ...points.map((p) => p.bankroll));
  const yDomainTop = Math.ceil((maxBankroll * 1.05) / 1000) * 1000;

  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={points} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
        <defs>
          <linearGradient id="bankrollGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity={0.4} />
            <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 11 }}
          tickFormatter={(v: string | number) => String(v).slice(5)} // MM-DD
        />
        <YAxis
          tick={{ fontSize: 11 }}
          domain={[0, yDomainTop]}
          tickFormatter={(v: number) => {
            if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
            if (v >= 1_000) return `${Math.round(v / 1_000)}k`;
            return String(v);
          }}
        />
        <Tooltip content={<_CustomTooltip />} />
        {/* 損益分岐ライン (= 初期資産) */}
        <ReferenceLine
          y={initialBudget}
          stroke="hsl(var(--muted-foreground))"
          strokeDasharray="4 4"
          label={{
            value: `初期 ${formatYen(initialBudget)}`,
            fontSize: 10,
            fill: 'hsl(var(--muted-foreground))',
            position: 'insideTopRight',
          }}
        />
        <Area
          type="monotone"
          dataKey="bankroll"
          stroke="hsl(var(--primary))"
          strokeWidth={2}
          fill="url(#bankrollGradient)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export const BankrollChart = memo(BankrollChartImpl);
