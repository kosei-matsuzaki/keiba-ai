import { useState, useMemo } from 'react';
import { Wallet, ChevronDown, ChevronUp, Download } from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';

import { useBetSummary } from '@/hooks/useBetSummary';
import { useBetTimeseries } from '@/hooks/useBetTimeseries';
import { useBetBreakdown } from '@/hooks/useBetBreakdown';
import { useBetList } from '@/hooks/useBetList';
import { buildBetExportUrl, type BetFilterParams } from '@/lib/api';
import { formatYen, formatPercent, formatDateTime } from '@/lib/formatters';
import { DateYMDPicker } from '@/components/DateYMDPicker';
import { MetricCard } from '@/components/MetricCard';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { SimulationTab } from '@/components/SimulationTab';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { BetBreakdownRow } from '@/types/api';

// ── Period presets ────────────────────────────────────────────────────────────

type PeriodPreset = '7d' | '30d' | 'all' | 'custom';

function getDateRange(preset: PeriodPreset, customFrom: string, customTo: string) {
  if (preset === 'all') return { from: undefined, to: undefined };
  if (preset === 'custom') {
    return {
      from: customFrom || undefined,
      to: customTo || undefined,
    };
  }
  const today = new Date();
  const days = preset === '7d' ? 7 : 30;
  const from = new Date(today);
  from.setDate(today.getDate() - days);
  return {
    from: from.toISOString().slice(0, 10),
    to: today.toISOString().slice(0, 10),
  };
}

// ── Breakdown table with sort ─────────────────────────────────────────────────

type SortKey = keyof BetBreakdownRow;

function BreakdownTable({ rows }: { rows: BetBreakdownRow[] }) {
  const [sortKey, setSortKey] = useState<SortKey>('payback_rate');
  const [sortAsc, setSortAsc] = useState(false);

  const sorted = useMemo(() => {
    return [...rows].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      const cmp = typeof av === 'string' ? av.localeCompare(bv as string) : (av as number) - (bv as number);
      return sortAsc ? cmp : -cmp;
    });
  }, [rows, sortKey, sortAsc]);

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortAsc((prev) => !prev);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  }

  function SortIcon({ col }: { col: SortKey }) {
    if (col !== sortKey) return null;
    return sortAsc ? <ChevronUp className="inline h-3 w-3" /> : <ChevronDown className="inline h-3 w-3" />;
  }

  const th = (col: SortKey, label: string) => (
    <TableHead
      className="cursor-pointer select-none"
      onClick={() => handleSort(col)}
    >
      {label} <SortIcon col={col} />
    </TableHead>
  );

  if (sorted.length === 0) {
    return <EmptyState message="データがありません" />;
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          {th('group_key', '券種')}
          {th('bets', '件数')}
          {th('invested', '投資額')}
          {th('payout', '払戻額')}
          {th('profit', '損益')}
          {th('payback_rate', '回収率')}
          {th('hit_rate', '的中率')}
        </TableRow>
      </TableHeader>
      <TableBody>
        {sorted.map((row) => (
          <TableRow key={row.group_key}>
            <TableCell className="font-medium">{row.group_key}</TableCell>
            <TableCell>{row.bets}</TableCell>
            <TableCell>{formatYen(row.invested)}</TableCell>
            <TableCell>{formatYen(row.payout)}</TableCell>
            <TableCell className={row.profit >= 0 ? 'text-green-600' : 'text-red-500'}>
              {row.profit >= 0 ? '+' : ''}{formatYen(row.profit)}
            </TableCell>
            <TableCell>{formatPercent(row.payback_rate)}</TableCell>
            <TableCell>{formatPercent(row.hit_rate)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// ── Detail table with pagination ──────────────────────────────────────────────

const PAGE_SIZE = 20;

function DetailTable({ params }: { params: BetFilterParams }) {
  const [page, setPage] = useState(0);
  const { data, isPending, isError } = useBetList(params);

  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const pageItems = items.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.ceil(items.length / PAGE_SIZE);

  if (isPending) return <Skeleton className="h-48 w-full" />;
  if (isError) return <EmptyState message="明細取得に失敗しました" />;
  if (items.length === 0) return <EmptyState message="ベット記録がありません" />;

  return (
    <div className="space-y-3">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>日時</TableHead>
            <TableHead>レース ID</TableHead>
            <TableHead>券種</TableHead>
            <TableHead>組合せ</TableHead>
            <TableHead>投資</TableHead>
            <TableHead>払戻</TableHead>
            <TableHead>損益</TableHead>
            <TableHead>ソース</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {pageItems.map((bet) => (
            <TableRow key={bet.id}>
              <TableCell className="text-xs">{formatDateTime(bet.created_at)}</TableCell>
              <TableCell className="text-xs font-mono">{bet.race_id}</TableCell>
              <TableCell>{bet.bet_type}</TableCell>
              <TableCell className="font-mono">{bet.combo}</TableCell>
              <TableCell>{formatYen(bet.stake)}</TableCell>
              <TableCell>
                {bet.payout !== null ? formatYen(bet.payout) : '—'}
              </TableCell>
              <TableCell
                className={
                  bet.profit === null
                    ? 'text-muted-foreground'
                    : bet.profit >= 0
                    ? 'text-green-600'
                    : 'text-red-500'
                }
              >
                {bet.profit !== null
                  ? `${bet.profit >= 0 ? '+' : ''}${formatYen(bet.profit)}`
                  : '未確定'}
              </TableCell>
              <TableCell className="text-xs">
                {bet.source === 'recommendation' ? '推奨' : '手動'}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {totalPages > 1 && (
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>
            {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, total)} / {total} 件
          </span>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
            >
              前へ
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= totalPages - 1}
              onClick={() => setPage((p) => p + 1)}
            >
              次へ
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Cumulative profit chart ───────────────────────────────────────────────────

function ProfitChart({ params }: { params: BetFilterParams & { bucket?: 'day' | 'week' | 'month' } }) {
  const { data, isPending, isError } = useBetTimeseries(params);

  if (isPending) return <Skeleton className="h-60 w-full" />;
  if (isError) return <EmptyState message="チャートデータ取得に失敗しました" />;
  if (!data || data.points.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center text-sm text-muted-foreground">
        データがありません
      </div>
    );
  }

  const minProfit = Math.min(...data.points.map((p) => p.cumulative_profit));
  const maxProfit = Math.max(...data.points.map((p) => p.cumulative_profit));
  // Use green when all cumulative values are positive, red when all negative, gradient otherwise
  const areaColor = minProfit >= 0 ? '#22c55e' : maxProfit <= 0 ? '#ef4444' : 'hsl(var(--primary))';

  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data.points} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
        <defs>
          <linearGradient id="profitGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={areaColor} stopOpacity={0.3} />
            <stop offset="95%" stopColor={areaColor} stopOpacity={0.0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 11 }}
          tickFormatter={(v: string) => v.slice(5)}
        />
        <YAxis
          tick={{ fontSize: 11 }}
          tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`}
        />
        <Tooltip
          labelFormatter={(label: string) => label}
          formatter={(value: number) => [formatYen(value), '累計損益']}
        />
        <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" />
        <Area
          type="monotone"
          dataKey="cumulative_profit"
          stroke={areaColor}
          strokeWidth={2}
          fill="url(#profitGrad)"
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ── Main Ledger page ──────────────────────────────────────────────────────────

export function Ledger() {
  const [period, setPeriod] = useState<PeriodPreset>('30d');
  const [customFrom, setCustomFrom] = useState('');
  const [customTo, setCustomTo] = useState('');
  const [source, setSource] = useState<string>('all');
  const [showDetail, setShowDetail] = useState(false);
  const [bucket, setBucket] = useState<'day' | 'week' | 'month'>('day');

  const { from, to } = getDateRange(period, customFrom, customTo);
  const sourceFilter = source === 'all' ? undefined : source;

  const filterParams: BetFilterParams = {
    ...(from ? { from } : {}),
    ...(to ? { to } : {}),
    ...(sourceFilter ? { source: sourceFilter } : {}),
  };

  const summaryQuery = useBetSummary(filterParams);
  const breakdownQuery = useBetBreakdown({ ...filterParams, group_by: 'bet_type' });

  async function handleCsvDownload() {
    const url = await buildBetExportUrl(filterParams);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'bet_records.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      <PageHeader
        icon={Wallet}
        title="Ledger"
        description="ベット記録の集計・損益推移・モデルシミュレーション"
      />

      <Tabs defaultValue="actual" className="flex flex-col gap-6">
        <TabsList className="self-start">
          <TabsTrigger value="actual">実績</TabsTrigger>
          <TabsTrigger value="simulation">シミュレーション</TabsTrigger>
        </TabsList>

        <TabsContent value="simulation" className="mt-0">
          <SimulationTab />
        </TabsContent>

        <TabsContent value="actual" className="mt-0 flex flex-col gap-6">
      <div className="flex justify-end">
        <Button variant="outline" size="sm" onClick={handleCsvDownload}>
          <Download className="h-4 w-4" />
          CSV
        </Button>
      </div>

      {/* Period & source filters */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1">
          {(
            [
              { label: '直近 7 日', value: '7d' },
              { label: '直近 30 日', value: '30d' },
              { label: '全期間', value: 'all' },
              { label: 'カスタム', value: 'custom' },
            ] as const
          ).map(({ label, value }) => (
            <Button
              key={value}
              variant={period === value ? 'default' : 'outline'}
              size="sm"
              onClick={() => setPeriod(value)}
            >
              {label}
            </Button>
          ))}
        </div>
        {period === 'custom' && (
          <div className="flex items-center gap-2">
            <DateYMDPicker
              value={customFrom}
              onChange={setCustomFrom}
              ariaLabel="開始日"
            />
            <span className="text-muted-foreground">〜</span>
            <DateYMDPicker
              value={customTo}
              onChange={setCustomTo}
              ariaLabel="終了日"
            />
          </div>
        )}
        <Select value={source} onValueChange={setSource}>
          <SelectTrigger className="w-36">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">全件</SelectItem>
            <SelectItem value="recommendation">推奨のみ</SelectItem>
            <SelectItem value="manual">手動のみ</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* KPI cards */}
      {summaryQuery.isPending ? (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-lg" />
          ))}
        </div>
      ) : summaryQuery.isError ? (
        <EmptyState message="サマリ取得に失敗しました" />
      ) : (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
          <MetricCard
            title="累計投資"
            value={summaryQuery.data.total_invested}
            format="yen"
            description={`${summaryQuery.data.total_bets} 件`}
          />
          <MetricCard
            title="累計払戻"
            value={summaryQuery.data.total_payout}
            format="yen"
            description={`確定 ${summaryQuery.data.settled_bets} 件`}
          />
          <MetricCard
            title="純利益"
            value={summaryQuery.data.total_profit}
            format="yen"
            description={summaryQuery.data.total_profit >= 0 ? 'プラス収支' : 'マイナス収支'}
          />
          <MetricCard
            title="回収率"
            value={summaryQuery.data.payback_rate}
            format="ratio"
            description="1.00 = 損益分岐点"
          />
          <MetricCard
            title="的中率"
            value={summaryQuery.data.hit_rate}
            format="percent"
            description="確定済み中"
          />
        </div>
      )}

      {/* Cumulative profit chart */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <CardTitle className="text-base">累計損益推移</CardTitle>
          <div className="flex gap-1">
            {(['day', 'week', 'month'] as const).map((b) => (
              <Button
                key={b}
                variant={bucket === b ? 'default' : 'ghost'}
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={() => setBucket(b)}
              >
                {b === 'day' ? '日' : b === 'week' ? '週' : '月'}
              </Button>
            ))}
          </div>
        </CardHeader>
        <CardContent>
          <ProfitChart params={{ ...filterParams, bucket }} />
        </CardContent>
      </Card>

      {/* Breakdown table */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">券種別ブレイクダウン</CardTitle>
        </CardHeader>
        <CardContent>
          {breakdownQuery.isPending ? (
            <Skeleton className="h-40 w-full" />
          ) : breakdownQuery.isError ? (
            <EmptyState message="ブレイクダウン取得に失敗しました" />
          ) : (
            <BreakdownTable rows={breakdownQuery.data.rows} />
          )}
        </CardContent>
      </Card>

      {/* Detail table (collapsible) */}
      <Card>
        <CardHeader className="pb-2">
          <button
            type="button"
            className="flex w-full items-center justify-between"
            onClick={() => setShowDetail((v) => !v)}
          >
            <CardTitle className="text-base">明細一覧</CardTitle>
            {showDetail ? (
              <ChevronUp className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            )}
          </button>
        </CardHeader>
        {showDetail && (
          <CardContent>
            <DetailTable params={filterParams} />
          </CardContent>
        )}
      </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
