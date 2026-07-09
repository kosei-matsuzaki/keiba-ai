import { useState, useMemo, Fragment } from 'react';
import { Wallet, ChevronDown, ChevronUp, Download, Trash2 } from 'lucide-react';
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
import { useDeleteBets } from '@/hooks/useDeleteBets';
import { buildBetExportUrl, type BetFilterParams } from '@/lib/api';
import { formatYen, formatPercent, formatDateTime } from '@/lib/formatters';
import { AddBetDialog } from '@/components/AddBetDialog';
import { DateYMDPicker } from '@/components/DateYMDPicker';
import { MetricCard } from '@/components/MetricCard';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { BetBreakdownRow, BetRecordOut } from '@/types/api';

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

// ── Detail table grouped by 買い方 ─────────────────────────────────────────────

const PAGE_SIZE = 20;

/** 1 つの買い方（同一バッチ）= 一括登録時に共有する created_at でまとめた点群。 */
interface BetGroup {
  key: string;
  ids: number[];
  createdAt: string;
  raceId: string;
  betType: string;
  source: string;
  notes: string | null;
  count: number;
  stakeSum: number;
  settledCount: number;
  pendingCount: number;
  payoutSum: number; // 確定分のみ
  profitSum: number; // 確定分のみ
  items: BetRecordOut[];
}

/**
 * bet_records を買い方単位でまとめる。一括登録した点はバッチで同一の created_at を
 * 共有するため、(created_at, race_id, bet_type, source, notes) をキーにグループ化する。
 * 単発登録はそれぞれ別グループ（1 点）になる。API 返却は created_at 降順なので順序維持。
 */
function groupBets(items: BetRecordOut[]): BetGroup[] {
  const map = new Map<string, BetGroup>();
  for (const b of items) {
    const key = `${b.created_at}|${b.race_id}|${b.bet_type}|${b.source}|${b.notes ?? ''}`;
    let g = map.get(key);
    if (!g) {
      g = {
        key,
        ids: [],
        createdAt: b.created_at,
        raceId: b.race_id,
        betType: b.bet_type,
        source: b.source,
        notes: b.notes,
        count: 0,
        stakeSum: 0,
        settledCount: 0,
        pendingCount: 0,
        payoutSum: 0,
        profitSum: 0,
        items: [],
      };
      map.set(key, g);
    }
    g.ids.push(b.id);
    g.items.push(b);
    g.count += 1;
    g.stakeSum += b.stake;
    if (b.settled_at !== null) {
      g.settledCount += 1;
      g.payoutSum += b.payout ?? 0;
      g.profitSum += b.profit ?? 0;
    } else {
      g.pendingCount += 1;
    }
  }
  return [...map.values()];
}

function ProfitCell({ group }: { group: BetGroup }) {
  if (group.settledCount === 0) {
    return <span className="text-muted-foreground">未確定</span>;
  }
  const cls = group.profitSum >= 0 ? 'text-green-600' : 'text-red-500';
  return (
    <span className={cls}>
      {group.profitSum >= 0 ? '+' : ''}
      {formatYen(group.profitSum)}
      {group.pendingCount > 0 && (
        <span className="ml-1 text-[10px] text-muted-foreground">未確定{group.pendingCount}</span>
      )}
    </span>
  );
}

function DetailTable({ params }: { params: BetFilterParams }) {
  const [page, setPage] = useState(0);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const { data, isPending, isError } = useBetList(params);
  const deleteBets = useDeleteBets();

  function handleDeleteGroup(g: BetGroup) {
    const msg =
      g.count > 1 ? `この買い方（${g.count}点）を削除しますか？` : 'この購入記録を削除しますか？';
    if (window.confirm(msg)) {
      deleteBets.mutate(g.ids);
    }
  }

  function toggleExpand(key: string) {
    setExpanded((prev) => {
      const s = new Set(prev);
      if (s.has(key)) s.delete(key);
      else s.add(key);
      return s;
    });
  }

  const items = useMemo(() => data?.items ?? [], [data]);
  const groups = useMemo(() => groupBets(items), [items]);
  const pageGroups = groups.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.ceil(groups.length / PAGE_SIZE);

  if (isPending) return <Skeleton className="h-48 w-full" />;
  if (isError) return <EmptyState message="明細取得に失敗しました" />;
  if (items.length === 0) return <EmptyState message="購入記録がありません" />;

  return (
    <div className="space-y-3">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>日時</TableHead>
            <TableHead>レース ID</TableHead>
            <TableHead>券種</TableHead>
            <TableHead>買い目 / 点数</TableHead>
            <TableHead>投資</TableHead>
            <TableHead>払戻</TableHead>
            <TableHead>損益</TableHead>
            <TableHead>メモ</TableHead>
            <TableHead className="text-xs text-muted-foreground">区分</TableHead>
            <TableHead className="text-right">操作</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {pageGroups.map((g) => {
            const isOpen = expanded.has(g.key);
            return (
              <Fragment key={g.key}>
                <TableRow>
                  <TableCell className="text-xs">{formatDateTime(g.createdAt)}</TableCell>
                  <TableCell className="text-xs font-mono">{g.raceId}</TableCell>
                  <TableCell>{g.betType}</TableCell>
                  <TableCell>
                    {g.count === 1 ? (
                      <span className="font-mono">{g.items[0].combo}</span>
                    ) : (
                      <button
                        type="button"
                        className="inline-flex items-center gap-1 text-sm hover:text-foreground"
                        onClick={() => toggleExpand(g.key)}
                      >
                        <span className="font-medium">{g.count}点</span>
                        {isOpen ? (
                          <ChevronUp className="h-3 w-3" />
                        ) : (
                          <ChevronDown className="h-3 w-3" />
                        )}
                      </button>
                    )}
                  </TableCell>
                  <TableCell>{formatYen(g.stakeSum)}</TableCell>
                  <TableCell>{g.settledCount === 0 ? '—' : formatYen(g.payoutSum)}</TableCell>
                  <TableCell>
                    <ProfitCell group={g} />
                  </TableCell>
                  <TableCell
                    className="max-w-[12rem] truncate text-xs text-muted-foreground"
                    title={g.notes ?? ''}
                  >
                    {g.notes ?? ''}
                  </TableCell>
                  <TableCell className="text-[11px] text-muted-foreground">
                    {g.source === 'recommendation' ? 'AI推奨' : '手動'}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 px-2"
                      disabled={deleteBets.isPending}
                      onClick={() => handleDeleteGroup(g)}
                      aria-label="削除"
                    >
                      <Trash2 className="h-4 w-4 text-muted-foreground" />
                    </Button>
                  </TableCell>
                </TableRow>
                {isOpen && (
                  <TableRow>
                    <TableCell colSpan={10} className="bg-muted/30">
                      <div className="flex flex-col gap-1 py-1 pl-4">
                        {g.items.map((it) => (
                          <div key={it.id} className="flex items-center gap-4 text-xs">
                            <span className="w-28 font-mono">{it.combo}</span>
                            <span className="w-20 text-muted-foreground">
                              {formatYen(it.stake)}
                            </span>
                            <span className="w-20">
                              {it.payout !== null ? formatYen(it.payout) : '—'}
                            </span>
                            <span
                              className={
                                it.profit === null
                                  ? 'text-muted-foreground'
                                  : it.profit >= 0
                                    ? 'text-green-600'
                                    : 'text-red-500'
                              }
                            >
                              {it.profit !== null
                                ? `${it.profit >= 0 ? '+' : ''}${formatYen(it.profit)}`
                                : '未確定'}
                            </span>
                          </div>
                        ))}
                      </div>
                    </TableCell>
                  </TableRow>
                )}
              </Fragment>
            );
          })}
        </TableBody>
      </Table>
      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>
          {groups.length} 件（買い方） / {items.length} 点
        </span>
        {totalPages > 1 && (
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
        )}
      </div>
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
  // 全プラスなら success (emerald)、全マイナスなら destructive (rose)、混在は chart-1 (indigo)
  const areaColor =
    minProfit >= 0
      ? 'hsl(var(--success))'
      : maxProfit <= 0
      ? 'hsl(var(--destructive))'
      : 'hsl(var(--chart-1))';

  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data.points} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
        <defs>
          <linearGradient id="profitGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={areaColor} stopOpacity={0.4} />
            <stop offset="95%" stopColor={areaColor} stopOpacity={0.0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="2 4" stroke="hsl(var(--border))" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
          stroke="hsl(var(--border))"
          tickFormatter={(v: string) => v.slice(5)}
        />
        <YAxis
          tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
          stroke="hsl(var(--border))"
          tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`}
        />
        <Tooltip
          labelFormatter={(label: string) => label}
          formatter={(value: number) => [formatYen(value), '累計損益']}
          contentStyle={{
            background: 'hsl(var(--popover))',
            border: '1px solid hsl(var(--border))',
            borderRadius: 'var(--radius)',
            fontSize: 12,
          }}
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
  const [showDetail, setShowDetail] = useState(true);
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
        title="収支台帳"
        description="自分の購入履歴と成績管理（回収率・的中率・損益推移）"
      >
        <AddBetDialog />
      </PageHeader>

      {/* Period & source filters + CSV (一行で揃える) */}
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
        <Button
          variant="outline"
          size="sm"
          onClick={handleCsvDownload}
          className="ml-auto"
        >
          <Download className="h-4 w-4" />
          CSV
        </Button>
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
            <CardTitle className="text-base">購入明細</CardTitle>
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
    </div>
  );
}
