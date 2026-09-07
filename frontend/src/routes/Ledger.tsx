import { useState, useMemo, Fragment } from 'react';
import { ChevronDown, ChevronUp, Download, Trash2 } from 'lucide-react';
import { useBetSummary } from '@/hooks/useBetSummary';
import { useBetTimeseries } from '@/hooks/useBetTimeseries';
import { useBetBreakdown } from '@/hooks/useBetBreakdown';
import { useBetList } from '@/hooks/useBetList';
import { useDeleteBets } from '@/hooks/useDeleteBets';
import { buildBetExportUrl, type BetFilterParams } from '@/lib/api';
import {
  formatDateTime,
  formatPercent,
  formatRatio,
  formatSignedYen,
  formatYen,
} from '@/lib/formatters';
import { SectionHeading } from '@/components/SectionHeading';
import { AddBetDialog } from '@/components/AddBetDialog';
import { DateYMDPicker } from '@/components/DateYMDPicker';
import { ProfitChart } from '@/components/ProfitChart';
import { MetricCard } from '@/components/MetricCard';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
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

/**
 * 期間は「全期間」と「カスタム」だけ。
 *
 * **日数の窓を置かない。**JRA は土日開催なので 7 日 = 1〜3 開催日にしかならず、
 * 画面の主役である回収率がそこでは読めない。実測 (2025-09〜2026-02・7,006 点) で
 * 直近 7 日は 1.073、全期間は 0.835 — **短い窓だけが「黒字」に見える**。
 * 0.05 の差を言うのに複勝で 1.6 年ぶんが要る量なので、日数の窓は嘘をつく側にしか
 * 働かない (`docs/ai-model.md`「OOS 実測」)。
 *
 * 先週末だけ見たいときはカスタムで日付を入れる。**そこでも率は率なので、
 * 読むのは純利益・投資・払戻のほう。**
 */
type PeriodPreset = 'all' | 'custom';

function getDateRange(preset: PeriodPreset, customFrom: string, customTo: string) {
  if (preset === 'custom') {
    return {
      from: customFrom || undefined,
      to: customTo || undefined,
    };
  }
  return { from: undefined, to: undefined };
}

type BreakdownGroup = 'bet_type' | 'race_class' | 'month';

/**
 * 内訳の切り口。**`source` は置かない** — 上の「全件 / AI推奨 / 手動」フィルタが
 * 同じ軸で、2 か所から同じことを操作できると、どちらが効いているのか読めなくなる。
 */
const BREAKDOWN_TABS: { value: BreakdownGroup; label: string }[] = [
  { value: 'bet_type', label: '馬券種別' },
  { value: 'race_class', label: 'レース格別' },
  { value: 'month', label: '月別' },
];

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

  // 数値列は右揃え。オッズ・金額・率が縦に揃うことで読み比べやすくなる。
  const th = (col: SortKey, label: string, align: 'left' | 'right' = 'right') => (
    <TableHead
      className={`cursor-pointer select-none ${align === 'right' ? 'text-right' : ''}`}
      onClick={() => handleSort(col)}
    >
      {label} <SortIcon col={col} />
    </TableHead>
  );

  if (sorted.length === 0) {
    return (
      <EmptyState
        message="まだ確定した購入がありません"
        description="購入を記録して結果が確定すると、券種ごとの回収率がここに出ます。"
      />
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          {th('group_key', '券種', 'left')}
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
            <TableCell className="text-right">{row.bets}</TableCell>
            <TableCell className="text-right">{formatYen(row.invested)}</TableCell>
            <TableCell className="text-right">{formatYen(row.payout)}</TableCell>
            <TableCell
              className={`text-right ${row.profit >= 0 ? 'text-success' : 'text-destructive'}`}
            >
              {row.profit >= 0 ? '+' : ''}{formatYen(row.profit)}
            </TableCell>
            <TableCell className="text-right">{formatPercent(row.payback_rate)}</TableCell>
            <TableCell className="text-right">{formatPercent(row.hit_rate)}</TableCell>
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
  const cls = group.profitSum >= 0 ? 'text-success' : 'text-destructive';
  return (
    <span className={cls}>
      {group.profitSum >= 0 ? '+' : ''}
      {formatYen(group.profitSum)}
      {group.pendingCount > 0 && (
        <span className="ml-1 text-2xs text-muted-foreground">未確定{group.pendingCount}</span>
      )}
    </span>
  );
}

/**
 * 購入記録の内訳表。行を開くと同じ買い方の点をまとめて見せる。
 */
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
  if (items.length === 0)
    return (
      <EmptyState
        message="購入記録がありません"
        description="「購入を記録」から、またはレース画面の推奨買目からまとめて記録できます。"
      />
    );

  return (
    <div className="space-y-3">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>日時</TableHead>
            <TableHead>レース ID</TableHead>
            <TableHead>券種</TableHead>
            <TableHead>買い目 / 点数</TableHead>
            <TableHead className="text-right">投資</TableHead>
            <TableHead className="text-right">払戻</TableHead>
            <TableHead className="text-right">損益</TableHead>
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
                  <TableCell className="text-right">{formatYen(g.stakeSum)}</TableCell>
                  <TableCell className="text-right">
                    {g.settledCount === 0 ? '—' : formatYen(g.payoutSum)}
                  </TableCell>
                  <TableCell className="text-right">
                    <ProfitCell group={g} />
                  </TableCell>
                  <TableCell
                    className="max-w-[12rem] truncate text-xs text-muted-foreground"
                    title={g.notes ?? ''}
                  >
                    {g.notes ?? ''}
                  </TableCell>
                  <TableCell className="text-2xs text-muted-foreground">
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
                                    ? 'text-success'
                                    : 'text-destructive'
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

/**
 * 累計損益推移。**描くのは共通の `ProfitChart`** なので、シミュレーションの結果と
 * 同じ見た目・同じ規則になる (買っていない日を落とす / 符号で色が変わる /
 * 0 の基準線を必ず含む)。ここは API の形を合わせるだけ。
 */
function LedgerProfitChart({
  params,
}: {
  params: BetFilterParams & { bucket?: 'day' | 'week' | 'month' };
}) {
  const { data, isPending, isError } = useBetTimeseries(params);

  if (isPending) return <Skeleton className="h-60 w-full" />;
  if (isError) return <EmptyState message="チャートデータ取得に失敗しました" />;

  const points = (data?.points ?? []).map((p) => ({
    date: p.date,
    profit: p.cumulative_profit,
    invested: p.invested,
    payout: p.payout,
    n_bets: p.bets,
  }));

  return (
    <ProfitChart points={points} emptyMessage="購入を記録すると、ここに累計損益が出ます" />
  );
}

// ── Main Ledger page ──────────────────────────────────────────────────────────

export function Ledger() {
  const [period, setPeriod] = useState<PeriodPreset>('all');
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
  const [groupBy, setGroupBy] = useState<BreakdownGroup>('bet_type');
  const breakdownQuery = useBetBreakdown({ ...filterParams, group_by: groupBy });

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
    <div className="flex flex-col gap-8 p-6">
      <PageHeader
        eyebrow="Ledger"
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
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full rounded-sm" />
          ))}
        </div>
      ) : summaryQuery.isError ? (
        <EmptyState message="サマリ取得に失敗しました" />
      ) : (
        /* 収支は 5 つを並べる。**1 つを答えに選べない**からで、投資と払戻は対、
           回収率と的中率も対で読む (的中率だけ高くても配当が小さければ負ける)。
           シミュレーションの結果・モデル詳細と同じ並び・同じ言い回しにしてある。 */
        <div className={"grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5"}>
          <MetricCard
            label="累計投資"
            value={formatYen(summaryQuery.data.total_invested)}
            note={`${summaryQuery.data.total_bets.toLocaleString()} 点`}
          />
          <MetricCard
            label="累計払戻"
            value={formatYen(summaryQuery.data.total_payout)}
            note={`確定 ${summaryQuery.data.settled_bets.toLocaleString()} 点`}
          />
          <MetricCard
            label="純利益"
            value={formatSignedYen(summaryQuery.data.total_profit)}
            tone={summaryQuery.data.total_profit >= 0 ? 'positive' : 'negative'}
            note="0 から始めた場合の収支"
          />
          <MetricCard
            label="回収率"
            value={formatRatio(summaryQuery.data.payback_rate)}
            tone={summaryQuery.data.payback_rate >= 1 ? 'positive' : 'negative'}
            note="1.00 = 損益分岐点"
            hint="払戻 ÷ 投資。控除率 20% があるので 1.0 未満は平均で負け越し。"
          />
          <MetricCard
            label="的中率"
            value={formatPercent(summaryQuery.data.hit_rate)}
            note="確定済みのうち"
            hint="払戻が出た買い目の割合。的中率が高いほど儲かるとは限らない — 人気馬を選べば当たるが配当が小さい。"
          />
        </div>
      )}

      {/* Cumulative profit chart */}
      <Card className="border-t border-border pt-6">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <SectionHeading>累計損益推移</SectionHeading>
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
          <LedgerProfitChart params={{ ...filterParams, bucket }} />
        </CardContent>
      </Card>

      {/* 内訳。**切り口をタブで切り替える** — 同じ形の表を 4 つ積むと縦に伸びる
          だけで見比べられない (シミュレーションの結果と同じ作り)。 */}
      <Card className="border-t border-border pt-6">
        <CardHeader className="pb-2">
          <SectionHeading>内訳</SectionHeading>
        </CardHeader>
        <CardContent>
          <Tabs value={groupBy} onValueChange={(v) => setGroupBy(v as BreakdownGroup)}>
            <TabsList>
              {BREAKDOWN_TABS.map(({ value, label }) => (
                <TabsTrigger key={value} value={value}>
                  {label}
                </TabsTrigger>
              ))}
            </TabsList>
            <div className="pt-3">
              {breakdownQuery.isPending ? (
                <Skeleton className="h-40 w-full" />
              ) : breakdownQuery.isError ? (
                <EmptyState message="内訳の取得に失敗しました" />
              ) : breakdownQuery.data.rows.length === 0 ? (
                <EmptyState
                  message="まだ確定した購入がありません"
                  description="購入を記録して結果が確定すると、切り口ごとの回収率がここに出ます。"
                />
              ) : (
                <BreakdownTable rows={breakdownQuery.data.rows} />
              )}
            </div>
          </Tabs>
        </CardContent>
      </Card>

      {/* Detail table (collapsible) */}
      <Card className="border-t border-border pt-6">
        <CardHeader className="pb-2">
          <button
            type="button"
            className="flex w-full items-center justify-between"
            onClick={() => setShowDetail((v) => !v)}
          >
            <SectionHeading>購入明細</SectionHeading>
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
