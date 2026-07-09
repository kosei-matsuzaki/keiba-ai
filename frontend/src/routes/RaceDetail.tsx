import { useMemo, useState } from 'react';
import { useParams, useSearchParams, Link } from 'react-router-dom';
import { Trophy, ChevronLeft, ChevronUp, ChevronDown, ChevronsUpDown, Sparkles, Download, RefreshCw } from 'lucide-react';

import { useRaceDetail } from '@/hooks/useRaceDetail';
import { usePredictions } from '@/hooks/usePredictions';
import { useRecommendations } from '@/hooks/useRecommendations';
import { useRunShutuba } from '@/hooks/useRunShutuba';
import { RecommendationsCard } from '@/components/RecommendationsCard';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { isNotFoundError, isServiceUnavailableError, formatErrorMessage } from '@/lib/api';
import { formatOdds, formatPercent, formatRatio, formatScore, formatYen } from '@/lib/formatters';
import { toast } from '@/components/ui/toast';
import type { EntrySummary, HorsePrediction } from '@/types/api';

function RaceDetailSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-40 w-full rounded-lg" />
      <Skeleton className="h-64 w-full rounded-lg" />
      <Skeleton className="h-64 w-full rounded-lg" />
    </div>
  );
}

/**
 * 単勝期待値 (EV) = 単勝確率 × 単勝オッズ。
 * 現行モデルは decision-focused（ROI 直接最適化）なので、score より EV が
 * 「買うべきか」の主指標。オッズ未確定 (odds_win=null) のときは null。
 */
function winEv(pred: HorsePrediction | null, entry: EntrySummary | undefined): number | null {
  if (!pred || entry?.odds_win == null) return null;
  return pred.win_prob * entry.odds_win;
}

/** Indicates BUY when single-win expected value > 1.1 (推奨ベットルール 単勝) */
function isBuy(pred: HorsePrediction, entry: EntrySummary | undefined): boolean {
  const ev = winEv(pred, entry);
  return ev !== null && ev > 1.1;
}

/** 単勝 EV の色分け: 1.1 超 (BUY 条件) を強調、それ以外は控えめ。 */
function winEvClass(ev: number | null): string {
  if (ev === null) return 'text-muted-foreground';
  return ev > 1.1 ? 'font-semibold text-green-600' : 'text-muted-foreground';
}

interface EntryRow {
  entry: EntrySummary;
  pred: HorsePrediction | null;
}

type SortKey =
  | 'post_position'
  | 'horse_name'
  | 'odds_win'
  | 'popularity'
  | 'finish_position'
  | 'score'
  | 'win_prob'
  | 'place_prob'
  | 'win_ev';

type SortDir = 'asc' | 'desc';

interface SortState {
  key: SortKey;
  dir: SortDir;
}

// Columns that use asc as the initial direction when first clicked
const ASC_FIRST_KEYS: ReadonlySet<SortKey> = new Set(['post_position', 'popularity']);

/**
 * Compare two nullable numbers for sort purposes, returning the raw numeric
 * difference (positive = a after b, negative = a before b, 0 = equal).
 * null / NaN comparisons are NOT handled here — handled in sortRows to keep
 * null-last logic independent of sort direction.
 */
function compareNumbers(a: number, b: number): number {
  return a - b;
}

function compareStrings(a: string, b: string): number {
  return a.localeCompare(b, 'ja');
}

/**
 * Extract the raw numeric value for a given sort key from a row.
 * Returns undefined when the value is null / NaN / undefined.
 */
function numericValue(row: EntryRow, key: SortKey): number | undefined {
  let v: number | null | undefined;
  switch (key) {
    case 'post_position': v = row.entry.post_position; break;
    case 'odds_win':      v = row.entry.odds_win; break;
    case 'popularity':    v = row.entry.popularity; break;
    case 'finish_position': v = row.entry.finish_position; break;
    case 'score':         v = row.pred?.score; break;
    case 'win_prob':      v = row.pred?.win_prob; break;
    case 'place_prob':    v = row.pred?.place_prob; break;
    case 'win_ev':        v = winEv(row.pred, row.entry); break;
    default: return undefined;
  }
  if (v == null || isNaN(v as number)) return undefined;
  return v as number;
}

function sortRows(rows: EntryRow[], sort: SortState): EntryRow[] {
  const multiplier = sort.dir === 'asc' ? 1 : -1;

  return [...rows].sort((a, b) => {
    if (sort.key === 'horse_name') {
      const aNull = a.entry.horse_name == null;
      const bNull = b.entry.horse_name == null;
      if (aNull && bNull) return 0;
      // null is always last regardless of direction
      if (aNull) return 1;
      if (bNull) return -1;
      return compareStrings(a.entry.horse_name!, b.entry.horse_name!) * multiplier;
    }

    const av = numericValue(a, sort.key);
    const bv = numericValue(b, sort.key);

    if (av === undefined && bv === undefined) return 0;
    // null / NaN is always last regardless of direction
    if (av === undefined) return 1;
    if (bv === undefined) return -1;

    return compareNumbers(av, bv) * multiplier;
  });
}

interface SortableHeaderProps {
  label: string;
  sortKey: SortKey;
  sort: SortState;
  onSort: (key: SortKey) => void;
  className?: string;
}

function SortableHeader({ label, sortKey, sort, onSort, className }: SortableHeaderProps) {
  const isActive = sort.key === sortKey;
  const Icon = isActive
    ? sort.dir === 'asc'
      ? ChevronUp
      : ChevronDown
    : ChevronsUpDown;

  return (
    <TableHead
      className={`cursor-pointer select-none whitespace-nowrap ${className ?? ''}`}
      onClick={() => onSort(sortKey)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        <Icon
          className={`h-3 w-3 ${isActive ? 'text-foreground' : 'text-muted-foreground/40'}`}
        />
      </span>
    </TableHead>
  );
}

const BUY_TOOLTIP =
  '単勝 EV（単勝確率 × オッズ）> 1.1 の馬。本番モデルの OOS 単勝回収率は 0.856 ' +
  '（人気1番 0.789 を上回るが依然 1.0 未満＝平均では負け越し）。参考値。';

interface EntryPredictionTableProps {
  entries: EntrySummary[];
  predictions: HorsePrediction[] | null;
}

/**
 * Unified table merging entry data and prediction scores.
 * Default sort: score desc when predictions are available, post_position asc otherwise.
 * Clicking a sortable column header toggles sort direction.
 * null / NaN values always sort to the bottom regardless of direction.
 */
function EntryPredictionTable({ entries, predictions }: EntryPredictionTableProps) {
  const defaultSort: SortState = predictions
    ? { key: 'score', dir: 'desc' }
    : { key: 'post_position', dir: 'asc' };

  const [sort, setSort] = useState<SortState>(defaultSort);

  function handleSort(key: SortKey) {
    setSort((prev) => {
      if (prev.key === key) {
        return { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' };
      }
      // Different column: start with asc for asc-first keys, desc for the rest
      const dir: SortDir = ASC_FIRST_KEYS.has(key) ? 'asc' : 'desc';
      return { key, dir };
    });
  }

  const rows = useMemo<EntryRow[]>(() => {
    const predMap = new Map(predictions?.map((p) => [p.horse_id, p]) ?? []);
    const merged: EntryRow[] = entries.map((entry) => ({
      entry,
      pred: predMap.get(entry.horse_id) ?? null,
    }));
    return merged;
  }, [entries, predictions]);

  const sortedRows = useMemo(() => sortRows(rows, sort), [rows, sort]);

  const headerProps = { sort, onSort: handleSort };

  return (
    <Table>
      <TableHeader>
        {/* 実績 vs AI 予想 を視覚的に分離するためのグループ行 */}
        <TableRow>
          <TableHead
            colSpan={8}
            className="border-r border-border/50 bg-muted/40 text-center text-[11px] font-semibold uppercase tracking-wider text-emerald-700 dark:text-emerald-400"
          >
            実績データ
          </TableHead>
          <TableHead
            colSpan={4}
            className="border-r border-border/50 bg-muted/40 text-center text-[11px] font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-400"
          >
            AI 予想
          </TableHead>
          <TableHead colSpan={1} className="bg-muted/40" />
        </TableRow>
        <TableRow>
          <SortableHeader label="馬番" sortKey="post_position" className="w-12" {...headerProps} />
          <SortableHeader label="馬名" sortKey="horse_name" {...headerProps} />
          <TableHead className="text-center">年齢/性別</TableHead>
          <TableHead>騎手</TableHead>
          <TableHead className="text-right">馬体重</TableHead>
          <SortableHeader label="単勝オッズ" sortKey="odds_win" className="text-right" {...headerProps} />
          <SortableHeader label="人気" sortKey="popularity" className="text-right" {...headerProps} />
          <SortableHeader label="着順" sortKey="finish_position" className="border-r border-border/50 text-center" {...headerProps} />
          <SortableHeader label="スコア" sortKey="score" className="text-right" {...headerProps} />
          <SortableHeader label="単勝確率" sortKey="win_prob" className="text-right" {...headerProps} />
          <SortableHeader label="複勝確率" sortKey="place_prob" className="text-right" {...headerProps} />
          <SortableHeader label="単勝EV" sortKey="win_ev" className="border-r border-border/50 text-right" {...headerProps} />
          <TableHead className="text-center">推奨</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {sortedRows.map(({ entry, pred }) => (
          <TableRow key={entry.horse_id}>
            <TableCell className="font-medium">{entry.post_position ?? '—'}</TableCell>
            <TableCell>
              {entry.horse_name ?? (
                <span className="font-mono text-xs text-muted-foreground">{entry.horse_id}</span>
              )}
            </TableCell>
            <TableCell className="text-center">
              {entry.age ?? '?'}{entry.sex ?? ''}
            </TableCell>
            <TableCell>{entry.jockey_name ?? '—'}</TableCell>
            <TableCell className="text-right">
              {entry.horse_weight != null ? (
                <>
                  {entry.horse_weight}kg
                  {entry.horse_weight_diff != null && (
                    <span className="ml-1 text-xs text-muted-foreground">
                      ({entry.horse_weight_diff > 0 ? '+' : ''}{entry.horse_weight_diff})
                    </span>
                  )}
                </>
              ) : (
                '—'
              )}
            </TableCell>
            <TableCell className="text-right">{formatOdds(entry.odds_win)}</TableCell>
            <TableCell className="text-right">{entry.popularity ?? '—'}</TableCell>
            <TableCell className="border-r border-border/50 text-center">
              {entry.finish_position != null ? (
                <Badge variant={entry.finish_position <= 3 ? 'default' : 'secondary'}>
                  {entry.finish_position}着
                </Badge>
              ) : (
                '—'
              )}
            </TableCell>
            {/* Prediction columns */}
            <TableCell className="text-right">
              {pred != null ? formatScore(pred.score) : '—'}
            </TableCell>
            <TableCell className="text-right">
              {pred != null ? formatPercent(pred.win_prob) : '—'}
            </TableCell>
            <TableCell className="text-right">
              {pred != null ? formatPercent(pred.place_prob) : '—'}
            </TableCell>
            <TableCell
              className={`border-r border-border/50 text-right ${winEvClass(winEv(pred, entry))}`}
              title={BUY_TOOLTIP}
            >
              {pred != null ? formatRatio(winEv(pred, entry)) : '—'}
            </TableCell>
            <TableCell className="text-center">
              {pred != null && isBuy(pred, entry) && (
                <Badge variant="success" title={BUY_TOOLTIP}>BUY</Badge>
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function BuyBadgeNote() {
  return (
    <p className="mt-3 text-xs text-muted-foreground">
      BUY バッジは単勝 EV&gt;1.1 の馬を示しますが、本番モデルの OOS 単勝回収率は
      0.856（人気1番 0.789 を上回るものの 1.0 未満）です。回収率 1.0 超は未達なので
      実買いは慎重に。スコアはオッズ込み（value head）の総合評価、EV は単勝確率 × オッズ。
    </p>
  );
}

interface MetaItemProps {
  label: string;
  value: string;
  mono?: boolean;
}

function MetaItem({ label, value, mono }: MetaItemProps) {
  return (
    <div>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className={mono ? 'font-mono text-xs' : ''}>{value}</dd>
    </div>
  );
}

export function RaceDetail() {
  const { race_id = '' } = useParams<{ race_id: string }>();
  const [searchParams] = useSearchParams();
  const dateParam = searchParams.get('date');

  // AI 予想 (予想スコア + 推奨買い目) は画面を開いた瞬間ではなく、ボタンを
  // 押して初めて走らせる (重い推論を自動実行しない / ユーザー要望)。
  const [aiRequested, setAiRequested] = useState(false);

  const raceQuery = useRaceDetail(race_id);
  const predQuery = usePredictions(race_id, aiRequested);
  const recQuery = useRecommendations(
    race_id,
    aiRequested && Boolean(race_id) && !raceQuery.isPending && !raceQuery.isError,
  );
  // runShutuba scoped to this race so raceDetail is invalidated on completion
  const runShutubaMutation = useRunShutuba(race_id);

  const race = raceQuery.data;

  // NOTE: 出馬表取込・AI 予想はいずれも画面表示時に自動実行しない。
  // すべて下部の各ボタン (出馬表を取得 / AI 予想を実行) で明示的に開始する。

  // Race ページの Past タブへ戻る (?tab=past)。date を引き継いで一覧の選択日を復元。
  // 旧 `/past` は /races へ redirect され query を落とすため直接 /races を指す。
  const backLink = dateParam
    ? `/races?tab=past&date=${dateParam}`
    : '/races?tab=past';

  if (raceQuery.isPending) {
    return (
      <div className="flex flex-col gap-6 p-6">
        <BackLink to={backLink} />
        <PageHeader icon={Trophy} title="Race Detail" description={race_id} />
        <RaceDetailSkeleton />
      </div>
    );
  }

  if (raceQuery.isError) {
    const is404 = isNotFoundError(raceQuery.error);
    return (
      <div className="flex flex-col gap-6 p-6">
        <BackLink to={backLink} />
        <PageHeader icon={Trophy} title="Race Detail" description={race_id} />
        <EmptyState
          message={is404 ? '指定レース ID は見つかりません' : 'レース詳細の取得に失敗しました'}
          description={is404 ? undefined : 'バックエンドが起動しているか確認してください。'}
        />
        {is404 && (
          <div className="flex justify-center">
            <Button asChild variant="outline">
              <Link to="/upcoming">Upcoming Races へ戻る</Link>
            </Button>
          </div>
        )}
      </div>
    );
  }

  // raceQuery が success であっても TanStack Query の型は data: RaceDetail | undefined。
  // ここで明示的に narrowing し、以降は race を非 null として扱えるようにする。
  if (!race) {
    return (
      <div className="flex flex-col gap-6 p-6">
        <BackLink to={backLink} />
        <PageHeader icon={Trophy} title="Race Detail" description={race_id} />
        <RaceDetailSkeleton />
      </div>
    );
  }

  const predictions = predQuery.data?.predictions ?? null;

  const hasEntries = race.entries.length > 0;

  const isScrapingShutuba = runShutubaMutation.isPending || runShutubaMutation.isPolling;
  const isPredicting = aiRequested && (predQuery.isFetching || recQuery.isFetching);

  function handleRunShutuba() {
    runShutubaMutation.mutate(
      { race_ids: [race_id] },
      {
        onSuccess: (data) => {
          toast.success(`出馬表取込ジョブを開始しました（Job: ${data.job_id}）`);
        },
        onError: async (err) => {
          toast.error(`出馬表の取得に失敗しました: ${await formatErrorMessage(err)}`);
        },
      }
    );
  }

  function handleRunAi() {
    if (!aiRequested) {
      setAiRequested(true);
      return;
    }
    // 既にリクエスト済みなら再実行 (最新モデル / オッズ反映)
    predQuery.refetch();
    recQuery.refetch();
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      <BackLink to={backLink} />

      <PageHeader
        icon={Trophy}
        title={race.name ?? `${race.course} ${race.race_class ?? ''}`.trim()}
        description={`${race.date}・${race.surface}${race.distance}m・${race.race_id}`}
      >
        {!hasEntries && (
          <Button
            variant="outline"
            size="sm"
            disabled={isScrapingShutuba}
            onClick={handleRunShutuba}
          >
            <Download className="mr-1.5 h-4 w-4" />
            {isScrapingShutuba ? '出馬表取得中...' : '出馬表を取得'}
          </Button>
        )}
        {hasEntries && (
          <Button
            variant="outline"
            size="sm"
            disabled={isScrapingShutuba}
            onClick={handleRunShutuba}
            title="出馬表を再取得して単勝オッズ・人気・馬場状態を最新化します（発走が近いほど確定値に近づく）"
          >
            <RefreshCw className="mr-1.5 h-4 w-4" />
            {isScrapingShutuba ? 'オッズ更新中...' : 'オッズ更新'}
          </Button>
        )}
        {hasEntries && (
          <Button
            variant="outline"
            size="sm"
            disabled={isPredicting}
            onClick={handleRunAi}
          >
            <Sparkles className="mr-1.5 h-4 w-4" />
            {isPredicting
              ? 'AI 予想 実行中...'
              : aiRequested
                ? 'AI 予想を再実行'
                : 'AI 予想を実行'}
          </Button>
        )}
      </PageHeader>

      {/* Job progress banners (button 起動の取込/取得ジョブの進捗) */}
      {isScrapingShutuba && (
        <div className="rounded-md border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-800 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-200">
          出馬表を取得中...
        </div>
      )}

      {/* Race overview */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">レース概要</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3 lg:grid-cols-4">
            <MetaItem label="レース ID" value={race.race_id} mono />
            <MetaItem label="レース名" value={race.name ?? '—'} />
            <MetaItem label="開催日" value={race.date} />
            <MetaItem label="競馬場" value={race.course} />
            <MetaItem label="馬場種別" value={race.surface} />
            <MetaItem label="距離" value={race.distance ? `${race.distance} m` : '—'} />
            <MetaItem label="天候" value={race.weather ?? '—'} />
            <MetaItem label="馬場状態" value={race.track_condition ?? '—'} />
            <MetaItem label="クラス" value={race.race_class ?? '—'} />
            <MetaItem label="出走頭数" value={race.n_runners?.toString() ?? '—'} />
            <MetaItem label="単勝払戻" value={race.payout_win != null ? formatYen(race.payout_win) : '—'} />
            <MetaItem label="複勝払戻" value={race.payout_place ?? '—'} />
          </dl>
        </CardContent>
      </Card>

      {/* 出馬表が未取得のとき: ボタンで取り込む (自動取得しない) */}
      {!hasEntries && (
        <Card>
          <CardContent className="pt-6">
            <EmptyState
              message="出馬表が未取得です"
              description="自動取得は行いません。下のボタンで出馬表を取り込んでください。"
            >
              <Button onClick={handleRunShutuba} disabled={isScrapingShutuba}>
                <Download className="mr-1.5 h-4 w-4" />
                {isScrapingShutuba ? '出馬表取得中...' : '出馬表を取得'}
              </Button>
            </EmptyState>
          </CardContent>
        </Card>
      )}

      {/* Unified entry + prediction table */}
      {hasEntries && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">出走馬一覧</CardTitle>
          </CardHeader>
          <CardContent>
            {!aiRequested ? (
              // AI 予想 未実行: 実績データのみ表示 (予想列は空欄)。
              // 上部の「AI 予想を実行」ボタンでスコア + 推奨を取得する。
              <>
                <p className="mb-3 text-sm text-muted-foreground">
                  「AI 予想を実行」ボタンで予想スコア（単勝/複勝確率）と推奨買い目を取得します。
                </p>
                <EntryPredictionTable entries={race.entries} predictions={null} />
                <BuyBadgeNote />
              </>
            ) : predQuery.isPending ? (
              <Skeleton className="h-40 w-full" />
            ) : predQuery.isError ? (
              <>
                <p className="mb-3 text-sm text-muted-foreground">
                  {isServiceUnavailableError(predQuery.error)
                    ? 'active モデルが見つかりません。予想スコア列は非表示です。'
                    : '予想データを取得できません。予想スコア列は非表示です。'}
                </p>
                <EntryPredictionTable entries={race.entries} predictions={null} />
                <BuyBadgeNote />
              </>
            ) : (
              <>
                <EntryPredictionTable entries={race.entries} predictions={predictions} />
                <BuyBadgeNote />
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* Recommendations card — AI 予想を実行したときのみ表示 */}
      {hasEntries && aiRequested && (
        <RecommendationsCard
          raceId={race_id}
          data={recQuery.data}
          isPending={recQuery.isPending}
          isError={recQuery.isError}
          error={recQuery.error}
        />
      )}
    </div>
  );
}

interface BackLinkProps {
  to: string;
}

function BackLink({ to }: BackLinkProps) {
  return (
    <Link
      to={to}
      className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      aria-label="Past Races へ戻る"
    >
      <ChevronLeft className="h-4 w-4" />
      戻る
    </Link>
  );
}
