import { useEffect, useState } from 'react';
import { CalendarClock, RefreshCw, Sparkles } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import { useThisWeekendRaces } from '@/hooks/useThisWeekendRaces';
import { useRunShutuba } from '@/hooks/useRunShutuba';
import { useJobStatus } from '@/hooks/useJobStatus';
import { useBulkPredictions } from '@/hooks/useBulkPredictions';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { discoverThisWeekendRaceIds, formatErrorMessage } from '@/lib/api';
import { PastRaces } from '@/routes/PastRaces';
import { toast } from '@/components/ui/toast';
import type { RaceSummary, RacePredictionSummary } from '@/types/api';

// ── helpers ───────────────────────────────────────────────────────────────────

/** Extract 2-digit race number from race_id (last 2 chars). */
function raceNumber(raceId: string): string {
  return raceId.slice(-2);
}

/**
 * Format date string (YYYY-MM-DD) to Japanese short form, e.g. "5/9 (土)".
 * Falls back to the raw string if parsing fails.
 */
function formatRaceDate(dateStr: string): string {
  try {
    // Parse as local date to avoid timezone shifting
    const [y, m, d] = dateStr.split('-').map(Number);
    const dt = new Date(y, m - 1, d);
    const month = dt.getMonth() + 1;
    const day = dt.getDate();
    const weekdays = ['日', '月', '火', '水', '木', '金', '土'];
    const dow = weekdays[dt.getDay()];
    return `${month}/${day} (${dow})`;
  } catch {
    return dateStr;
  }
}

/**
 * Format top-N horse predictions into a compact string.
 * Example: "①メイショウ ②キタサン ③ドゥラ"
 */
function formatTopHorses(summary: RacePredictionSummary | undefined): string {
  if (!summary || summary.top_horses.length === 0) return '—';
  const circled = ['①', '②', '③', '④', '⑤'];
  return summary.top_horses
    .map((h, i) => {
      const rank = circled[i] ?? `${i + 1}.`;
      const name = h.horse_name ?? (h.post_position != null ? `${h.post_position}番` : '?');
      return `${rank}${name}`;
    })
    .join(' ');
}

// ── grouping ──────────────────────────────────────────────────────────────────

interface CourseSection {
  course: string;
  races: RaceSummary[];
}

interface DaySection {
  /** YYYY-MM-DD (group key) */
  date: string;
  /** "5/10 (土)" 等の表示用 */
  label: string;
  /** その日の course 別さらに sub-section */
  courseSections: CourseSection[];
}

function groupByCourse(races: RaceSummary[]): CourseSection[] {
  const map = new Map<string, RaceSummary[]>();
  for (const race of races) {
    const list = map.get(race.course) ?? [];
    list.push(race);
    map.set(race.course, list);
  }
  return Array.from(map.entries()).map(([course, rs]) => ({ course, races: rs }));
}

/**
 * race を 日付 → コース の二段階で grouping する。
 * 土曜と日曜が同じテーブルに混じらないよう、日付セクションでまず分割。
 */
function groupByDayAndCourse(races: RaceSummary[]): DaySection[] {
  const byDate = new Map<string, RaceSummary[]>();
  for (const race of races) {
    const list = byDate.get(race.date) ?? [];
    list.push(race);
    byDate.set(race.date, list);
  }
  const sortedDates = Array.from(byDate.keys()).sort();
  return sortedDates.map((date) => ({
    date,
    label: formatRaceDate(date),
    courseSections: groupByCourse(byDate.get(date) ?? []),
  }));
}

// ── skeleton ──────────────────────────────────────────────────────────────────

function TableSkeleton() {
  return (
    <div className="space-y-4">
      {Array.from({ length: 2 }).map((_, i) => (
        <Skeleton key={i} className="h-32 w-full rounded-lg" />
      ))}
    </div>
  );
}

// ── race table ────────────────────────────────────────────────────────────────

interface RaceTableProps {
  section: CourseSection;
  predictions: Record<string, RacePredictionSummary>;
  onRowClick: (race: RaceSummary) => void;
}

function RaceTable({ section, predictions, onRowClick }: RaceTableProps) {
  return (
    <section aria-labelledby={`upcoming-course-${section.course}`}>
      <h3
        id={`upcoming-course-${section.course}`}
        className="mb-2 text-sm font-semibold text-muted-foreground"
      >
        {section.course}
      </h3>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-16">R</TableHead>
            <TableHead>レース名</TableHead>
            <TableHead>クラス</TableHead>
            <TableHead>馬場</TableHead>
            <TableHead className="text-right">距離</TableHead>
            <TableHead className="text-right">頭数</TableHead>
            <TableHead>AI 予想</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {section.races.map((race) => (
            <TableRow
              key={race.race_id}
              className="cursor-pointer"
              onClick={() => onRowClick(race)}
              role="button"
              aria-label={`${section.course} ${raceNumber(race.race_id)}R`}
            >
              <TableCell className="font-medium">{raceNumber(race.race_id)}R</TableCell>
              <TableCell>{race.name ?? '—'}</TableCell>
              <TableCell>{race.race_class ?? '—'}</TableCell>
              <TableCell>{race.surface}</TableCell>
              <TableCell className="text-right">{race.distance} m</TableCell>
              <TableCell className="text-right">{race.n_runners ?? '—'}</TableCell>
              <TableCell className="text-xs text-muted-foreground">
                {formatTopHorses(predictions[race.race_id])}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </section>
  );
}

// ── bootstrap state machine ───────────────────────────────────────────────────

type BootstrapState =
  | { phase: 'idle' }
  | { phase: 'discovering' }
  | { phase: 'no_races' }
  | { phase: 'scraping'; jobId: string }
  | { phase: 'done' }
  | { phase: 'error'; message: string };

// ── main component ────────────────────────────────────────────────────────────

interface UpcomingRacesProps {
  /** Race route のタブ内に埋め込まれる場合 true。自前の PageHeader と外周 padding を抑制する。 */
  embedded?: boolean;
}

/** 相対時間ラベル ("3 分前" / "2 時間前" / "今") を返す。0 のとき空文字。 */
function _relativeTimeJa(timestamp: number): string {
  if (!timestamp) return '';
  const sec = Math.floor((Date.now() - timestamp) / 1000);
  if (sec < 30) return '今';
  if (sec < 60) return `${sec} 秒前`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} 分前`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr} 時間前`;
  const day = Math.floor(hr / 24);
  return `${day} 日前`;
}

export function UpcomingRaces({ embedded = false }: UpcomingRacesProps = {}) {
  const navigate = useNavigate();
  const { data, isPending, isError, refetch, dataUpdatedAt } = useThisWeekendRaces();
  // dataUpdatedAt は react-query の最終 fetch 成功時刻 (ms)。これを 30 秒
  // 間隔で再評価して "X 分前" 表示を実時間更新する。
  const [, setTickNow] = useState(Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setTickNow(Date.now()), 30_000);
    return () => window.clearInterval(id);
  }, []);
  const lastUpdatedLabel = _relativeTimeJa(dataUpdatedAt);

  const [bootstrap, setBootstrap] = useState<BootstrapState>({ phase: 'idle' });
  // 一覧の「AI 予想」列は、開いた瞬間に全レース分の推論を裏で走らせず、
  // ボタンを押して初めて取得する (ユーザー要望でボタン主導化)。
  const [predictionsRequested, setPredictionsRequested] = useState(false);

  const runShutuba = useRunShutuba();

  const scrapingJobId =
    bootstrap.phase === 'scraping' ? bootstrap.jobId : null;
  const jobStatus = useJobStatus(scrapingJobId);

  // Collect race IDs for bulk predictions — fetched only after the user opts in.
  const allRaceIds = data?.races.map((r) => r.race_id) ?? [];
  const {
    data: bulkPredData,
    isFetching: isPredicting,
    refetch: refetchPredictions,
  } = useBulkPredictions(allRaceIds, 3, predictionsRequested);
  const predictions = bulkPredData?.predictions ?? {};

  // React to job reaching terminal state
  useEffect(() => {
    if (bootstrap.phase !== 'scraping' || !jobStatus.data) return;

    const { status } = jobStatus.data;
    if (status === 'completed') {
      setBootstrap({ phase: 'done' });
      refetch();
    } else if (status === 'failed') {
      const msg = jobStatus.data.error ?? '不明なエラー';
      setBootstrap({ phase: 'error', message: msg });
      toast.error(`レース情報の取得に失敗しました: ${msg}`);
    }
  }, [jobStatus.data, bootstrap.phase, refetch]);

  async function runBootstrap(forced: boolean = false) {
    setBootstrap({ phase: 'discovering' });

    let raceIds: string[];
    try {
      // 強制再取込時は backend キャッシュ (30 分 TTL) を bypass する
      const result = await discoverThisWeekendRaceIds(forced);
      raceIds = result.race_ids;
    } catch (err) {
      const msg = await formatErrorMessage(err);
      setBootstrap({ phase: 'error', message: msg });
      toast.error(`race_id の自動発見に失敗しました: ${msg}`);
      return;
    }

    if (raceIds.length === 0) {
      setBootstrap({ phase: 'no_races' });
      return;
    }

    runShutuba.mutate(
      { race_ids: raceIds },
      {
        onSuccess: (jobAccepted) => {
          setBootstrap({ phase: 'scraping', jobId: jobAccepted.job_id });
        },
        onError: async (err) => {
          const msg = await formatErrorMessage(err);
          setBootstrap({ phase: 'error', message: msg });
          toast.error(`出馬表取込の開始に失敗しました: ${msg}`);
        },
      },
    );
  }

  function handleStartIngest(forced: boolean) {
    // forced=true: races > 0 でも shutuba ingest を強制再実行し、馬体重・直前
    // odds_win・除外馬・補欠繰上り等を最新化する (再取込ボタン)。
    // forced=false: まだ取り込んでいない週末を初回取得する (空状態のボタン)。
    if (isBootstrapping) return;
    runBootstrap(forced);
  }

  function handleRunPredictions() {
    if (allRaceIds.length === 0) return;
    if (!predictionsRequested) {
      setPredictionsRequested(true);
    } else {
      refetchPredictions();
    }
  }

  function handleRowClick(race: RaceSummary) {
    navigate(`/races/${race.race_id}?date=${race.date}`);
  }

  const isBootstrapping =
    bootstrap.phase === 'discovering' || bootstrap.phase === 'scraping';

  const daySections = data ? groupByDayAndCourse(data.races) : [];

  const headerActions = (
    <>
      <Button
        variant="outline"
        size="sm"
        disabled={isPredicting || allRaceIds.length === 0}
        onClick={handleRunPredictions}
        aria-label="AI 予想を実行"
      >
        <Sparkles className="mr-1.5 h-4 w-4" />
        {isPredicting
          ? 'AI 予想 実行中...'
          : predictionsRequested
            ? 'AI 予想を再取得'
            : 'AI 予想を実行'}
      </Button>
      <Button
        variant="outline"
        size="sm"
        disabled={isBootstrapping}
        onClick={() => handleStartIngest(true)}
        aria-label="再取込"
      >
        <RefreshCw className="mr-1.5 h-4 w-4" />
        再取込
      </Button>
    </>
  );

  return (
    <div className={embedded ? 'flex flex-col gap-6' : 'flex flex-col gap-6 p-6'}>
      {!embedded && (
        <PageHeader
          icon={CalendarClock}
          title="今週末のレース（JRA）"
          description="今週土・日に予定されている JRA レース一覧"
        >
          {lastUpdatedLabel && (
            <span className="text-xs text-muted-foreground">
              最終更新: {lastUpdatedLabel}
            </span>
          )}
          {headerActions}
        </PageHeader>
      )}
      {embedded && (
        <div className="flex items-center justify-between gap-3">
          <p className="text-sm text-muted-foreground">
            今週土・日に予定されている JRA レース一覧
          </p>
          <div className="flex items-center gap-3">
            {lastUpdatedLabel && (
              <span className="text-xs text-muted-foreground">
                最終更新: {lastUpdatedLabel}
              </span>
            )}
            {headerActions}
          </div>
        </div>
      )}

      {/* Bootstrap progress banner */}
      {isBootstrapping && (
        <div className="rounded-lg border border-info/30 bg-info/10 px-4 py-3 text-sm text-info">
          {bootstrap.phase === 'discovering'
            ? '今週末の JRA レースを確認中...'
            : '今週末の JRA レースを取得中...（最大 5 分）'}
        </div>
      )}

      {isPending ? (
        <TableSkeleton />
      ) : isError ? (
        <EmptyState
          message="レース情報の取得に失敗しました"
          description="バックエンドが起動しているか確認してください。"
        />
      ) : daySections.length === 0 ? (
        bootstrap.phase === 'no_races' ? (
          <EmptyState
            message="今週末の JRA レースはありません"
            description="開催予定がない週末です。次の開催日にご利用ください。"
          />
        ) : bootstrap.phase === 'error' ? (
          <EmptyState
            message="レース情報の自動取得に失敗しました"
            description={bootstrap.message}
          />
        ) : isBootstrapping ? (
          <TableSkeleton />
        ) : (
          <EmptyState
            message="今週末のレースは未取得です"
            description="自動取得は行いません。下のボタンで今週末の JRA レースを取り込んでください。"
          >
            <Button onClick={() => handleStartIngest(false)} disabled={isBootstrapping}>
              <RefreshCw className="mr-1.5 h-4 w-4" />
              今週末のレースを取得
            </Button>
          </EmptyState>
        )
      ) : (
        <Tabs defaultValue={daySections[0].date} className="flex flex-col gap-6">
          <TabsList className="self-start">
            {daySections.map((day) => {
              const total = day.courseSections.reduce(
                (sum, s) => sum + s.races.length,
                0,
              );
              return (
                <TabsTrigger key={day.date} value={day.date}>
                  {day.label}
                  <span className="ml-2 text-xs text-muted-foreground">
                    {total}
                  </span>
                </TabsTrigger>
              );
            })}
            <TabsTrigger value="past">Past</TabsTrigger>
          </TabsList>
          {daySections.map((day) => (
            <TabsContent
              key={day.date}
              value={day.date}
              className="mt-0 flex flex-col gap-6"
            >
              {day.courseSections.map((section) => (
                <RaceTable
                  key={`${day.date}-${section.course}`}
                  section={section}
                  predictions={predictions}
                  onRowClick={handleRowClick}
                />
              ))}
            </TabsContent>
          ))}
          <TabsContent value="past" className="mt-0">
            <PastRaces embedded />
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
