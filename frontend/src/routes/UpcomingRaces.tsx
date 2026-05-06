import { useEffect, useRef, useState } from 'react';
import { CalendarClock, RefreshCw } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import { useUpcomingRaces } from '@/hooks/useUpcomingRaces';
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
import { discoverTodayRaceIds, formatErrorMessage } from '@/lib/api';
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

function groupByCourse(races: RaceSummary[]): CourseSection[] {
  const map = new Map<string, RaceSummary[]>();
  for (const race of races) {
    const list = map.get(race.course) ?? [];
    list.push(race);
    map.set(race.course, list);
  }
  return Array.from(map.entries()).map(([course, rs]) => ({ course, races: rs }));
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
      <h2
        id={`upcoming-course-${section.course}`}
        className="mb-2 text-base font-semibold text-foreground"
      >
        {section.course}
      </h2>
      <div className="overflow-hidden rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-24">日付</TableHead>
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
                className="cursor-pointer hover:bg-accent/60"
                onClick={() => onRowClick(race)}
                role="button"
                aria-label={`${section.course} ${raceNumber(race.race_id)}R`}
              >
                <TableCell className="text-sm text-muted-foreground">
                  {formatRaceDate(race.date)}
                </TableCell>
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
      </div>
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

export function UpcomingRaces() {
  const navigate = useNavigate();
  const { data, isPending, isError, refetch } = useUpcomingRaces(7);

  const [bootstrap, setBootstrap] = useState<BootstrapState>({ phase: 'idle' });
  // Tracks whether auto-bootstrap has already been attempted this mount.
  const autoFiredRef = useRef(false);

  const runShutuba = useRunShutuba();

  const scrapingJobId =
    bootstrap.phase === 'scraping' ? bootstrap.jobId : null;
  const jobStatus = useJobStatus(scrapingJobId);

  // Collect race IDs for bulk predictions — only when we have data
  const allRaceIds = data?.races.map((r) => r.race_id) ?? [];
  const { data: bulkPredData } = useBulkPredictions(allRaceIds, 3);
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

  // Auto-bootstrap: fire once when races are empty
  useEffect(() => {
    if (isPending || isError) return;
    if (data && data.races.length > 0) return;
    if (autoFiredRef.current) return;
    if (bootstrap.phase !== 'idle') return;

    autoFiredRef.current = true;
    runBootstrap();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPending, isError, data, bootstrap.phase]);

  async function runBootstrap() {
    setBootstrap({ phase: 'discovering' });

    let raceIds: string[];
    try {
      const result = await discoverTodayRaceIds();
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

  function handleManualRefetch() {
    autoFiredRef.current = false;
    setBootstrap({ phase: 'idle' });
    refetch();
  }

  function handleRowClick(race: RaceSummary) {
    navigate(`/races/${race.race_id}?date=${race.date}`);
  }

  const isBootstrapping =
    bootstrap.phase === 'discovering' || bootstrap.phase === 'scraping';

  const sections = data ? groupByCourse(data.races) : [];

  return (
    <div className="flex flex-col gap-6 p-6">
      <PageHeader
        icon={CalendarClock}
        title="Upcoming Races"
        description="直近 7 日に予定されているレース一覧"
      >
        <Button
          variant="outline"
          size="sm"
          disabled={isBootstrapping}
          onClick={handleManualRefetch}
          aria-label="再取込"
        >
          <RefreshCw className="mr-1.5 h-4 w-4" />
          再取込
        </Button>
      </PageHeader>

      {/* Bootstrap progress banner */}
      {isBootstrapping && (
        <div className="rounded-md border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-800 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-200">
          {bootstrap.phase === 'discovering'
            ? '本日の開催レースを確認中...'
            : '当日のレース情報を取得中...（最大 3 分かかります）'}
        </div>
      )}

      {isPending ? (
        <TableSkeleton />
      ) : isError ? (
        <EmptyState
          message="レース情報の取得に失敗しました"
          description="バックエンドが起動しているか確認してください。"
        />
      ) : sections.length === 0 ? (
        bootstrap.phase === 'no_races' ? (
          <EmptyState
            message="本日の JRA レースはありません"
            description="開催予定がない日です。週末の開催日にご利用ください。"
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
            message="今週の予定レースはありません"
            description="スクレイパーでデータを取り込んでください。"
          />
        )
      ) : (
        <div className="flex flex-col gap-8">
          {sections.map((section) => (
            <RaceTable
              key={section.course}
              section={section}
              predictions={predictions}
              onRowClick={handleRowClick}
            />
          ))}
        </div>
      )}
    </div>
  );
}
