import { useEffect, useRef, useState } from 'react';
import { CalendarClock, RefreshCw } from 'lucide-react';

import { useUpcomingRaces } from '@/hooks/useUpcomingRaces';
import { useRunShutuba } from '@/hooks/useRunShutuba';
import { useJobStatus } from '@/hooks/useJobStatus';
import { RaceCard } from '@/components/RaceCard';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { discoverTodayRaceIds, formatErrorMessage } from '@/lib/api';
import { toast } from '@/components/ui/toast';

function RaceListSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 6 }).map((_, i) => (
        <Skeleton key={i} className="h-44 rounded-lg" />
      ))}
    </div>
  );
}

/**
 * Determine whether today or tomorrow is likely a JRA kaisai day.
 *
 * JRA holds races on Saturdays and Sundays (JST). We also allow Monday (0)
 * since a Saturday/Sunday ingest may not complete until early Monday.
 * This keeps the auto-bootstrap from firing on non-race days.
 */
function isLikelyKaisaiDay(): boolean {
  // Use local time as a pragmatic approximation of JST.
  const day = new Date().getDay(); // 0=Sun, 1=Mon, 6=Sat
  return day === 0 || day === 6 || day === 1;
}

type BootstrapState =
  | { phase: 'idle' }
  | { phase: 'discovering' }
  | { phase: 'no_races' }
  | { phase: 'scraping'; jobId: string }
  | { phase: 'done' }
  | { phase: 'error'; message: string };

export function UpcomingRaces() {
  const { data, isPending, isError, refetch } = useUpcomingRaces(7);

  const [bootstrap, setBootstrap] = useState<BootstrapState>({ phase: 'idle' });
  // Tracks whether auto-bootstrap has already been attempted this mount to
  // prevent infinite loops. Reset on manual re-fetch.
  const autoFiredRef = useRef(false);

  // Hook for running shutuba scraper (job polling built in)
  const runShutuba = useRunShutuba();

  // Poll the scraping job when we have one
  const scrapingJobId =
    bootstrap.phase === 'scraping' ? bootstrap.jobId : null;
  const jobStatus = useJobStatus(scrapingJobId);

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

  // Auto-bootstrap: fire once when races are empty and conditions are met
  useEffect(() => {
    if (isPending || isError) return;
    if (data && data.races.length > 0) return;
    if (autoFiredRef.current) return;
    if (bootstrap.phase !== 'idle') return;
    if (!isLikelyKaisaiDay()) return;

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

  const isBootstrapping =
    bootstrap.phase === 'discovering' || bootstrap.phase === 'scraping';

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
        <RaceListSkeleton />
      ) : isError ? (
        <EmptyState
          message="レース情報の取得に失敗しました"
          description="バックエンドが起動しているか確認してください。"
        />
      ) : data.races.length === 0 ? (
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
          <RaceListSkeleton />
        ) : (
          <EmptyState
            message="今週の予定レースはありません"
            description="スクレイパーでデータを取り込んでください。"
          />
        )
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data.races.map((race) => (
            <RaceCard key={race.race_id} race={race} />
          ))}
        </div>
      )}
    </div>
  );
}
