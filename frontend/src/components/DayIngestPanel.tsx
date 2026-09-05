import { useState } from 'react';
import { Download, Flag, Play, Square } from 'lucide-react';

import { useJobStatus } from '@/hooks/useJobStatus';
import { useRunResults } from '@/hooks/useRunResults';
import { useRunShutuba } from '@/hooks/useRunShutuba';
import { useScraperRun } from '@/hooks/useScraperRun';
import { useScraperStatus } from '@/hooks/useScraperStatus';
import { useScraperStop } from '@/hooks/useScraperStop';
import { useUpdateSettings } from '@/hooks/useSettings';
import { useScraperStore } from '@/store/app';
import { JobProgressCard } from '@/components/JobProgressCard';
import { Button } from '@/components/ui/button';
import { toast } from '@/lib/toast';
import { discoverTodayRaceIds, formatErrorMessage } from '@/lib/api';

interface DayIngestPanelProps {
  /** カレンダーで選択中の日 (ISO)。すべての操作はこの日に対して行う。 */
  date: string | undefined;
  /** その日に取り込めているレース数。0 なら未取得。 */
  raceCount: number;
  /** その日のレースが既に手元にあるか。 */
  hasResults: boolean;
}

type DayKind = 'past' | 'today' | 'future';

/**
 * 選択中の日が過去・当日・未来のどれか。出せる操作がこれで決まる。
 *
 *   past   … 結果のみ。**出馬表は出さない** — 結果ページに単勝オッズも載っており、
 *            単日 ingest が出走馬・オッズ・着順・払戻をまとめて入れるので、
 *            出馬表を取り直す意味が無い (netkeiba に無駄な負荷をかけるだけ)。
 *   today  … 両方。朝は出馬表、レースが終われば結果、と同じ日に両方要る。
 *   future … 出馬表のみ。結果はまだ存在しない。
 */
function dayKind(date: string): DayKind {
  const d = new Date();
  const today = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(
    d.getDate()
  ).padStart(2, '0')}`;
  if (date < today) return 'past';
  return date === today ? 'today' : 'future';
}

/**
 * 選択中の日に対する取込操作をまとめたパネル。
 *
 * 旧実装では「今週末のレースを取得」「再取込」「結果取込ダイアログ」「単日 ingest」が
 * 別々の場所にあり、どれが何をするのか分からなくなっていた。ここでは操作を
 * **日付基準の 2 つ**に統一する:
 *
 *   出馬表・オッズ … その日の出走馬とオッズ (発走前の情報)
 *   結果          … 着順と払戻 (レース後にしか無い)
 *
 * どちらも「選択中の日」に対して働くので、カレンダーで日を選ぶ → 隣で取り込む、
 * という一本の導線になる。
 */
export function DayIngestPanel({ date, raceCount, hasResults }: DayIngestPanelProps) {
  const shutubaMutation = useRunShutuba();
  const resultsMutation = useRunResults();
  const runMutation = useScraperRun();
  const stopMutation = useScraperStop();
  const updateSettings = useUpdateSettings();
  const statusQuery = useScraperStatus();
  const setRunning = useScraperStore((s) => s.setRunning);
  const trackedJobId = useScraperStore((s) => s.trackedJobId);
  const setTrackedJobId = useScraperStore((s) => s.setTrackedJobId);
  const [confirmStop, setConfirmStop] = useState(false);
  const [discovering, setDiscovering] = useState(false);

  const jobStatus = useJobStatus(trackedJobId);
  const stopped = statusQuery.data?.stopped ?? false;
  const busy =
    discovering ||
    runMutation.isPending ||
    shutubaMutation.isPending ||
    shutubaMutation.isPolling ||
    resultsMutation.isPending ||
    resultsMutation.isPolling ||
    (trackedJobId != null && jobStatus.data?.status === 'running');

  const kind: DayKind | null = date ? dayKind(date) : null;

  /** 出馬表・オッズ: その日の race_id を発見してから取り込む。 */
  async function handleFetchEntries() {
    if (!date) return;
    setDiscovering(true);
    try {
      const found = await discoverTodayRaceIds(date);
      if (found.race_ids.length === 0) {
        toast.info(`${date} に JRA の開催はありません`);
        return;
      }
      shutubaMutation.mutate(
        { race_ids: found.race_ids },
        {
          onSuccess: (job) => setTrackedJobId(job.job_id),
          onError: async (err) => {
            toast.error('出馬表の取得に失敗しました', {
              description: await formatErrorMessage(err),
              action: { label: '再試行', onClick: handleFetchEntries },
            });
          },
        }
      );
    } catch (err) {
      toast.error('開催レースの検出に失敗しました', {
        description: await formatErrorMessage(err),
        action: { label: '再試行', onClick: handleFetchEntries },
      });
    } finally {
      setDiscovering(false);
    }
  }

  /** 結果: 着順・払戻を取り込む (その日 1 日分)。 */
  function handleFetchResults() {
    if (!date) return;
    setRunning(true);
    // レース自体が未取得なら単日 ingest (レース + 結果) から入る
    const mutation = raceCount === 0 ? runMutation : resultsMutation;
    const body = raceCount === 0 ? { date } : { from: date, to: date };
    mutation.mutate(body as never, {
      onSuccess: (job: { job_id: string }) => {
        setTrackedJobId(job.job_id);
        toast.success(`${date} の結果取込を開始しました`);
      },
      onError: async (err: unknown) => {
        setRunning(false);
        toast.error('結果の取込に失敗しました', {
          description: await formatErrorMessage(err),
          action: { label: '再試行', onClick: handleFetchResults },
        });
      },
    });
  }

  function handleStop() {
    setConfirmStop(false);
    stopMutation.mutate(undefined, {
      onSuccess: () => toast.success('スクレイパーを停止しました'),
      onError: async (err) =>
        toast.error('停止に失敗しました', { description: await formatErrorMessage(err) }),
    });
  }

  /** 停止フラグを解除する。旧 OPS タブのトグルに代わる復帰手段。 */
  function handleResume() {
    updateSettings.mutate(
      { scraper_stopped: false },
      {
        onSuccess: () => toast.success('スクレイパーを再開しました'),
        onError: async (err) =>
          toast.error('再開に失敗しました', { description: await formatErrorMessage(err) }),
      }
    );
  }

  return (
    <div className="flex w-full flex-col gap-3">
      {/* 両端揃えにしない。列に置くと「この日のデータ ……… 36 R 取込済み」と
          離れ、見出しと値が別のものに見える。 */}
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-label-ja">この日のデータ</span>
        <span className="font-mono text-2xs tabular-nums text-subtle-foreground">
          {raceCount > 0 ? `${raceCount} R 取込済み` : '未取得'}
        </span>
      </div>

      <div className="flex flex-wrap gap-2">
        {/* 出馬表は発走前の情報。過去日では結果取込が上位互換なので出さない */}
        {kind !== 'past' && (
          <Button
            size="sm"
            variant={hasResults ? 'outline' : 'default'}
            onClick={handleFetchEntries}
            disabled={!date || busy || stopped}
            title="出走馬・単勝オッズ・馬場状態を取り込みます（発走前の情報）"
          >
            <Download className="mr-1.5 h-4 w-4" />
            {raceCount > 0 ? '出馬表・オッズを更新' : '出馬表を取得'}
          </Button>
        )}

        {/* 結果は未来日にはまだ無い。当日は終わったレースの分が取れる */}
        {kind !== null && kind !== 'future' && (
          <Button
            size="sm"
            variant="outline"
            onClick={handleFetchResults}
            disabled={!date || busy || stopped}
            title="着順と払戻を取り込みます（レース確定後にのみ取得できます）"
          >
            <Flag className="mr-1.5 h-4 w-4" />
            結果を取り込む
          </Button>
        )}

        {stopped ? (
          <Button size="sm" variant="outline" onClick={handleResume}>
            <Play className="mr-1.5 h-4 w-4" />
            スクレイパーを再開
          </Button>
        ) : busy ? (
          confirmStop ? (
            <>
              <Button size="sm" variant="destructive" onClick={handleStop}>
                停止する
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setConfirmStop(false)}>
                やめる
              </Button>
            </>
          ) : (
            <Button size="sm" variant="ghost" onClick={() => setConfirmStop(true)}>
              <Square className="mr-1.5 h-4 w-4" />
              停止
            </Button>
          )
        ) : null}
      </div>

      {stopped && (
        <p className="text-xs leading-relaxed text-destructive">
          スクレイパーは停止フラグが立っています（`KEIBA_SCRAPER_STOP=1` 相当）。
          取り込むには先に再開してください。
        </p>
      )}

      {trackedJobId && (
        <JobProgressCard
          jobId={trackedJobId}
          title="取込ジョブ"
          onDismiss={() => setTrackedJobId(null)}
        />
      )}
    </div>
  );
}
