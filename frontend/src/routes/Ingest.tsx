import { Database } from 'lucide-react';
import { useState } from 'react';

import { useScraperStatus } from '@/hooks/useScraperStatus';
import { useScraperRun } from '@/hooks/useScraperRun';
import { useScraperStop } from '@/hooks/useScraperStop';
import { useRunShutuba } from '@/hooks/useRunShutuba';
import { useFetchLiveOdds } from '@/hooks/useFetchLiveOdds';
import { useScraperStore } from '@/store/app';
import { PageHeader } from '@/components/PageHeader';
import { ScraperStatusCard } from '@/components/ScraperStatusCard';
import { JobProgressCard } from '@/components/JobProgressCard';
import { IngestRunDialog } from '@/components/IngestRunDialog';
import { EmptyState } from '@/components/EmptyState';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { toast } from '@/components/ui/toast';
import { formatErrorMessage } from '@/lib/api';
import type { ScraperRunRequest } from '@/types/api';

const RACE_ID_RE = /^\d{12}$/;
const ALL_ODDS_TYPES = ['b1', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'] as const;
type OddsType = (typeof ALL_ODDS_TYPES)[number];

const ODDS_TYPE_LABELS: Record<OddsType, string> = {
  b1: '単勝/複勝',
  b3: '枠連',
  b4: '馬連',
  b5: 'ワイド',
  b6: '馬単',
  b7: '三連複',
  b8: '三連単',
};

/** 出馬表取込（race_id 指定）カード */
function ShutubaIngestCard() {
  const setTrackedJobId = useScraperStore((s) => s.setTrackedJobId);
  const runShutuba = useRunShutuba();

  const [raceIdsInput, setRaceIdsInput] = useState('');
  const [limitInput, setLimitInput] = useState('');
  const [errors, setErrors] = useState<{ raceIds?: string }>({});

  function validate(): boolean {
    const raw = raceIdsInput.trim();
    if (!raw) {
      setErrors({ raceIds: 'race_id を 1 件以上入力してください' });
      return false;
    }
    const ids = raw.split(',').map((s) => s.trim()).filter(Boolean);
    const invalid = ids.filter((id) => !RACE_ID_RE.test(id));
    if (invalid.length > 0) {
      setErrors({ raceIds: `12 桁の数字でない ID があります: ${invalid.join(', ')}` });
      return false;
    }
    setErrors({});
    return true;
  }

  function handleSubmit() {
    if (!validate()) return;

    const ids = raceIdsInput.trim().split(',').map((s) => s.trim()).filter(Boolean);
    const limit = limitInput ? parseInt(limitInput, 10) : undefined;

    runShutuba.mutate(
      { race_ids: ids, limit },
      {
        onSuccess: (data) => {
          setTrackedJobId(data.job_id);
          toast.success(`出馬表取込を開始しました（Job: ${data.job_id}）`);
          setRaceIdsInput('');
          setLimitInput('');
        },
        onError: async (err) => {
          toast.error(`出馬表取込の開始に失敗しました: ${await formatErrorMessage(err)}`);
        },
      }
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">出馬表取込（race_id 指定）</CardTitle>
        <CardDescription>
          12 桁 race_id をカンマ区切りで指定してください（例: 202506050911,202506050912）
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="shutuba-race-ids">race_ids（カンマ区切り）</Label>
          <Input
            id="shutuba-race-ids"
            placeholder="202506050911,202506050912"
            value={raceIdsInput}
            onChange={(e) => setRaceIdsInput(e.target.value)}
            aria-describedby={errors.raceIds ? 'shutuba-race-ids-error' : undefined}
          />
          {errors.raceIds && (
            <p id="shutuba-race-ids-error" className="text-xs text-destructive">
              {errors.raceIds}
            </p>
          )}
        </div>

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="shutuba-limit">limit（省略可）</Label>
          <Input
            id="shutuba-limit"
            type="number"
            min={1}
            placeholder="件数上限"
            value={limitInput}
            onChange={(e) => setLimitInput(e.target.value)}
            className="w-32"
          />
        </div>

        <div className="flex items-center gap-2">
          <Button onClick={handleSubmit} disabled={runShutuba.isPending}>
            {runShutuba.isPending ? '取込中...' : '取込開始'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

/** 当日連系オッズ取得カード */
function LiveOddsCard() {
  const setTrackedJobId = useScraperStore((s) => s.setTrackedJobId);
  const fetchOdds = useFetchLiveOdds();

  const [raceIdInput, setRaceIdInput] = useState('');
  const [selectedTypes, setSelectedTypes] = useState<Set<OddsType>>(new Set(ALL_ODDS_TYPES));
  const [errors, setErrors] = useState<{ raceId?: string; types?: string }>({});

  function toggleType(t: OddsType) {
    setSelectedTypes((prev) => {
      const next = new Set(prev);
      if (next.has(t)) {
        next.delete(t);
      } else {
        next.add(t);
      }
      return next;
    });
  }

  function validate(): boolean {
    const newErrors: { raceId?: string; types?: string } = {};
    if (!RACE_ID_RE.test(raceIdInput.trim())) {
      newErrors.raceId = '12 桁の数字を入力してください';
    }
    if (selectedTypes.size === 0) {
      newErrors.types = '少なくとも 1 つの券種を選択してください';
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }

  function handleSubmit() {
    if (!validate()) return;

    fetchOdds.mutate(
      { race_id: raceIdInput.trim(), types: [...selectedTypes] },
      {
        onSuccess: (data) => {
          setTrackedJobId(data.job_id);
          toast.success(`オッズ取得を開始しました（Job: ${data.job_id}）`);
        },
        onError: async (err) => {
          toast.error(`オッズ取得の開始に失敗しました: ${await formatErrorMessage(err)}`);
        },
      }
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">当日連系オッズ取得</CardTitle>
        <CardDescription>
          レース ID を指定して netkeiba からリアルタイムオッズを取得します
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="live-odds-race-id">race_id（12 桁）</Label>
          <Input
            id="live-odds-race-id"
            placeholder="202506050911"
            value={raceIdInput}
            onChange={(e) => setRaceIdInput(e.target.value)}
            className="w-48"
            aria-describedby={errors.raceId ? 'live-odds-race-id-error' : undefined}
          />
          {errors.raceId && (
            <p id="live-odds-race-id-error" className="text-xs text-destructive">
              {errors.raceId}
            </p>
          )}
        </div>

        <div className="flex flex-col gap-2">
          <Label>券種</Label>
          <div className="flex flex-wrap gap-4">
            {ALL_ODDS_TYPES.map((t) => (
              <div key={t} className="flex items-center gap-1.5">
                <input
                  id={`odds-type-${t}`}
                  type="checkbox"
                  checked={selectedTypes.has(t)}
                  onChange={() => toggleType(t)}
                  className="h-4 w-4 cursor-pointer accent-primary"
                />
                <Label htmlFor={`odds-type-${t}`} className="cursor-pointer font-normal">
                  {ODDS_TYPE_LABELS[t]}
                </Label>
              </div>
            ))}
          </div>
          {errors.types && (
            <p className="text-xs text-destructive">{errors.types}</p>
          )}
        </div>

        <div>
          <Button onClick={handleSubmit} disabled={fetchOdds.isPending}>
            {fetchOdds.isPending ? '取得中...' : 'オッズ取得'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export function Ingest() {
  const statusQuery = useScraperStatus();
  const runMutation = useScraperRun();
  const stopMutation = useScraperStop();
  const setRunning = useScraperStore((s) => s.setRunning);
  const trackedJobId = useScraperStore((s) => s.trackedJobId);
  const setTrackedJobId = useScraperStore((s) => s.setTrackedJobId);
  const [stopDialogOpen, setStopDialogOpen] = useState(false);

  function handleRun(req: ScraperRunRequest) {
    setRunning(true);
    runMutation.mutate(req, {
      onSuccess: (data) => {
        setTrackedJobId(data.job_id);
        toast.success(`スクレイピングを開始しました（Job ID: ${data.job_id}）`);
      },
      onError: async (err) => {
        setRunning(false);
        toast.error(`スクレイピング開始に失敗しました: ${await formatErrorMessage(err)}`);
      },
    });
  }

  function handleStop() {
    setStopDialogOpen(false);
    stopMutation.mutate(undefined, {
      onSuccess: () => {
        setRunning(false);
        toast.success('スクレイパーを停止しました');
      },
      onError: async (err) => {
        toast.error(`停止に失敗しました: ${await formatErrorMessage(err)}`);
      },
    });
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      <PageHeader
        icon={Database}
        title="Ingest"
        description="netkeiba スクレイピングの実行と進捗確認"
      >
        <IngestRunDialog onSubmit={handleRun} isPending={runMutation.isPending} />
        <Dialog open={stopDialogOpen} onOpenChange={setStopDialogOpen}>
          <DialogTrigger asChild>
            <Button variant="destructive" disabled={stopMutation.isPending}>
              即時停止
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>スクレイパー停止確認</DialogTitle>
              <DialogDescription>
                実行中のスクレイピングジョブを即時停止しますか？この操作は取り消せません。
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setStopDialogOpen(false)}>
                キャンセル
              </Button>
              <Button variant="destructive" onClick={handleStop}>
                停止する
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </PageHeader>

      {trackedJobId && (
        <JobProgressCard
          jobId={trackedJobId}
          title="ingest ジョブ進捗"
          onDismiss={() => setTrackedJobId(null)}
        />
      )}

      {/* 当日取込用カード群 */}
      <ShutubaIngestCard />
      <LiveOddsCard />

      {statusQuery.isPending ? (
        <Skeleton className="h-40 w-full rounded-lg" />
      ) : statusQuery.isError ? (
        <EmptyState
          message="スクレイパー状態の取得に失敗しました"
          description="バックエンドが起動しているか確認してください。"
        />
      ) : (
        <ScraperStatusCard status={statusQuery.data} />
      )}
    </div>
  );
}
