import { useScraperStatus } from '@/hooks/useScraperStatus';
import { useScraperRun } from '@/hooks/useScraperRun';
import { useScraperStop } from '@/hooks/useScraperStop';
import { useScraperStore } from '@/store/app';
import { ScraperStatusCard } from '@/components/ScraperStatusCard';
import { IngestRunDialog } from '@/components/IngestRunDialog';
import { EmptyState } from '@/components/EmptyState';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
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
import type { ScraperRunRequest } from '@/types/api';
import { useState } from 'react';

export function Ingest() {
  const statusQuery = useScraperStatus();
  const runMutation = useScraperRun();
  const stopMutation = useScraperStop();
  const setRunning = useScraperStore((s) => s.setRunning);
  const [stopDialogOpen, setStopDialogOpen] = useState(false);

  function handleRun(req: ScraperRunRequest) {
    setRunning(true);
    runMutation.mutate(req, {
      onSuccess: (data) => {
        toast.success(`スクレイピングを開始しました（Job ID: ${data.job_id}）`);
      },
      onError: (err) => {
        setRunning(false);
        toast.error(`スクレイピング開始に失敗しました: ${(err as Error).message}`);
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
      onError: (err) => {
        toast.error(`停止に失敗しました: ${(err as Error).message}`);
      },
    });
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Ingest</h1>
        <div className="flex gap-2">
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
        </div>
      </div>

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
