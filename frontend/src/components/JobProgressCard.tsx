import type { ReactNode } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useJobStatus } from '@/hooks/useJobStatus';
import { formatDateTime } from '@/lib/formatters';
import type { JobInfo } from '@/types/api';

interface JobProgressCardProps {
  jobId: string | null;
  /** Called when user clicks dismiss (clears the tracked job_id) */
  onDismiss?: () => void;
  /** Optional title — defaults to "ジョブ進捗" */
  title?: string;
}

const TERMINAL_STATUSES = new Set(['success', 'failed', 'cancelled']);

function statusBadge(status: string) {
  switch (status) {
    case 'running':
      return <Badge variant="info">実行中</Badge>;
    case 'success':
      return <Badge variant="success">完了</Badge>;
    case 'failed':
      return <Badge variant="destructive">失敗</Badge>;
    case 'cancelled':
      return <Badge variant="secondary">中断</Badge>;
    case 'pending':
      return <Badge variant="outline">待機中</Badge>;
    default:
      return <Badge variant="outline">{status}</Badge>;
  }
}

function elapsedSeconds(job: JobInfo): number {
  const start = Date.parse(job.started_at);
  const end = job.finished_at ? Date.parse(job.finished_at) : Date.now();
  if (Number.isNaN(start) || Number.isNaN(end)) return 0;
  return Math.max(0, Math.round((end - start) / 1000));
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds} 秒`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `${m} 分 ${s} 秒`;
  const h = Math.floor(m / 60);
  return `${h} 時間 ${m % 60} 分`;
}

export function JobProgressCard({ jobId, onDismiss, title = 'ジョブ進捗' }: JobProgressCardProps) {
  const query = useJobStatus(jobId);

  if (!jobId) return null;

  const job = query.data;
  const isTerminal = job ? TERMINAL_STATUSES.has(job.status) : false;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-3 text-base">
          {title}
          {job && statusBadge(job.status)}
          {isTerminal && onDismiss && (
            <Button size="sm" variant="ghost" onClick={onDismiss} className="ml-auto h-7">
              閉じる
            </Button>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {query.isPending && !job && (
          <p className="text-muted-foreground">ジョブ情報を取得中…</p>
        )}
        {query.isError && !job && (
          <p className="text-muted-foreground">
            ジョブが見つかりませんでした (id: <span className="font-mono text-xs">{jobId}</span>)
          </p>
        )}
        {job && (
          <>
            <Row label="ジョブ ID" value={<span className="font-mono text-xs">{job.job_id}</span>} />
            <Row label="種別" value={job.type} />
            <Row label="開始時刻" value={formatDateTime(job.started_at)} />
            {job.finished_at && (
              <Row label="終了時刻" value={formatDateTime(job.finished_at)} />
            )}
            <Row label="経過時間" value={formatElapsed(elapsedSeconds(job))} />
            {job.error && (
              <div className="rounded border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive">
                {job.error}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-muted-foreground w-28">{label}:</span>
      <span>{value}</span>
    </div>
  );
}
