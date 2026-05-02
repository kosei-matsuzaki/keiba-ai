import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useScraperRecentActivity } from '@/hooks/useScraperRecentActivity';
import { formatCount, formatDateTime } from '@/lib/formatters';
import type { ScraperStatus } from '@/types/api';

interface ScraperStatusCardProps {
  status: ScraperStatus;
}

function StatusBadge({ status }: { status: ScraperStatus }) {
  if (status.current_job_id) {
    return <Badge className="bg-blue-600 text-white">実行中</Badge>;
  }
  if (status.stopped) {
    return <Badge variant="destructive">停止中</Badge>;
  }
  return <Badge variant="secondary">アイドル</Badge>;
}

export function ScraperStatusCard({ status }: ScraperStatusCardProps) {
  const recent = useScraperRecentActivity(10);

  // CLI 経由で ingest_range が走っているかの簡易判定: UI ジョブ無しで
  // 直近 10 分に ok 取得が 1 件以上 → CLI 進行中とみなす。
  const cliActive =
    !status.current_job_id && (recent.data?.ok_count ?? 0) > 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-3 text-base">
          スクレイパー状態
          <StatusBadge status={status} />
          {cliActive && <Badge className="bg-amber-600 text-white">CLI 進行中</Badge>}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <Row label="最終取得日" value={status.last_fetched_date ?? '未取得'} />
        <Row
          label="現在のジョブ ID"
          value={
            status.current_job_id ? (
              <span className="font-mono text-xs">{status.current_job_id}</span>
            ) : (
              '—'
            )
          }
        />

        {recent.data && (
          <>
            <div className="border-border/60 my-2 border-t" />
            <Row
              label={`直近 ${recent.data.window_minutes} 分`}
              value={
                <>
                  {formatCount(recent.data.total_fetched)} fetch
                  {' '}(<span className="text-emerald-700">ok {formatCount(recent.data.ok_count)}</span>
                  {recent.data.error_count > 0 && (
                    <>{' / '}<span className="text-destructive">err {formatCount(recent.data.error_count)}</span></>
                  )}
                  {recent.data.skipped_count > 0 && (
                    <>{' / '}<span className="text-muted-foreground">skip {formatCount(recent.data.skipped_count)}</span></>
                  )})
                </>
              }
            />
            <Row
              label="スループット"
              value={`${recent.data.rate_per_min.toFixed(1)} 件/分`}
            />
            {recent.data.latest_race_id && (
              <Row
                label="最新 race_id"
                value={<span className="font-mono text-xs">{recent.data.latest_race_id}</span>}
              />
            )}
            {recent.data.latest_fetched_at && (
              <Row
                label="最新 fetch 時刻"
                value={formatDateTime(recent.data.latest_fetched_at)}
              />
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-muted-foreground w-36">{label}:</span>
      <span>{value}</span>
    </div>
  );
}
