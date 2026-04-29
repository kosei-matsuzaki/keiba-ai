import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
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
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-3 text-base">
          スクレイパー状態
          <StatusBadge status={status} />
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground w-36">最終取得日:</span>
          <span>{status.last_fetched_date ?? '未取得'}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground w-36">現在のジョブ ID:</span>
          <span className="font-mono text-xs">{status.current_job_id ?? '—'}</span>
        </div>
      </CardContent>
    </Card>
  );
}
