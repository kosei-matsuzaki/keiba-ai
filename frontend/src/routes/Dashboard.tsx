import { useMetricsSummary } from '@/hooks/useMetricsSummary';
import { useMetricsTimeseries } from '@/hooks/useMetricsTimeseries';
import { useModels } from '@/hooks/useModels';
import { MetricCard } from '@/components/MetricCard';
import { ActiveModelCard } from '@/components/ActiveModelCard';
import { AccuracyChart } from '@/components/AccuracyChart';
import { EmptyState } from '@/components/EmptyState';
import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

function MetricsSkeleton() {
  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <Skeleton key={i} className="h-28 rounded-lg" />
      ))}
    </div>
  );
}

export function Dashboard() {
  const summary = useMetricsSummary();
  const timeseries = useMetricsTimeseries('ndcg3', '180d');
  const modelsQuery = useModels();
  const activeModel = modelsQuery.data?.find((m) => m.is_active) ?? null;

  return (
    <div className="flex flex-col gap-6 p-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>

      {/* Active model summary — clickable, jumps to Models page */}
      {modelsQuery.isPending ? (
        <Skeleton className="h-24 w-full rounded-lg" />
      ) : (
        <ActiveModelCard model={activeModel} />
      )}

      {/* Metric summary cards */}
      {summary.isPending ? (
        <MetricsSkeleton />
      ) : summary.isError ? (
        <EmptyState
          message="メトリクス取得に失敗しました"
          description="バックエンドが起動しているか確認してください。"
        />
      ) : (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <MetricCard
            title="NDCG@3"
            value={summary.data.ndcg3}
            format="decimal"
            description="直近 active モデル"
          />
          <MetricCard
            title="Top-1 ヒット率"
            value={summary.data.top1_hit}
            format="percent"
            description="1着予想的中率"
          />
          <MetricCard
            title="複勝的中率"
            value={summary.data.place_hit}
            format="percent"
            description="3着以内的中率"
          />
          <MetricCard
            title="単勝回収率"
            value={summary.data.payback_win}
            format="ratio"
            description="payback_win"
          />
        </div>
      )}

      {/* Timeseries chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">NDCG@3 推移（直近 180 日）</CardTitle>
        </CardHeader>
        <CardContent>
          {timeseries.isPending ? (
            <Skeleton className="h-60 w-full" />
          ) : timeseries.isError ? (
            <EmptyState message="チャートデータ取得に失敗しました" />
          ) : (
            <AccuracyChart points={timeseries.data.points} metricLabel="NDCG@3" />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
