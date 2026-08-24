import { BarChart3, AlertTriangle } from 'lucide-react';
import { useState } from 'react';
import { Link } from 'react-router-dom';

import { useMetricsSummary } from '@/hooks/useMetricsSummary';
import { useMetricsTimeseries } from '@/hooks/useMetricsTimeseries';
import { useModels } from '@/hooks/useModels';
import { useThisWeekendRaces } from '@/hooks/useThisWeekendRaces';
import { MetricBand, MetricItem } from '@/components/MetricBand';
import { ActiveModelCard } from '@/components/ActiveModelCard';
import { AccuracyChart } from '@/components/AccuracyChart';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { MetricsSummary } from '@/types/api';

function MetricsSkeleton() {
  return (
    <div className="grid grid-cols-2 border-y border-border lg:grid-cols-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="px-5 py-5">
          <Skeleton className="h-16 w-full rounded-sm" />
        </div>
      ))}
    </div>
  );
}

/**
 * いまの状態を 1 行で出す帯 (B-2 ①)。
 *
 * 指標が並んでいるだけだと「次に何をすればいいか」が分からない。
 * 週末のレースが未取得でも、モデルが無くても、画面が同じ顔をしてしまう。
 * **問題がないときは何も出さない**のが要点 (常時出ていると情報にならない)。
 */
function StatusBand({
  hasActiveModel,
  weekendRaceCount,
  isLoading,
}: {
  hasActiveModel: boolean;
  weekendRaceCount: number | null;
  isLoading: boolean;
}) {
  if (isLoading) return null;

  const issue = !hasActiveModel
    ? {
        message: '有効なモデルがありません',
        detail: 'モデルを学習すると予想と評価が動きます。',
        to: '/models',
        action: '学習する',
      }
    : weekendRaceCount === 0
      ? {
          message: '今週末のレースがまだ取り込まれていません',
          detail: 'レース一覧から出馬表を取得できます。',
          to: '/races',
          action: '取得する',
        }
      : null;

  if (!issue) return null;

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-y border-warning/30 bg-warning/[0.06] px-4 py-3">
      <AlertTriangle className="h-4 w-4 shrink-0 text-warning" aria-hidden="true" />
      <span className="text-sm font-medium">{issue.message}</span>
      <span className="text-sm text-muted-foreground">{issue.detail}</span>
      <Button asChild size="sm" variant="outline" className="ml-auto">
        <Link to={issue.to}>{issue.action}</Link>
      </Button>
    </div>
  );
}

/** 4 指標がすべて未算出なら「まだ評価していない」= カードを並べる意味がない。 */
function hasAnyMetric(summary: MetricsSummary): boolean {
  return [summary.ndcg3, summary.top1_hit, summary.place_hit, summary.payback_win].some(
    (v) => typeof v === 'number' && Number.isFinite(v),
  );
}

const RANGES = [
  { value: '30d', label: '30日' },
  { value: '90d', label: '90日' },
  { value: '180d', label: '180日' },
  { value: 'all', label: '全期間' },
] as const;

export function Dashboard() {
  const [range, setRange] = useState<(typeof RANGES)[number]['value']>('180d');
  const summary = useMetricsSummary();
  const timeseries = useMetricsTimeseries('ndcg3', range);
  const modelsQuery = useModels();
  const weekend = useThisWeekendRaces();
  const activeModel = modelsQuery.data?.find((m) => m.is_active) ?? null;

  return (
    <div className="flex flex-col gap-12 p-6">
      <PageHeader
        eyebrow="Dashboard"
        title="モデル成績の概観"
        description="直近の指標推移と active モデルの状態"
      />

      <StatusBand
        hasActiveModel={activeModel != null}
        weekendRaceCount={weekend.data?.races.length ?? null}
        isLoading={modelsQuery.isPending || weekend.isPending}
      />

      {/* Active model summary — clickable, jumps to Models page */}
      {modelsQuery.isPending ? (
        <Skeleton className="h-24 w-full rounded-sm" />
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
      ) : !hasAnyMetric(summary.data) ? (
        // 4 枚とも未算出のときは「—」だらけのカードを並べず、1 枚の空状態にする。
        <div className="border-y border-border">
          <EmptyState
            icon={BarChart3}
            message="評価結果がまだありません"
            description="モデルを学習して評価を実行すると、ここに指標が出ます。"
          >
            <Button asChild>
              <Link to="/models">Models 画面へ</Link>
            </Button>
          </EmptyState>
        </div>
      ) : (
        <MetricBand cols={4}>
          <MetricItem
            title="NDCG@3"
            value={summary.data.ndcg3}
            format="decimal"
            description="直近 active モデル"
            hint="上位3頭の並びの正確さ（1.0 が完全一致）。回収率とは無関係に上がる点に注意"
            to="/models"
          />
          <MetricItem
            title="Top-1 ヒット率"
            value={summary.data.top1_hit}
            format="percent"
            description="1着予想的中率"
            hint="予想1位の馬が実際に1着だった割合"
            to="/models"
          />
          <MetricItem
            title="複勝的中率"
            value={summary.data.place_hit}
            format="percent"
            description="3着以内的中率"
            hint="予想上位3頭のうち1頭以上が3着以内に入ったレースの割合"
            to="/models"
          />
          <MetricItem
            title="単勝回収率"
            value={summary.data.payback_win}
            format="ratio"
            tone={
              summary.data.payback_win != null && summary.data.payback_win >= 1
                ? 'positive'
                : 'negative'
            }
            description="1.00 = 収支トントン"
            hint="期待値が基準を超えた馬に単勝を買ったときの払戻 ÷ 投資。1.0 未満は平均で負け越し"
            to="/ledger"
          />
        </MetricBand>
      )}

      {/* Timeseries chart — 箱に入れず、上端の罫線だけで区切る */}
      <Card className="border-t border-border pt-6">
        <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
          <CardTitle className="text-label-ja">NDCG@3 推移</CardTitle>
          <div className="flex gap-1">
            {RANGES.map((r) => (
              <Button
                key={r.value}
                variant={range === r.value ? 'soft' : 'ghost'}
                size="sm"
                className="h-7 px-2"
                onClick={() => setRange(r.value)}
              >
                {r.label}
              </Button>
            ))}
          </div>
        </CardHeader>
        <CardContent>
          {timeseries.isPending ? (
            <Skeleton className="h-60 w-full rounded-sm" />
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
