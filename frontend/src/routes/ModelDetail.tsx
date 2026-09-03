import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

import { fetchModel } from '@/lib/api';
import { useActivateModel } from '@/hooks/useActivateModel';
import { MetricBand, MetricItem } from '@/components/MetricBand';
import { ModelSimulationPanel } from '@/components/ModelSimulationPanel';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from '@/lib/toast';
import { formatDateTime, formatScore } from '@/lib/formatters';
import { formatErrorMessage } from '@/lib/api';
import {
  placeHitLabel,
  readModelMeta,
} from '@/lib/modelMetrics';
import type { ModelMeta } from '@/types/api';

const PLACEHOLDER = '—';

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-sm font-medium">{value}</span>
    </div>
  );
}

/** モデル 1 件の成績。出所・買い方・評価窓を数字と一緒に出す。 */
function ModelScoreBand({ model }: { model: ModelMeta }) {
  const m = readModelMeta(model);
  const edge =
    m.logLoss != null && m.marketLogLoss != null ? m.logLoss - m.marketLogLoss : null;

  if (m.source === null) {
    return (
      <EmptyState
        message="評価がまだ走っていません"
        description="下のバックテストを実行すると成績が出ます"
      />
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <MetricBand cols={5}>
        <MetricItem
          title="単勝回収率"
          value={m.paybackWin}
          format="ratio"
          tone={m.paybackWin != null && m.paybackWin >= 1 ? 'positive' : 'negative'}
          description="1.00 = 収支トントン"
        />
        <MetricItem
          title="複勝回収率"
          value={m.paybackPlace}
          format="ratio"
          tone={m.paybackPlace != null && m.paybackPlace >= 1 ? 'positive' : 'negative'}
          description="1.00 = 収支トントン"
        />
        <MetricItem
          title="本命の的中率"
          value={m.top1Hit}
          format="percent"
          description="予想1位が1着"
          hint="的中率が高いほど儲かるとは限らない。人気馬を選べば当たるが配当が小さい"
        />
        <MetricItem
          title="複勝的中率"
          value={m.placeHit}
          format="percent"
          description={placeHitLabel(m.source)}
          hint="出所で別の量になる (実測は予想1位が3着以内、学習時は上位3頭のうち1頭以上)"
        />
        <MetricItem
          title="確率の質 (log-loss)"
          value={m.logLoss}
          format="decimal"
          tone={edge != null && edge < 0 ? 'positive' : 'default'}
          description={
            m.marketLogLoss != null
              ? `市場 ${formatScore(m.marketLogLoss)} / 差 ${edge != null && edge < 0 ? '−' : '+'}${formatScore(Math.abs(edge ?? 0))}`
              : '小さいほど正確'
          }
          hint="本命についての二値 log-loss。市場 (1/オッズ) を下回れないモデルが市場より systematically に儲けることは原理的にできない"
        />
      </MetricBand>

    </div>
  );
}

export function ModelDetail() {
  const params = useParams<{ model_id: string }>();
  const modelId = Number(params.model_id);
  const activateMutation = useActivateModel();

  const modelQuery = useQuery<ModelMeta>({
    queryKey: ['models', modelId],
    queryFn: () => fetchModel(modelId),
    enabled: Number.isFinite(modelId),
  });

  function handleActivate() {
    activateMutation.mutate(modelId, {
      onSuccess: () => toast.success(`モデル ${modelId} をアクティブにしました`),
      onError: async (err) =>
        toast.error(`Activate に失敗しました: ${await formatErrorMessage(err)}`),
    });
  }

  if (!Number.isFinite(modelId)) {
    return (
      <div className="p-6">
        <EmptyState message="不正なモデル ID です" />
      </div>
    );
  }

  const model = modelQuery.data ?? null;
  const title = model?.name?.trim() ? model.name : `モデル ${modelId}`;

  return (
    <div className="flex flex-col gap-8 p-6">
      <PageHeader eyebrow="Model Detail" title={title}>
        <Button variant="outline" size="sm" asChild>
          <Link to="/">
            <ArrowLeft className="mr-1.5 h-4 w-4" />
            一覧へ
          </Link>
        </Button>
      </PageHeader>

      {/* モデルメタ */}
      {modelQuery.isPending ? (
        <Skeleton className="h-32 w-full rounded-sm" />
      ) : modelQuery.isError || !model ? (
        <EmptyState
          message="モデルが見つかりません"
          description="削除済みか、ID が不正な可能性があります。"
        />
      ) : (
        <>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
              <CardTitle className="flex items-center gap-3 text-base">
                {title}
                {/* 役割は 2 つある: Active = 買い目を決める / 確率 = 確からしさを出す。
                    兼務もありうるので併記する。 */}
                {model.is_active ? (
                  <Badge tone="success">Active</Badge>
                ) : (
                  !model.is_probability_model && <Badge variant="outline">非アクティブ</Badge>
                )}
                {model.is_probability_model && (
                  <Badge title="複勝の確信度と連系の確率に使われています">確率</Badge>
                )}
              </CardTitle>
              {!model.is_active && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleActivate}
                  disabled={activateMutation.isPending}
                >
                  {activateMutation.isPending ? '切り替え中…' : 'Activate'}
                </Button>
              )}
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
              <MetaRow
                label="役割"
                value={
                  [
                    model.is_active ? '買い目を決める' : null,
                    model.is_probability_model ? '確からしさを出す' : null,
                  ]
                    .filter(Boolean)
                    .join(' / ') || '未使用'
                }
              />
              <MetaRow label="ID" value={String(model.id)} />
              <MetaRow label="作成日時" value={formatDateTime(model.created_at)} />
              <MetaRow label="学習期間" value={model.train_range ?? PLACEHOLDER} />
              <MetaRow label="検証期間" value={model.valid_range ?? PLACEHOLDER} />
              <MetaRow label="評価窓" value={readModelMeta(model).evalRange ?? PLACEHOLDER} />
            </CardContent>
          </Card>

          {/* 成績。**回収率を先に、順位精度は下に小さく。** 順位精度は上げても
              回収率が上がらないことが実測で分かっているので、判断に使う数字を上に置く。 */}
          <ModelScoreBand model={model} />

          {/* このモデルのバックテスト */}
          <ModelSimulationPanel modelId={modelId} />
        </>
      )}
    </div>
  );
}
