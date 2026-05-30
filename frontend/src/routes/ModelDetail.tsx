import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, Brain } from 'lucide-react';

import { fetchModel } from '@/lib/api';
import { useActivateModel } from '@/hooks/useActivateModel';
import { ModelSimulationPanel } from '@/components/ModelSimulationPanel';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from '@/components/ui/toast';
import { formatDateTime, formatRatio, formatScore } from '@/lib/formatters';
import { formatErrorMessage } from '@/lib/api';
import type { ModelMeta } from '@/types/api';

const PLACEHOLDER = '—';

function metric(
  metrics: Record<string, unknown> | null,
  key: string,
  fmt: 'score' | 'ratio',
): string {
  if (!metrics) return PLACEHOLDER;
  const v = metrics[key];
  if (typeof v !== 'number') return PLACEHOLDER;
  return fmt === 'score' ? formatScore(v) : formatRatio(v);
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-sm font-medium">{value}</span>
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
    <div className="flex flex-col gap-6 p-6">
      <PageHeader icon={Brain} title={title} description="モデル詳細とバックテスト">
        <Button variant="outline" size="sm" asChild>
          <Link to="/models">
            <ArrowLeft className="mr-1.5 h-4 w-4" />
            一覧へ
          </Link>
        </Button>
      </PageHeader>

      {/* モデルメタ */}
      {modelQuery.isPending ? (
        <Skeleton className="h-32 w-full rounded-lg" />
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
                {model.is_active ? (
                  <Badge variant="success">Active</Badge>
                ) : (
                  <Badge variant="outline">非アクティブ</Badge>
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
              <MetaRow label="ID" value={String(model.id)} />
              <MetaRow label="作成日時" value={formatDateTime(model.created_at)} />
              <MetaRow label="学習期間" value={model.train_range ?? PLACEHOLDER} />
              <MetaRow label="検証期間" value={model.valid_range ?? PLACEHOLDER} />
              <MetaRow label="NDCG@3" value={metric(model.metrics, 'ndcg3', 'score')} />
              <MetaRow
                label="単勝回収率"
                value={metric(model.metrics, 'payback_win', 'ratio')}
              />
            </CardContent>
          </Card>

          {/* このモデルのバックテスト */}
          <ModelSimulationPanel modelId={modelId} />
        </>
      )}
    </div>
  );
}
