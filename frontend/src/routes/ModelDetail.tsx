import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

import { fetchModel } from '@/lib/api';
import { useActivateModel } from '@/hooks/useActivateModel';
import { SectionHeading } from '@/components/SectionHeading';
import { Figures } from '@/components/Figures';
import { MetricCard } from '@/components/MetricCard';
import { ModelSimulationPanel } from '@/components/ModelSimulationPanel';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from '@/lib/toast';
import { formatDateTime, formatPercent, formatRatio, formatScore } from '@/lib/formatters';
import { formatErrorMessage } from '@/lib/api';
import {
  inSampleWarning,
  placeHitLabel,
  readModelMeta,
  roiNote,
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
      {/* **カードにするのは答えだけ。**このモデルで買って勝てるかは単勝回収率
          1 つで決まり、残りはその読み方を支える数字 (収支台帳・シミュレーション
          の結果・モデル画面と同じ組み立て)。 */}
      <div className="flex flex-wrap items-start gap-6">
        <MetricCard
          className="min-w-[11rem]"
          label="単勝回収率"
          value={shown(m.paybackWin, formatRatio)}
          tone={m.paybackWin != null && m.paybackWin >= 1 ? 'positive' : 'negative'}
          hint="本命に単勝を買い続けたときの払戻 ÷ 投資。1.00 = トントンで、控除率 20% があるので 1.0 未満は平均で負け越し。"
          note={roiNote(m.paybackWinCi, m.nRaces)}
        />
        <Figures
          className="pt-1"
          items={[
            {
              label: '複勝回収率',
              value: shown(m.paybackPlace, formatRatio),
              paren: m.paybackPlaceCi
                ? `95% ${m.paybackPlaceCi[0].toFixed(2)}–${m.paybackPlaceCi[1].toFixed(2)}`
                : undefined,
            },
            {
              label: '本命の的中率',
              value: shown(m.top1Hit, formatPercent),
              hint: '予想 1 位が 1 着になった割合。的中率が高いほど儲かるとは限らない — 人気馬を選べば当たるが配当が小さい。',
            },
            {
              label: '複勝的中率',
              value: shown(m.placeHit, formatPercent),
              hint: `${placeHitLabel(m.source)}だった割合。実測と学習時で数え方が違うので、出所と一緒に読む。`,
            },
            {
              label: 'log-loss',
              value: shown(m.logLoss, formatScore),
              paren:
                m.marketLogLoss != null ? `市場 ${formatScore(m.marketLogLoss)}` : undefined,
              hint: '本命についての二値 log-loss。小さいほど確率として正確。市場 (1/オッズ) を下回れないモデルが市場より systematically に儲けることは原理的にできない。',
            },
          ]}
        />
      </div>

    </div>
  );
}

/** 未算出をひとところに寄せる。'未算出' の文字列が散らないように。 */
function shown(value: number | null | undefined, format: (n: number) => string): string {
  return value == null ? '未算出' : format(value);
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
              <SectionHeading className="gap-3">
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
              </SectionHeading>
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
              {inSampleWarning(readModelMeta(model)) && (
                <p className="pt-1 text-xs text-destructive">
                  {inSampleWarning(readModelMeta(model))}
                </p>
              )}
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
