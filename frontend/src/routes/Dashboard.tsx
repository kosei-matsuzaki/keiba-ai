import { AlertTriangle } from 'lucide-react';
import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import { useActivateModel } from '@/hooks/useActivateModel';
import { useCompactModelIds } from '@/hooks/useCompactModelIds';
import { useDeleteModel } from '@/hooks/useDeleteModel';
import { useEvaluateModel } from '@/hooks/useEvaluateModel';
import { useMetricsSummary } from '@/hooks/useMetricsSummary';
import { useModels } from '@/hooks/useModels';
import { useTrainModel } from '@/hooks/useTrainModel';
import { useUpdateModel } from '@/hooks/useUpdateModel';
import { useUpdateSettings } from '@/hooks/useSettings';
import { DeleteModelDialog } from '@/components/DeleteModelDialog';
import { EditModelNameDialog } from '@/components/EditModelNameDialog';
import { EmptyState } from '@/components/EmptyState';
import { HelpDot } from '@/components/HelpDot';
import { JobProgressCard } from '@/components/JobProgressCard';
import { ModelTable } from '@/components/ModelTable';
import { OperatingModels } from '@/components/OperatingModels';
import { PageHeader } from '@/components/PageHeader';
import { TrainModelDialog } from '@/components/TrainModelDialog';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from '@/components/ui/toast';
import { formatErrorMessage } from '@/lib/api';
import { useTrainingStore } from '@/store/app';
import type { ModelMeta, TrainRequest } from '@/types/api';

/**
 * active モデルが無いことだけを知らせる帯。
 *
 * **問題がないときは何も出さない** (常時出ていると情報にならない)。
 * 週末のレースの取り込み漏れは、取り込む場所である RACE 画面に出す
 * (`WeekendIngestNotice`)。知らせと操作が別画面にあると動線が伸びる。
 */
function StatusBand({
  hasActiveModel,
  isLoading,
}: {
  hasActiveModel: boolean;
  isLoading: boolean;
}) {
  if (isLoading || hasActiveModel) return null;

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-y border-warning/30 bg-warning/[0.06] px-4 py-3">
      <AlertTriangle className="h-4 w-4 shrink-0 text-warning" aria-hidden="true" />
      <span className="text-sm font-medium">有効なモデルがありません</span>
      <span className="text-sm text-muted-foreground">
        下の一覧から Activate するか、新しく学習してください。
      </span>
    </div>
  );
}

/**
 * モデルに関する操作をすべて持つ 1 画面。
 *
 * 成績 (KPI)・比較 (一覧)・学習・役割の割り当てを別画面に分けていたが、
 * **見比べてから選ぶ**という流れが画面をまたいでいた。active を切り替えるのも
 * 確率モデルを割り当てるのも「数字を見た直後」にやることなので同じ画面に置く。
 * 個別モデルのバックテストだけは重いので詳細 (`/models/:id`) に残す。
 */
export function Dashboard() {
  const summary = useMetricsSummary();
  const modelsQuery = useModels();
  const queryClient = useQueryClient();

  const activateMutation = useActivateModel();
  // 確率モデルの割り当ては settings に持つが、操作はモデルを見比べるこの画面で行う。
  const updateSettings = useUpdateSettings();
  const updateMutation = useUpdateModel();
  const deleteMutation = useDeleteModel();
  const compactMutation = useCompactModelIds();
  const evaluateMutation = useEvaluateModel();
  const trainMutation = useTrainModel();
  const trackedJobId = useTrainingStore((s) => s.trackedJobId);
  const setTrackedJobId = useTrainingStore((s) => s.setTrackedJobId);
  const [activatingId, setActivatingId] = useState<number | null>(null);
  const [evaluatingId, setEvaluatingId] = useState<number | null>(null);
  const [editTarget, setEditTarget] = useState<ModelMeta | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ModelMeta | null>(null);

  const activeModel = modelsQuery.data?.find((m) => m.is_active) ?? null;

  function handleActivate(id: number) {
    setActivatingId(id);
    activateMutation.mutate(id, {
      onSuccess: () => {
        toast.success(`モデル ${id} をアクティブにしました`);
        setActivatingId(null);
      },
      onError: async (err) => {
        toast.error('Activate に失敗しました', {
          description: await formatErrorMessage(err),
          action: { label: '再試行', onClick: () => handleActivate(id) },
        });
        setActivatingId(null);
      },
    });
  }

  /** 確率モデルを割り当てる / 解除する。model=null で未設定に戻す。 */
  function handleSetProbability(model: ModelMeta | null) {
    updateSettings.mutate(
      { probability_model_path: model ? model.model_path : null },
      {
        onSuccess: () => {
          void queryClient.invalidateQueries({ queryKey: ['models'] });
          toast.success(
            model
              ? `ID ${model.id} を確率モデルにしました（複勝の確信度と連系の確率に使われます）`
              : '確率モデルの割り当てを解除しました'
          );
        },
        onError: async (err) => {
          toast.error('設定の更新に失敗しました', {
            description: await formatErrorMessage(err),
          });
        },
      }
    );
  }

  function handleEditSubmit(id: number, name: string | null) {
    updateMutation.mutate(
      { id, body: { name } },
      {
        onSuccess: () => {
          toast.success(`モデル ${id} の名称を更新しました`);
          setEditTarget(null);
        },
        onError: async (err) => {
          toast.error(`名称更新に失敗しました: ${await formatErrorMessage(err)}`);
        },
      }
    );
  }

  function handleDeleteConfirm(id: number) {
    deleteMutation.mutate(id, {
      onSuccess: () => {
        toast.success(`モデル ${id} を削除しました`);
        setDeleteTarget(null);
      },
      onError: async (err) => {
        toast.error('削除に失敗しました', {
          description: await formatErrorMessage(err),
          action: { label: '再試行', onClick: () => handleDeleteConfirm(id) },
        });
      },
    });
  }

  /** 実運用の賭けルールで測り直す。進捗は JobProgressCard が拾う。 */
  function handleEvaluate(model: ModelMeta) {
    setEvaluatingId(model.id);
    evaluateMutation.mutate(model.id, {
      onSuccess: (data) => {
        setTrackedJobId(data.job_id);
        setEvaluatingId(null);
        toast.success(`ID ${model.id} の計測を開始しました`, {
          description: '5,000 レース規模で 10 分前後かかります。終わると指標が入ります。',
        });
      },
      onError: async (err) => {
        setEvaluatingId(null);
        toast.error(`計測の開始に失敗しました: ${await formatErrorMessage(err)}`);
      },
    });
  }

  function handleCompact() {
    compactMutation.mutate(undefined, {
      onSuccess: () => toast.success('モデル ID を詰めました'),
      onError: async (err) => {
        toast.error(`ID 詰めに失敗しました: ${await formatErrorMessage(err)}`);
      },
    });
  }

  function handleTrain(req: TrainRequest) {
    trainMutation.mutate(req, {
      onSuccess: (data) => {
        setTrackedJobId(data.job_id);
        toast.success(`学習ジョブを受け付けました（Job ID: ${data.job_id}）`);
      },
      onError: async (err) => {
        toast.error(`再学習に失敗しました: ${await formatErrorMessage(err)}`);
      },
    });
  }

  return (
    <div className="flex flex-col gap-10 p-6">
      <PageHeader eyebrow="Dashboard" title="モデル" />

      <StatusBand
        hasActiveModel={activeModel != null}
        isLoading={modelsQuery.isPending}
      />

      {trackedJobId && (
        <JobProgressCard
          jobId={trackedJobId}
          title="ジョブ進捗 (学習 / 計測)"
          onDismiss={() => setTrackedJobId(null)}
        />
      )}

      {/* ── 塊 1: いま動いているもの ──────────────────────────────
          運用中の 2 モデルと、その数字。**役割カードと KPI 帯を分けない** —
          分けると同じ active の回収率が上下 2 箇所に出て、どのモデルの数字か
          読み取れなくなる。左は利用者が得る回収率、右は確率としての正しさ。 */}
      <section aria-label="運用中" className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <h2 className="text-label-ja">いま予想に使っているモデル</h2>
          <HelpDot
            label="いま予想に使っているモデル"
            text="買い目を決める側 (active) と、確からしさを出す側 (確率モデル) の 2 つで動きます。買う馬を決めるのは前者、複勝を買うかの判定と連系の確率は後者です。"
          />
        </div>
        {summary.isError ? (
          <EmptyState
            message="メトリクス取得に失敗しました"
            description="バックエンドが起動しているか確認してください。"
          />
        ) : modelsQuery.isPending || summary.isPending ? (
          <Skeleton className="h-48 w-full rounded-sm" />
        ) : (
          <OperatingModels models={modelsQuery.data} summary={summary.data} />
        )}
      </section>

      {/* ── 塊 2: 手持ちのモデル ──────────────────────────────────
          一覧と、一覧に対する操作 (学習で増やす / ID を詰める) を同じ塊に置く。
          ページ見出しの横に置いていたときは、何に対する操作か分からなかった。

          **推移グラフを置かないのは、モデルごとに評価窓が違うから。**
          学習の --train-end を変えれば test 期間も動くので、時系列に並べても
          「良くなった / 悪くなった」は読めない。窓を列で見せる。 */}
      <section aria-label="モデル一覧" className="flex flex-col gap-3">
        <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-2">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <h2 className="text-label-ja">手持ちのモデル</h2>
            <HelpDot
              label="手持ちのモデル"
              text="役割の割り当て (Activate / 確率に設定)・実運用の賭けルールでの測り直し (計測)・名称編集・削除はこの表から行います。"
            />
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleCompact}
              disabled={compactMutation.isPending}
              title="ModelRun.id を作成日時順に 1..N に詰める"
            >
              {compactMutation.isPending ? 'ID 詰め中…' : 'ID を詰める'}
            </Button>
            <TrainModelDialog onSubmit={handleTrain} isPending={trainMutation.isPending} />
          </div>
        </div>
        {modelsQuery.isPending ? (
          <Skeleton className="h-64 w-full rounded-sm" />
        ) : modelsQuery.isError ? (
          <EmptyState
            message="モデル情報の取得に失敗しました"
            description="バックエンドが起動しているか確認してください。"
          />
        ) : modelsQuery.data.length === 0 ? (
          <EmptyState
            message="学習済みモデルはありません"
            description="「再学習を実行」ボタンから最初のモデルを学習してください。"
          />
        ) : (
          <ModelTable
              models={modelsQuery.data}
              onActivate={handleActivate}
              onEvaluate={handleEvaluate}
              onSetProbability={handleSetProbability}
              onEdit={setEditTarget}
              onDelete={setDeleteTarget}
              activatingId={activatingId}
              settingProbability={updateSettings.isPending}
            evaluatingId={evaluatingId}
          />
        )}
      </section>

      <EditModelNameDialog
        open={editTarget !== null}
        onOpenChange={(o) => !o && setEditTarget(null)}
        modelId={editTarget?.id ?? null}
        currentName={editTarget?.name ?? null}
        onSubmit={handleEditSubmit}
        isPending={updateMutation.isPending}
      />
      <DeleteModelDialog
        open={deleteTarget !== null}
        onOpenChange={(o) => !o && setDeleteTarget(null)}
        modelId={deleteTarget?.id ?? null}
        modelName={deleteTarget?.name ?? null}
        onConfirm={handleDeleteConfirm}
        isPending={deleteMutation.isPending}
      />
    </div>
  );
}
