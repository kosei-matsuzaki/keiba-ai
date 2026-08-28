import { useState } from 'react';

import { useModels } from '@/hooks/useModels';
import { useQueryClient } from '@tanstack/react-query';

import { useActivateModel } from '@/hooks/useActivateModel';
import { useUpdateSettings } from '@/hooks/useSettings';
import { useUpdateModel } from '@/hooks/useUpdateModel';
import { useDeleteModel } from '@/hooks/useDeleteModel';
import { useCompactModelIds } from '@/hooks/useCompactModelIds';
import { useTrainModel } from '@/hooks/useTrainModel';
import { OperatingModelsCard } from '@/components/OperatingModelsCard';
import { ModelTable } from '@/components/ModelTable';
import { TrainModelDialog } from '@/components/TrainModelDialog';
import { EditModelNameDialog } from '@/components/EditModelNameDialog';
import { DeleteModelDialog } from '@/components/DeleteModelDialog';
import { JobProgressCard } from '@/components/JobProgressCard';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from '@/components/ui/toast';
import { formatErrorMessage } from '@/lib/api';
import { useTrainingStore } from '@/store/app';
import type { ModelMeta, TrainRequest } from '@/types/api';

export function Models() {
  const modelsQuery = useModels();
  const queryClient = useQueryClient();
  const activateMutation = useActivateModel();
  // 確率モデルの割り当ては settings に持つが、操作はモデルを見比べるこの画面で行う。
  const updateSettings = useUpdateSettings();
  const updateMutation = useUpdateModel();
  const deleteMutation = useDeleteModel();
  const compactMutation = useCompactModelIds();
  const trainMutation = useTrainModel();
  const trackedJobId = useTrainingStore((s) => s.trackedJobId);
  const setTrackedJobId = useTrainingStore((s) => s.setTrackedJobId);
  const [activatingId, setActivatingId] = useState<number | null>(null);
  const [editTarget, setEditTarget] = useState<ModelMeta | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ModelMeta | null>(null);

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
      },
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

  function handleCompact() {
    compactMutation.mutate(undefined, {
      onSuccess: () => {
        toast.success('モデル ID を詰めました');
      },
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
    <div className="flex flex-col gap-8 p-6">
      <PageHeader
        eyebrow="Models"
        title="学習済みモデル"
        description="各モデルの行を開くとバックテストを実行できます。"
      >
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
      </PageHeader>

      <div className="flex flex-col gap-6">
        {modelsQuery.isPending ? (
          <Skeleton className="h-24 w-full rounded-sm" />
        ) : modelsQuery.data ? (
          <OperatingModelsCard models={modelsQuery.data} linkToModels={false} />
        ) : null}

        {trackedJobId && (
          <JobProgressCard
            jobId={trackedJobId}
            title="train ジョブ進捗"
            onDismiss={() => setTrackedJobId(null)}
          />
        )}

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
            onSetProbability={handleSetProbability}
            onEdit={setEditTarget}
            onDelete={setDeleteTarget}
            activatingId={activatingId}
            settingProbability={updateSettings.isPending}
          />
        )}
      </div>

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
