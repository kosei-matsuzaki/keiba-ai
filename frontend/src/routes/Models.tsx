import { useState } from 'react';
import { Brain } from 'lucide-react';

import { useModels } from '@/hooks/useModels';
import { useActivateModel } from '@/hooks/useActivateModel';
import { useUpdateModel } from '@/hooks/useUpdateModel';
import { useDeleteModel } from '@/hooks/useDeleteModel';
import { useCompactModelIds } from '@/hooks/useCompactModelIds';
import { useTrainModel } from '@/hooks/useTrainModel';
import { ActiveModelCard } from '@/components/ActiveModelCard';
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
  const activateMutation = useActivateModel();
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
        toast.error(`Activate に失敗しました: ${await formatErrorMessage(err)}`);
        setActivatingId(null);
      },
    });
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
        toast.error(`削除に失敗しました: ${await formatErrorMessage(err)}`);
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
    <div className="flex flex-col gap-6 p-6">
      <PageHeader
        icon={Brain}
        title="Models"
        description="学習済みモデルの管理。各モデルの行を開くとバックテストを実行できます。"
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
          <Skeleton className="h-24 w-full rounded-lg" />
        ) : modelsQuery.data ? (
          <ActiveModelCard
            model={modelsQuery.data.find((m) => m.is_active) ?? null}
            linkToModels={false}
          />
        ) : null}

        {trackedJobId && (
          <JobProgressCard
            jobId={trackedJobId}
            title="train ジョブ進捗"
            onDismiss={() => setTrackedJobId(null)}
          />
        )}

        {modelsQuery.isPending ? (
          <Skeleton className="h-64 w-full rounded-lg" />
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
            onEdit={setEditTarget}
            onDelete={setDeleteTarget}
            activatingId={activatingId}
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
