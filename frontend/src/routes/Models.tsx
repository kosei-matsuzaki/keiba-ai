import { useState } from 'react';
import { useModels } from '@/hooks/useModels';
import { useActivateModel } from '@/hooks/useActivateModel';
import { useTrainModel } from '@/hooks/useTrainModel';
import { ActiveModelCard } from '@/components/ActiveModelCard';
import { ModelTable } from '@/components/ModelTable';
import { TrainModelDialog } from '@/components/TrainModelDialog';
import { JobProgressCard } from '@/components/JobProgressCard';
import { EmptyState } from '@/components/EmptyState';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from '@/components/ui/toast';
import { formatErrorMessage } from '@/lib/api';
import { useTrainingStore } from '@/store/app';
import type { TrainRequest } from '@/types/api';

export function Models() {
  const modelsQuery = useModels();
  const activateMutation = useActivateModel();
  const trainMutation = useTrainModel();
  const trackedJobId = useTrainingStore((s) => s.trackedJobId);
  const setTrackedJobId = useTrainingStore((s) => s.setTrackedJobId);
  const [activatingId, setActivatingId] = useState<number | null>(null);

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
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Models</h1>
        <TrainModelDialog onSubmit={handleTrain} isPending={trainMutation.isPending} />
      </div>

      {/* Active model summary at the top of the page (no link to Models — already here) */}
      {modelsQuery.data && (
        <ActiveModelCard
          model={modelsQuery.data.find((m) => m.is_active) ?? null}
          linkToModels={false}
        />
      )}

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
          activatingId={activatingId}
        />
      )}
    </div>
  );
}
