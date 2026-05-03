import { Settings2 } from 'lucide-react';

import { useSettings, useUpdateSettings } from '@/hooks/useSettings';
import { SettingsForm } from '@/components/SettingsForm';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from '@/components/ui/toast';
import { formatErrorMessage } from '@/lib/api';
import type { SettingsUpdate } from '@/types/api';

export function Settings() {
  const settingsQuery = useSettings();
  const updateMutation = useUpdateSettings();

  function handleSubmit(values: SettingsUpdate) {
    updateMutation.mutate(values, {
      onSuccess: () => {
        toast.success('設定を保存しました');
      },
      onError: async (err) => {
        toast.error(`保存に失敗しました: ${await formatErrorMessage(err)}`);
      },
    });
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      <PageHeader
        icon={Settings2}
        title="Settings"
        description="スクレイパーのレート制御・ベッティング期待値・User-Agent の調整"
      />

      {settingsQuery.isPending ? (
        <Skeleton className="h-96 w-full max-w-3xl rounded-lg" />
      ) : settingsQuery.isError ? (
        <EmptyState
          message="設定の取得に失敗しました"
          description="バックエンドが起動しているか確認してください。"
        />
      ) : (
        <SettingsForm
          defaults={settingsQuery.data}
          onSubmit={handleSubmit}
          isPending={updateMutation.isPending}
        />
      )}
    </div>
  );
}
