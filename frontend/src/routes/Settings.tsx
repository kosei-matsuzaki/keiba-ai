import { useSettings, useUpdateSettings } from '@/hooks/useSettings';
import { SettingsForm } from '@/components/SettingsForm';
import { EmptyState } from '@/components/EmptyState';
import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { toast } from '@/components/ui/toast';
import type { SettingsUpdate } from '@/types/api';

export function Settings() {
  const settingsQuery = useSettings();
  const updateMutation = useUpdateSettings();

  function handleSubmit(values: SettingsUpdate) {
    updateMutation.mutate(values, {
      onSuccess: () => {
        toast.success('設定を保存しました');
      },
      onError: (err) => {
        toast.error(`保存に失敗しました: ${(err as Error).message}`);
      },
    });
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      <h1 className="text-2xl font-bold">Settings</h1>

      {settingsQuery.isPending ? (
        <Skeleton className="h-80 w-full rounded-lg" />
      ) : settingsQuery.isError ? (
        <EmptyState
          message="設定の取得に失敗しました"
          description="バックエンドが起動しているか確認してください。"
        />
      ) : (
        <Card className="max-w-xl">
          <CardHeader>
            <CardTitle className="text-base">スクレイパー・ベット設定</CardTitle>
          </CardHeader>
          <CardContent>
            <SettingsForm
              defaults={settingsQuery.data}
              onSubmit={handleSubmit}
              isPending={updateMutation.isPending}
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
