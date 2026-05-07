import { Settings2 } from 'lucide-react';

import { useSettings, useUpdateSettings } from '@/hooks/useSettings';
import { SettingsForm } from '@/components/SettingsForm';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from '@/components/ui/toast';
import { formatErrorMessage } from '@/lib/api';
import { Ingest } from '@/routes/Ingest';
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
        description="アプリ設定 / スクレイパー実行"
      />

      <Tabs defaultValue="general" className="flex flex-col gap-6">
        <TabsList className="self-start">
          <TabsTrigger value="general">一般</TabsTrigger>
          <TabsTrigger value="ingest">Ingest</TabsTrigger>
        </TabsList>

        <TabsContent value="general" className="mt-0">
          {settingsQuery.isPending ? (
            <Skeleton className="h-96 w-full rounded-lg" />
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
        </TabsContent>

        <TabsContent value="ingest" className="mt-0">
          <Ingest embedded />
        </TabsContent>
      </Tabs>
    </div>
  );
}
