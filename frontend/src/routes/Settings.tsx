import { Settings2 } from 'lucide-react';

import { useSettings, useUpdateSettings } from '@/hooks/useSettings';
import { SettingsForm } from '@/components/SettingsForm';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
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
        description="アプリ設定とスクレイパー実行をまとめて管理"
      />

      {/* 一般設定: 4 セクション (scraper / betting / bet_types / ops) を縦並びで描画 */}
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

      {/* Ingest: スクレイピング操作 + 状態カード */}
      <Ingest embedded />
    </div>
  );
}
