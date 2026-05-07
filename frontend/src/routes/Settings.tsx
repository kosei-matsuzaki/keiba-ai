import { useState } from 'react';
import { Settings2 } from 'lucide-react';

import { useSettings, useUpdateSettings } from '@/hooks/useSettings';
import { SettingsForm, type SettingsSection } from '@/components/SettingsForm';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from '@/components/ui/toast';
import { formatErrorMessage } from '@/lib/api';
import { Ingest } from '@/routes/Ingest';
import type { SettingsUpdate } from '@/types/api';

type TabKey = SettingsSection | 'ingest';

const TABS: { value: TabKey; label: string }[] = [
  { value: 'scraper', label: 'スクレイパー' },
  { value: 'betting', label: 'ベッティング' },
  { value: 'bet_types', label: '馬券種' },
  { value: 'ops', label: '運用' },
  { value: 'ingest', label: 'Ingest' },
];

export function Settings() {
  const settingsQuery = useSettings();
  const updateMutation = useUpdateSettings();
  const [activeTab, setActiveTab] = useState<TabKey>('scraper');

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

  // 注: SettingsForm と Ingest は両方常時マウントしておき、CSS で hidden 切替する。
  // これにより活発な form state や Ingest の polling 状態が tab 切替で失われない。
  const isIngest = activeTab === 'ingest';
  const formActiveSection = !isIngest ? (activeTab as SettingsSection) : undefined;

  return (
    <div className="flex flex-col gap-6 p-6">
      <PageHeader
        icon={Settings2}
        title="Settings"
        description="アプリ設定とスクレイパー実行をまとめて管理"
      />

      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as TabKey)}>
        <TabsList className="self-start">
          {TABS.map((t) => (
            <TabsTrigger key={t.value} value={t.value}>
              {t.label}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      {/* 一般設定: 4 セクション。SettingsForm は activeSection で 1 つだけ表示。 */}
      {settingsQuery.isPending ? (
        <Skeleton className={isIngest ? 'hidden' : 'h-96 w-full rounded-lg'} />
      ) : settingsQuery.isError ? (
        <EmptyState
          message="設定の取得に失敗しました"
          description="バックエンドが起動しているか確認してください。"
        />
      ) : (
        <div className={isIngest ? 'hidden' : 'block'}>
          <SettingsForm
            defaults={settingsQuery.data}
            onSubmit={handleSubmit}
            isPending={updateMutation.isPending}
            activeSection={formActiveSection}
          />
        </div>
      )}

      {/* Ingest: scrape ジョブ操作 / 状態。tab 切替時もマウント維持 */}
      <div className={isIngest ? 'block' : 'hidden'}>
        <Ingest embedded />
      </div>
    </div>
  );
}
