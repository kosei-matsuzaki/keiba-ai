import { useState } from 'react';

import { useSettings, useUpdateSettings } from '@/hooks/useSettings';
import { SettingsForm, type SettingsSection } from '@/components/SettingsForm';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from '@/components/ui/toast';
import { formatErrorMessage } from '@/lib/api';
import type { SettingsUpdate } from '@/types/api';

type TabKey = SettingsSection;

// 等幅で並べるので英字に揃える。
// INGEST はレース画面 (Race > 過去のレース) の取込パネルへ、
// OPS の緊急停止はスクレイパー状態カードへ移設したため、ここには置かない。
const TABS: { value: TabKey; label: string }[] = [
  { value: 'scraper', label: 'SCRAPER' },
  { value: 'betting', label: 'BETTING' },
  { value: 'bet_types', label: 'BET TYPES' },
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
        toast.error('保存に失敗しました', {
          description: await formatErrorMessage(err),
          action: { label: '再試行', onClick: () => handleSubmit(values) },
        });
      },
    });
  }

  return (
    <div className="flex flex-col gap-12 p-6">
      <PageHeader
        eyebrow="Settings"
        title="設定"
        description="全レース共通の予想パラメータとスクレイパーの動作設定"
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

      {/* SettingsForm は activeSection で 1 セクションだけ表示する
          (マウントは維持されるので、タブを切り替えても入力中の値は消えない)。 */}
      {settingsQuery.isPending ? (
        <Skeleton className="h-96 w-full rounded-sm" />
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
          activeSection={activeTab}
        />
      )}

      <p className="text-xs leading-relaxed text-subtle-foreground">
        券種・予算・EV 閾値はここが既定値です。レースごとに変えたいときは、
        レース詳細の「推奨買目」で上書きできます。
      </p>
    </div>
  );
}
