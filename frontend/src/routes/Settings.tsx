import { useSettings, useUpdateSettings } from '@/hooks/useSettings';
import { SettingsForm } from '@/components/SettingsForm';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from '@/lib/toast';
import { formatErrorMessage } from '@/lib/api';
import type { SettingsUpdate } from '@/types/api';

// 画面の組み方は docs/design.md「UI 画面構成 > Settings」。
// INGEST はレース画面の取込パネルへ、OPS の緊急停止はスクレイパー状態カードへ
// 移設済みなので、ここには無い。

export function Settings() {
  const settingsQuery = useSettings();
  const updateMutation = useUpdateSettings();

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

      {/* 行長の上限はここ 1 か所。付けないと 1440px の画面でラベルが左端・
          入力が右端に張り付き、1 項目を読むのに目が 1000px 以上動く。
          見出しは全幅のまま、中身だけ中央に寄せる (左寄せだと右半分が空く)。 */}
      <div className="mx-auto w-full max-w-3xl">
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
        />
      )}
      </div>
    </div>
  );
}
