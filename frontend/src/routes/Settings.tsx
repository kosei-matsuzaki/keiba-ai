import { useSettings, useUpdateSettings } from '@/hooks/useSettings';
import { SettingsForm } from '@/components/SettingsForm';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from '@/lib/toast';
import { formatErrorMessage } from '@/lib/api';
import type { SettingsUpdate } from '@/types/api';

// タブ (SCRAPER / BETTING) は外した。合計 8 項目しかなく、隠す量ではない。
// 分けていると「取り込み方」と「買い方」を両方直したいときに切り替えが要り、
// 保存も 2 回になる。1 画面に 2 グループを縦に並べれば 1 回で済む。
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

      {/* 見出しは他のタブと同じく全幅。中身だけ中央に寄せる —
          行長を max-w-3xl に締めてあるので、左寄せだと右半分が空いたままになる。 */}
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
