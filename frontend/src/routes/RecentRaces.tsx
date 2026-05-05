import { useState } from 'react';
import { History } from 'lucide-react';

import { useRecentRaces } from '@/hooks/useRecentRaces';
import { RaceCard } from '@/components/RaceCard';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

const PRESET_DAYS = [7, 30, 90] as const;
type PresetDays = (typeof PRESET_DAYS)[number];

type Mode = { kind: 'preset'; days: PresetDays } | { kind: 'range'; from: string; to: string };

function RaceListSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 6 }).map((_, i) => (
        <Skeleton key={i} className="h-44 rounded-lg" />
      ))}
    </div>
  );
}

export function RecentRaces() {
  const [mode, setMode] = useState<Mode>({ kind: 'preset', days: 30 });
  const [draftFrom, setDraftFrom] = useState<string>('');
  const [draftTo, setDraftTo] = useState<string>('');

  const queryArgs =
    mode.kind === 'preset' ? { days: mode.days } : { from: mode.from, to: mode.to };
  const { data, isPending, isError } = useRecentRaces(queryArgs);

  const rangeValid = draftFrom !== '' && draftTo !== '' && draftFrom <= draftTo;

  return (
    <div className="flex flex-col gap-6 p-6">
      <PageHeader
        icon={History}
        title="Recent Races"
        description="過去のレース結果。期間プリセット または 日付範囲で絞り込み"
      >
        <div className="flex flex-wrap items-center gap-2" aria-label="期間選択">
          <div className="flex items-center gap-1" role="group" aria-label="期間プリセット">
            {PRESET_DAYS.map((d) => {
              const active = mode.kind === 'preset' && mode.days === d;
              return (
                <Button
                  key={d}
                  size="sm"
                  variant={active ? 'default' : 'outline'}
                  onClick={() => setMode({ kind: 'preset', days: d })}
                  aria-pressed={active}
                >
                  {d} 日
                </Button>
              );
            })}
          </div>
          <div className="ml-2 flex items-center gap-1" aria-label="日付範囲">
            <Input
              type="date"
              value={draftFrom}
              onChange={(e) => setDraftFrom(e.target.value)}
              aria-label="開始日"
              className="h-8 w-40"
            />
            <span className="text-muted-foreground">–</span>
            <Input
              type="date"
              value={draftTo}
              onChange={(e) => setDraftTo(e.target.value)}
              aria-label="終了日"
              className="h-8 w-40"
            />
            <Button
              size="sm"
              variant={mode.kind === 'range' ? 'default' : 'outline'}
              disabled={!rangeValid}
              onClick={() => setMode({ kind: 'range', from: draftFrom, to: draftTo })}
              aria-pressed={mode.kind === 'range'}
            >
              適用
            </Button>
          </div>
        </div>
      </PageHeader>

      {isPending ? (
        <RaceListSkeleton />
      ) : isError ? (
        <EmptyState
          message="レース情報の取得に失敗しました"
          description="バックエンドが起動しているか確認してください。"
        />
      ) : data.races.length === 0 ? (
        <EmptyState
          message="該当期間にレースがありません"
          description="期間を変更するか、データを取り込んでください。"
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data.races.map((race) => (
            <RaceCard key={race.race_id} race={race} />
          ))}
        </div>
      )}
    </div>
  );
}
