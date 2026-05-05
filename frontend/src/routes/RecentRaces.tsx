import { useState } from 'react';
import { History } from 'lucide-react';

import { useRecentRaces } from '@/hooks/useRecentRaces';
import { RaceCard } from '@/components/RaceCard';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';

const PRESET_DAYS = [7, 30, 90] as const;
type PresetDays = (typeof PRESET_DAYS)[number];

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
  const [days, setDays] = useState<PresetDays>(30);
  const { data, isPending, isError } = useRecentRaces(days);

  return (
    <div className="flex flex-col gap-6 p-6">
      <PageHeader
        icon={History}
        title="Recent Results"
        description="過去 30 日のレース結果"
      >
        <div className="flex items-center gap-1" role="group" aria-label="期間選択">
          {PRESET_DAYS.map((d) => (
            <Button
              key={d}
              size="sm"
              variant={days === d ? 'default' : 'outline'}
              onClick={() => setDays(d)}
              aria-pressed={days === d}
            >
              {d} 日
            </Button>
          ))}
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
