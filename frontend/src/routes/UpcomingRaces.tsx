import { CalendarClock } from 'lucide-react';

import { useUpcomingRaces } from '@/hooks/useUpcomingRaces';
import { RaceCard } from '@/components/RaceCard';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';

function RaceListSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 6 }).map((_, i) => (
        <Skeleton key={i} className="h-44 rounded-lg" />
      ))}
    </div>
  );
}

export function UpcomingRaces() {
  const { data, isPending, isError } = useUpcomingRaces(7);

  return (
    <div className="flex flex-col gap-6 p-6">
      <PageHeader
        icon={CalendarClock}
        title="Upcoming Races"
        description="直近 7 日に予定されているレース一覧"
      />

      {isPending ? (
        <RaceListSkeleton />
      ) : isError ? (
        <EmptyState
          message="レース情報の取得に失敗しました"
          description="バックエンドが起動しているか確認してください。"
        />
      ) : data.races.length === 0 ? (
        <EmptyState
          message="今週の予定レースはありません"
          description="スクレイパーでデータを取り込んでください。"
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
