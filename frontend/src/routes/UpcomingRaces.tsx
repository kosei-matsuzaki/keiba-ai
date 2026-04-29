import { useUpcomingRaces } from '@/hooks/useUpcomingRaces';
import { RaceCard } from '@/components/RaceCard';
import { EmptyState } from '@/components/EmptyState';
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
      <h1 className="text-2xl font-bold">Upcoming Races</h1>

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
