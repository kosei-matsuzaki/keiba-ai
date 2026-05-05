import { useQuery } from '@tanstack/react-query';
import { fetchRecentRaces } from '@/lib/api';
import type { UpcomingRacesResponse } from '@/types/api';

export type RecentRacesArgs = { days: number } | { from: string; to: string };

export function useRecentRaces(args: RecentRacesArgs = { days: 30 }) {
  const isRange = 'from' in args;
  return useQuery<UpcomingRacesResponse>({
    queryKey: isRange
      ? ['races', 'recent', 'range', args.from, args.to]
      : ['races', 'recent', 'days', args.days],
    queryFn: () => fetchRecentRaces(args),
    staleTime: 5 * 60_000,
  });
}
