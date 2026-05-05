import { useQuery } from '@tanstack/react-query';
import { fetchRecentRaces } from '@/lib/api';
import type { UpcomingRacesResponse } from '@/types/api';

export function useRecentRaces(days = 30) {
  return useQuery<UpcomingRacesResponse>({
    queryKey: ['races', 'recent', days],
    queryFn: () => fetchRecentRaces(days),
    staleTime: 5 * 60_000,
  });
}
