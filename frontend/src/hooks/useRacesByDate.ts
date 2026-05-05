import { useQuery } from '@tanstack/react-query';
import { fetchRacesByDate } from '@/lib/api';
import type { UpcomingRacesResponse } from '@/types/api';

export function useRacesByDate(date: string) {
  return useQuery<UpcomingRacesResponse>({
    queryKey: ['races', 'by_date', date],
    queryFn: () => fetchRacesByDate(date),
    staleTime: 5 * 60_000,
    // Skip the query when date is empty (initial render before default is resolved)
    enabled: date.length > 0,
  });
}
