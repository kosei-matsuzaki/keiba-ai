import { useQuery } from '@tanstack/react-query';
import { fetchUpcomingRaces } from '@/lib/api';

export function useUpcomingRaces(days = 7) {
  return useQuery({
    queryKey: ['races', 'upcoming', days],
    queryFn: () => fetchUpcomingRaces(days),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}
