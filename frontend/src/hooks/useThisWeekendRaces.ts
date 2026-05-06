import { useQuery } from '@tanstack/react-query';
import { fetchThisWeekendRaces } from '@/lib/api';

export function useThisWeekendRaces() {
  return useQuery({
    queryKey: ['races', 'this_weekend'],
    queryFn: fetchThisWeekendRaces,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}
