import { useQuery } from '@tanstack/react-query';
import { fetchRaceDetail } from '@/lib/api';

export function useRaceDetail(raceId: string) {
  return useQuery({
    queryKey: ['races', raceId],
    queryFn: () => fetchRaceDetail(raceId),
    enabled: Boolean(raceId),
    retry: false,
  });
}
