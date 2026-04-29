import { useQuery } from '@tanstack/react-query';
import { fetchPredictions } from '@/lib/api';

export function usePredictions(raceId: string) {
  return useQuery({
    queryKey: ['predictions', raceId],
    queryFn: () => fetchPredictions(raceId),
    enabled: Boolean(raceId),
    retry: false,
  });
}
