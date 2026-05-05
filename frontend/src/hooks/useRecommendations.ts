import { useQuery } from '@tanstack/react-query';
import { fetchRecommendations } from '@/lib/api';

export function useRecommendations(
  raceId: string,
  enabled = true,
  params?: { top_n_horses?: number; top_k?: number },
) {
  return useQuery({
    queryKey: ['recommendations', raceId, params],
    queryFn: () => fetchRecommendations(raceId, params),
    enabled: enabled && Boolean(raceId),
    retry: false,
  });
}
