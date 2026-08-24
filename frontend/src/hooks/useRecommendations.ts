import { useQuery } from '@tanstack/react-query';
import { fetchRecommendations, type RecommendationParams } from '@/lib/api';

/**
 * 推奨買目。params でこのレースだけ設定を上書きできる
 * (未指定の項目は Settings の全レース共通値が使われる)。
 */
export function useRecommendations(
  raceId: string,
  enabled = true,
  params?: RecommendationParams,
) {
  return useQuery({
    queryKey: ['recommendations', raceId, params],
    queryFn: () => fetchRecommendations(raceId, params),
    enabled: enabled && Boolean(raceId),
    retry: false,
  });
}
