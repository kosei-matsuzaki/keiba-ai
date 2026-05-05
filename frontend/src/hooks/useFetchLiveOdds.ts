import { useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchLiveOdds } from '@/lib/api';
import type { FetchLiveOddsRequest } from '@/types/api';

export function useFetchLiveOdds(raceId?: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (body: FetchLiveOddsRequest) => fetchLiveOdds(body),
    retry: false,
    onSuccess: () => {
      // recommendations を invalidate してオッズ更新後の推奨を再取得する
      if (raceId) {
        queryClient.invalidateQueries({ queryKey: ['recommendations', raceId] });
      } else {
        queryClient.invalidateQueries({ queryKey: ['recommendations'] });
      }
      queryClient.invalidateQueries({ queryKey: ['scraper', 'status'] });
    },
  });
}
