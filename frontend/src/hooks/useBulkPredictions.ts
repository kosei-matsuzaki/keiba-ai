import { useQuery } from '@tanstack/react-query';
import { fetchBulkPredictions } from '@/lib/api';

/**
 * Fetch top-N predictions for a list of race IDs in a single request.
 * Returns an empty map when race_ids is empty or when no active model exists.
 *
 * `enabled` を false にすると、画面を開いた瞬間に全レース分の予測が裏で走る
 * のを抑止できる (Race タブ一覧で「AI 予想を実行」ボタンを押すまで false)。
 */
export function useBulkPredictions(race_ids: string[], top_n = 3, enabled = true) {
  return useQuery({
    queryKey: ['predictions', 'bulk', race_ids, top_n],
    queryFn: () => fetchBulkPredictions(race_ids, top_n),
    enabled: enabled && race_ids.length > 0,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}
