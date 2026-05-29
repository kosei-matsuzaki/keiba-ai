import { useQuery } from '@tanstack/react-query';
import { fetchPredictions } from '@/lib/api';

/**
 * AI 予想 (予想スコア + 確率) を取得する。
 *
 * 推論はモデルをロードして走らせる重い処理なので、画面を開いた瞬間に自動実行
 * せず、`enabled` でボタン主導に gate できるようにしている (RaceDetail で
 * 「AI 予想を実行」ボタンを押すまで false)。
 */
export function usePredictions(raceId: string, enabled = true) {
  return useQuery({
    queryKey: ['predictions', raceId],
    queryFn: () => fetchPredictions(raceId),
    enabled: enabled && Boolean(raceId),
    retry: false,
  });
}
