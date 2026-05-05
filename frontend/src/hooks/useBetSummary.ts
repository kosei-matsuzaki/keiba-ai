import { useQuery } from '@tanstack/react-query';
import { fetchBetSummary, type BetFilterParams } from '@/lib/api';

export function useBetSummary(params: BetFilterParams = {}) {
  return useQuery({
    queryKey: ['bets', 'summary', params.from ?? null, params.to ?? null, params.bet_type ?? null, params.source ?? null],
    queryFn: () => fetchBetSummary(params),
    staleTime: 60 * 1000, // 1 minute
  });
}
