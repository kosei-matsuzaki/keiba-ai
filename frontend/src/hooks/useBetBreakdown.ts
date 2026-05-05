import { useQuery } from '@tanstack/react-query';
import { fetchBetBreakdown, type BetFilterParams } from '@/lib/api';

export function useBetBreakdown(
  params: BetFilterParams & { group_by?: 'bet_type' | 'race_class' | 'month' | 'source' } = {}
) {
  return useQuery({
    queryKey: ['bets', 'breakdown', params.from ?? null, params.to ?? null, params.bet_type ?? null, params.source ?? null, params.group_by ?? null],
    queryFn: () => fetchBetBreakdown(params),
    staleTime: 60 * 1000,
  });
}
