import { useQuery } from '@tanstack/react-query';
import { fetchBetTimeseries, type BetFilterParams } from '@/lib/api';

export function useBetTimeseries(
  params: BetFilterParams & { bucket?: 'day' | 'week' | 'month' } = {}
) {
  return useQuery({
    queryKey: ['bets', 'timeseries', params.from ?? null, params.to ?? null, params.bet_type ?? null, params.source ?? null, params.bucket ?? null],
    queryFn: () => fetchBetTimeseries(params),
    staleTime: 60 * 1000,
  });
}
