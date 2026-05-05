import { useQuery } from '@tanstack/react-query';
import { fetchBetList, type BetFilterParams } from '@/lib/api';

export function useBetList(params: BetFilterParams = {}) {
  return useQuery({
    queryKey: ['bets', 'list', params],
    queryFn: () => fetchBetList(params),
    staleTime: 60 * 1000,
  });
}
