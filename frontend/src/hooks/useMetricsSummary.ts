import { useQuery } from '@tanstack/react-query';
import { fetchMetricsSummary } from '@/lib/api';

export function useMetricsSummary(range = '30d') {
  return useQuery({
    queryKey: ['metrics', 'summary', range],
    queryFn: () => fetchMetricsSummary(range),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}
