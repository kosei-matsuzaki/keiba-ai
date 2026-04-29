import { useQuery } from '@tanstack/react-query';
import { fetchMetricsTimeseries } from '@/lib/api';

export function useMetricsTimeseries(metric = 'ndcg3', range = '180d') {
  return useQuery({
    queryKey: ['metrics', 'timeseries', metric, range],
    queryFn: () => fetchMetricsTimeseries(metric, range),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}
