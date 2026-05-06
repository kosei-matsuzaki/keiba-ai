import { useEffect, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchLiveOdds } from '@/lib/api';
import { useJobStatus } from '@/hooks/useJobStatus';
import { toast } from '@/components/ui/toast';
import type { FetchLiveOddsRequest } from '@/types/api';

/**
 * Mutation hook for fetching live odds.
 *
 * After the 202 response the backend job runs asynchronously.
 * This hook internally polls the job via useJobStatus and invalidates
 * the recommendations cache once the job reaches a terminal state.
 *
 * raceId is used to scope the recommendations invalidation.
 */
export function useFetchLiveOdds(raceId?: string) {
  const queryClient = useQueryClient();
  const [pendingJobId, setPendingJobId] = useState<string | null>(null);

  const jobStatus = useJobStatus(pendingJobId);

  // React to job completion / failure
  useEffect(() => {
    if (!jobStatus.data) return;

    const { status } = jobStatus.data;
    if (status === 'completed') {
      if (raceId) {
        queryClient.invalidateQueries({ queryKey: ['recommendations', raceId] });
      } else {
        queryClient.invalidateQueries({ queryKey: ['recommendations'] });
      }
      queryClient.invalidateQueries({ queryKey: ['scraper', 'status'] });
      setPendingJobId(null);
    } else if (status === 'failed') {
      toast.error(`オッズ取得ジョブが失敗しました: ${jobStatus.data.error ?? '不明なエラー'}`);
      setPendingJobId(null);
    }
  }, [jobStatus.data, raceId, queryClient]);

  const mutation = useMutation({
    mutationFn: (body: FetchLiveOddsRequest) => fetchLiveOdds(body),
    retry: false,
    onSuccess: (data) => {
      setPendingJobId(data.job_id);
    },
  });

  return {
    ...mutation,
    // Expose whether a job is actively being polled (mutation sent + job not terminal)
    isPolling: pendingJobId !== null,
  };
}
