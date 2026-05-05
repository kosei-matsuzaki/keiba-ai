import { useEffect, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { runShutubaScraper } from '@/lib/api';
import { useJobStatus } from '@/hooks/useJobStatus';
import { toast } from '@/components/ui/toast';
import type { ScraperRunShutubaRequest } from '@/types/api';

/**
 * Mutation hook for running the shutuba scraper.
 *
 * After the 202 response the backend job runs asynchronously.
 * This hook internally polls the job via useJobStatus and invalidates
 * the races and raceDetail caches once the job reaches a terminal state.
 *
 * raceId is used to scope the raceDetail invalidation when scraping a single race.
 */
export function useRunShutuba(raceId?: string) {
  const queryClient = useQueryClient();
  const [pendingJobId, setPendingJobId] = useState<string | null>(null);

  const jobStatus = useJobStatus(pendingJobId);

  // React to job completion / failure
  useEffect(() => {
    if (!jobStatus.data) return;

    const { status } = jobStatus.data;
    if (status === 'completed') {
      queryClient.invalidateQueries({ queryKey: ['scraper', 'status'] });
      queryClient.invalidateQueries({ queryKey: ['races'] });
      if (raceId) {
        queryClient.invalidateQueries({ queryKey: ['races', raceId] });
      }
      setPendingJobId(null);
    } else if (status === 'failed') {
      toast.error(`出馬表取込ジョブが失敗しました: ${jobStatus.data.error ?? '不明なエラー'}`);
      setPendingJobId(null);
    }
  }, [jobStatus.data, raceId, queryClient]);

  const mutation = useMutation({
    mutationFn: (body: ScraperRunShutubaRequest) => runShutubaScraper(body),
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
