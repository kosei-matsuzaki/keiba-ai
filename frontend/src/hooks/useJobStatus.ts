import { useQuery } from '@tanstack/react-query';

import { fetchJob } from '@/lib/api';
import type { JobInfo } from '@/types/api';

const TERMINAL_STATUSES = new Set(['success', 'failed', 'cancelled']);

/**
 * Poll a single job by id every 2 seconds until it reaches a terminal state.
 *
 * Pass `null` (or undefined) for jobId when no job is being tracked — the
 * query stays disabled so we don't spam the backend with /jobs/null lookups.
 */
export function useJobStatus(jobId: string | null | undefined) {
  return useQuery<JobInfo>({
    queryKey: ['job', jobId],
    queryFn: () => fetchJob(jobId as string),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      // Stop polling once the job is in a terminal state.
      const data = query.state.data as JobInfo | undefined;
      if (data && TERMINAL_STATUSES.has(data.status)) return false;
      return 2_000;
    },
    // Don't keep stale data when the user changes the tracked jobId.
    gcTime: 0,
  });
}
