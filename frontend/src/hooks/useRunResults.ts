import { useEffect, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { runResultsScraper } from '@/lib/api';
import { useJobStatus } from '@/hooks/useJobStatus';
import { toast } from '@/components/ui/toast';
import type { ScraperRunResultsRequest } from '@/types/api';

/**
 * 現在までに確定したレース（結果＋確定オッズ）を未取得分だけ取り込むジョブを起動する。
 *
 * 202 応答後はバックグラウンドで動くため、job を polling し、終了時に
 * races / bets / scraper status のキャッシュを無効化する。
 */
export function useRunResults() {
  const queryClient = useQueryClient();
  const [pendingJobId, setPendingJobId] = useState<string | null>(null);

  const jobStatus = useJobStatus(pendingJobId);

  useEffect(() => {
    if (!jobStatus.data) return;
    const { status } = jobStatus.data;
    if (status === 'completed') {
      queryClient.invalidateQueries({ queryKey: ['scraper', 'status'] });
      queryClient.invalidateQueries({ queryKey: ['races'] });
      queryClient.invalidateQueries({ queryKey: ['bets'] });
      toast.success('確定レースの取込が完了しました');
      setPendingJobId(null);
    } else if (status === 'failed') {
      toast.error(`結果取込ジョブが失敗しました: ${jobStatus.data.error ?? '不明なエラー'}`);
      setPendingJobId(null);
    }
  }, [jobStatus.data, queryClient]);

  const mutation = useMutation({
    mutationFn: (body: ScraperRunResultsRequest = {}) => runResultsScraper(body),
    retry: false,
    onSuccess: (data) => {
      setPendingJobId(data.job_id);
    },
  });

  return {
    ...mutation,
    isPolling: pendingJobId !== null,
  };
}
