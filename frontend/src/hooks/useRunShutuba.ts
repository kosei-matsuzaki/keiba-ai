import { useMutation, useQueryClient } from '@tanstack/react-query';
import { runShutubaScraper } from '@/lib/api';
import type { ScraperRunShutubaRequest } from '@/types/api';

export function useRunShutuba() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (body: ScraperRunShutubaRequest) => runShutubaScraper(body),
    retry: false,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scraper', 'status'] });
      queryClient.invalidateQueries({ queryKey: ['races'] });
    },
  });
}
