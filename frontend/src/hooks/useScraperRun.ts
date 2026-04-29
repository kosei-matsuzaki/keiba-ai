import { useMutation, useQueryClient } from '@tanstack/react-query';
import { runScraper } from '@/lib/api';
import type { ScraperRunRequest } from '@/types/api';

export function useScraperRun() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (body: ScraperRunRequest) => runScraper(body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scraper', 'status'] });
    },
  });
}
