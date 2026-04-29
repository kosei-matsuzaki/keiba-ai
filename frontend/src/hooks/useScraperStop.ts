import { useMutation, useQueryClient } from '@tanstack/react-query';
import { stopScraper } from '@/lib/api';

export function useScraperStop() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: stopScraper,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scraper', 'status'] });
    },
  });
}
