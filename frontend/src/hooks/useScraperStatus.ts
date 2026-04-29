import { useQuery } from '@tanstack/react-query';
import { fetchScraperStatus } from '@/lib/api';
import { useScraperStore } from '@/store/app';

export function useScraperStatus() {
  const isRunning = useScraperStore((s) => s.isRunning);

  return useQuery({
    queryKey: ['scraper', 'status'],
    queryFn: fetchScraperStatus,
    // Poll every 5 s when a job is running, otherwise every 30 s
    refetchInterval: isRunning ? 5_000 : 30_000,
  });
}
