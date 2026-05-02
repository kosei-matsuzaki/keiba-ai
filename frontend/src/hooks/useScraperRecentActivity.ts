import { useQuery } from '@tanstack/react-query';

import { fetchScraperRecentActivity } from '@/lib/api';
import { useScraperStore } from '@/store/app';
import type { ScraperRecentActivity } from '@/types/api';

/**
 * Aggregate of scrape_log over the last `minutes` minutes — surfaces both
 * UI-launched and CLI-launched ingest activity. Polls more aggressively while
 * a UI-launched job is running, since that's the only time the user is
 * actively watching the screen.
 */
export function useScraperRecentActivity(minutes = 10) {
  const isRunning = useScraperStore((s) => s.isRunning);

  return useQuery<ScraperRecentActivity>({
    queryKey: ['scraper', 'recent_activity', minutes],
    queryFn: () => fetchScraperRecentActivity(minutes),
    refetchInterval: isRunning ? 5_000 : 30_000,
  });
}
