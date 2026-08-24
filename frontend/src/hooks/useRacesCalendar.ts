import { useQuery } from '@tanstack/react-query';
import { fetchDataCoverage, fetchRacesCalendar } from '@/lib/api';

/**
 * 指定期間の日別取込状況。カレンダーの 1 か月表示に使う。
 *
 * 月単位でキャッシュされるので、月を行き来しても再取得は起きない。
 */
export function useRacesCalendar(from: string, to: string) {
  return useQuery({
    queryKey: ['races', 'calendar', from, to],
    queryFn: () => fetchRacesCalendar(from, to),
    staleTime: 5 * 60 * 1000,
  });
}

/** 取込済みデータ全体の状況 (期間・レース数・結果の有無)。 */
export function useDataCoverage() {
  return useQuery({
    queryKey: ['races', 'coverage'],
    queryFn: fetchDataCoverage,
    staleTime: 5 * 60 * 1000,
  });
}
