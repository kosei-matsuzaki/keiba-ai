import type { ReactNode } from 'react';

import { useDataCoverage } from '@/hooks/useRacesCalendar';
import { Skeleton } from '@/components/ui/skeleton';
import { formatCount, formatDate, formatPercent } from '@/lib/formatters';

/**
 * 取込済みデータの状況を 1 行で出す。
 *
 * 「どこからどこまで、何レース入っていて、結果まで取れているのはどれだけか」が
 * 分からないと、予想が古いデータで出ているのか判断できない。
 */
export function DataCoverageBand() {
  const { data, isPending, isError } = useDataCoverage();

  if (isPending) return <Skeleton className="h-12 w-full" />;
  if (isError || !data) return null;

  const resultRate = data.race_count > 0 ? data.result_count / data.race_count : null;
  const pendingResults = data.race_count - data.result_count;

  return (
    <dl className="block-surface-compact flex flex-wrap items-baseline gap-x-8 gap-y-2">
      <Item label="取込期間">
        <span className="font-mono tabular-nums">
          {formatDate(data.first_date)}
          <span className="mx-1 text-subtle-foreground">→</span>
          {formatDate(data.last_date)}
        </span>
      </Item>
      <Item label="レース">
        <span className="font-mono tabular-nums">{formatCount(data.race_count)}</span>
      </Item>
      <Item label="出走馬">
        <span className="font-mono tabular-nums">{formatCount(data.entry_count)}</span>
      </Item>
      <Item label="結果あり">
        <span className="font-mono tabular-nums">
          {formatCount(data.result_count)}
          {resultRate != null && (
            <span className="ml-1 text-unit">({formatPercent(resultRate, 1)})</span>
          )}
        </span>
      </Item>
      {pendingResults > 0 && (
        <Item label="結果待ち">
          <span className="font-mono tabular-nums text-primary">
            {formatCount(pendingResults)}
            <span className="ml-1 text-unit">R</span>
          </span>
        </Item>
      )}
      <Item label={`直近${data.recent_days_span}日`}>
        <span className="font-mono tabular-nums">
          {formatCount(data.recent_days_with_data)}
          <span className="ml-1 text-unit">開催日</span>
        </span>
      </Item>
    </dl>
  );
}

function Item({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline gap-2">
      <dt className="text-label-ja">{label}</dt>
      <dd className="text-sm">{children}</dd>
    </div>
  );
}
