import { CalendarDays } from 'lucide-react';

import { PageHeader } from '@/components/PageHeader';
import { UpcomingRaces } from '@/routes/UpcomingRaces';

/**
 * Race route — 1 つのタブ列で 「今週末の土」「日」「Past」 を切替える構成。
 * 二重タブを避けるため、Upcoming/Past の外枠タブは廃止し、
 * UpcomingRaces 内部の day-tabs に Past をマージしている。
 */
export function Races() {
  return (
    <div className="flex flex-col gap-4 p-6">
      <PageHeader
        icon={CalendarDays}
        title="Race"
        description="今週末の予定 / 過去レース"
      />
      <UpcomingRaces embedded />
    </div>
  );
}
