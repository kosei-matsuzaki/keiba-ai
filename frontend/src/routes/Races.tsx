import { CalendarDays } from 'lucide-react';

import { PageHeader } from '@/components/PageHeader';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { PastRaces } from '@/routes/PastRaces';
import { UpcomingRaces } from '@/routes/UpcomingRaces';

/**
 * Race route — Upcoming と Past を 1 ページにまとめ、上部タブで切り替える。
 * 既存の UpcomingRaces / PastRaces コンポーネントを子としてそのまま埋め込み、
 * 各々の自前 PageHeader / レイアウトを保持する (差分最小化のため)。
 */
export function Races() {
  return (
    <div className="flex flex-col gap-4 p-6">
      <PageHeader
        icon={CalendarDays}
        title="Race"
        description="今週末のレース予定 / 過去レース閲覧"
      />

      <Tabs defaultValue="upcoming" className="flex flex-col gap-4">
        <TabsList className="self-start">
          <TabsTrigger value="upcoming">Upcoming</TabsTrigger>
          <TabsTrigger value="past">Past</TabsTrigger>
        </TabsList>

        <TabsContent value="upcoming" className="mt-0">
          <UpcomingRaces embedded />
        </TabsContent>
        <TabsContent value="past" className="mt-0">
          <PastRaces embedded />
        </TabsContent>
      </Tabs>
    </div>
  );
}
