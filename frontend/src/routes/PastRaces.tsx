import { useSearchParams, useNavigate } from 'react-router-dom';
import { History } from 'lucide-react';

import { useRacesByDate } from '@/hooks/useRacesByDate';
import { DateYMDPicker } from '@/components/DateYMDPicker';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { Label } from '@/components/ui/label';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { RaceSummary } from '@/types/api';

/** Extract 2-digit race number from race_id (last 2 chars). */
function raceNumber(raceId: string): string {
  return raceId.slice(-2);
}

/** Today's date in YYYY-MM-DD (local time). */
function todayIso(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function TableSkeleton() {
  return (
    <div className="space-y-4">
      {Array.from({ length: 2 }).map((_, i) => (
        <Skeleton key={i} className="h-32 w-full rounded-lg" />
      ))}
    </div>
  );
}

interface CourseSection {
  course: string;
  races: RaceSummary[];
}

function groupByCourse(races: RaceSummary[]): CourseSection[] {
  const map = new Map<string, RaceSummary[]>();
  for (const race of races) {
    const list = map.get(race.course) ?? [];
    list.push(race);
    map.set(race.course, list);
  }
  return Array.from(map.entries()).map(([course, rs]) => ({ course, races: rs }));
}

interface RaceTableProps {
  section: CourseSection;
  onRowClick: (race: RaceSummary) => void;
}

function RaceTable({ section, onRowClick }: RaceTableProps) {
  return (
    <section aria-labelledby={`course-${section.course}`}>
      <h2
        id={`course-${section.course}`}
        className="mb-2 text-base font-semibold text-foreground"
      >
        {section.course}
      </h2>
      <div className="overflow-hidden rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-16">R</TableHead>
              <TableHead>レース名</TableHead>
              <TableHead>クラス</TableHead>
              <TableHead>馬場</TableHead>
              <TableHead className="text-right">距離</TableHead>
              <TableHead className="text-right">頭数</TableHead>
              <TableHead>複勝払戻</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {section.races.map((race) => (
              <TableRow
                key={race.race_id}
                className="cursor-pointer hover:bg-accent/60"
                onClick={() => onRowClick(race)}
                role="button"
                aria-label={`${section.course} ${raceNumber(race.race_id)}R`}
              >
                <TableCell className="font-medium">{raceNumber(race.race_id)}R</TableCell>
                <TableCell>{race.name ?? '—'}</TableCell>
                <TableCell>{race.race_class ?? '—'}</TableCell>
                <TableCell>{race.surface}</TableCell>
                <TableCell className="text-right">{race.distance} m</TableCell>
                <TableCell className="text-right">{race.n_runners ?? '—'}</TableCell>
                <TableCell className="text-xs text-muted-foreground">—</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </section>
  );
}

interface PastRacesProps {
  /** Race route のタブ内に埋め込まれる場合 true。自前の PageHeader と外周 padding を抑制する。 */
  embedded?: boolean;
}

export function PastRaces({ embedded = false }: PastRacesProps = {}) {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  // Initialise date from ?date= query param, or fall back to today
  const initialDate = searchParams.get('date') ?? todayIso();
  const selectedDate = initialDate;

  const { data, isPending, isError } = useRacesByDate(selectedDate);

  function handleDateChange(value: string) {
    setSearchParams(value ? { date: value } : {});
  }

  function handleRowClick(race: RaceSummary) {
    navigate(`/races/${race.race_id}?date=${selectedDate}`);
  }

  const sections = data ? groupByCourse(data.races) : [];

  const datePicker = (
    <div className="flex items-center gap-2">
      <Label className="shrink-0 text-sm">日付</Label>
      <DateYMDPicker
        value={selectedDate}
        onChange={handleDateChange}
        ariaLabel="日付"
      />
    </div>
  );

  return (
    <div className={embedded ? 'flex flex-col gap-6' : 'flex flex-col gap-6 p-6'}>
      {!embedded && (
        <PageHeader
          icon={History}
          title="Past Races"
          description="日付を選んで過去のレース一覧を確認"
        >
          {datePicker}
        </PageHeader>
      )}
      {embedded && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">日付を選んで過去のレース一覧を確認</p>
          {datePicker}
        </div>
      )}

      {isPending ? (
        <TableSkeleton />
      ) : isError ? (
        <EmptyState
          message="レース情報の取得に失敗しました"
          description="バックエンドが起動しているか確認してください。"
        />
      ) : sections.length === 0 ? (
        <EmptyState
          message="該当日にレースがありません"
          description="別の日付を選択するか、データを取り込んでください。"
        />
      ) : (
        <div className="flex flex-col gap-8">
          {sections.map((section) => (
            <RaceTable key={section.course} section={section} onRowClick={handleRowClick} />
          ))}
        </div>
      )}
    </div>
  );
}
