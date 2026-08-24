import { useEffect, useMemo } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

import { useRacesByDate } from '@/hooks/useRacesByDate';
import { useThisWeekendRaces } from '@/hooks/useThisWeekendRaces';
import { DataCoverageBand } from '@/components/DataCoverageBand';
import { DayIngestPanel } from '@/components/DayIngestPanel';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { RaceCalendar } from '@/components/RaceCalendar';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { RaceSummary } from '@/types/api';

// ── helpers ───────────────────────────────────────────────────────────────────

/** Extract 2-digit race number from race_id (last 2 chars). */
function raceNumber(raceId: string): string {
  return raceId.slice(-2);
}

/** Today's date in YYYY-MM-DD (local time). */
function todayIso(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(
    d.getDate()
  ).padStart(2, '0')}`;
}

/** "2026-08-23" → "8/23 (日)"。 */
function formatDayLabel(dateStr: string): string {
  const [y, m, d] = dateStr.split('-').map(Number);
  if (!y || !m || !d) return dateStr;
  const dow = ['日', '月', '火', '水', '木', '金', '土'][new Date(y, m - 1, d).getDay()];
  return `${m}/${d} (${dow})`;
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

function TableSkeleton() {
  return (
    <div className="space-y-4">
      {Array.from({ length: 2 }).map((_, i) => (
        <Skeleton key={i} className="h-32 w-full rounded-sm" />
      ))}
    </div>
  );
}

// ── race table ────────────────────────────────────────────────────────────────

interface RaceTableProps {
  section: CourseSection;
  onRowClick: (race: RaceSummary) => void;
}

function RaceTable({ section, onRowClick }: RaceTableProps) {
  return (
    <section aria-labelledby={`course-${section.course}`}>
      {/* 「札幌 ───────── 12 R」。罫線が見出しから右に伸びると番組表らしくなる */}
      <h3
        id={`course-${section.course}`}
        className="mb-2 flex items-center gap-4 font-mono text-[11px] tracking-[0.18em] text-subtle-foreground"
      >
        {section.course}
        <span className="h-px flex-1 bg-border" aria-hidden="true" />
        <span className="tabular-nums">{section.races.length} R</span>
      </h3>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-16 text-label">R</TableHead>
            <TableHead>レース名</TableHead>
            <TableHead>クラス</TableHead>
            <TableHead>馬場</TableHead>
            <TableHead className="text-right">距離</TableHead>
            <TableHead className="text-right">頭数</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {section.races.map((race) => (
            <TableRow
              key={race.race_id}
              className="cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
              onClick={() => onRowClick(race)}
              // role="button" だけではキーで押せない (アクセシビリティ上の問題)。
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onRowClick(race);
                }
              }}
              role="button"
              tabIndex={0}
              aria-label={`${section.course} ${raceNumber(race.race_id)}R`}
            >
              {/* 一覧の中で数字だけが光る */}
              <TableCell className="font-mono tabular-nums font-medium text-primary">
                {raceNumber(race.race_id)}
                <span className="text-unit">R</span>
              </TableCell>
              <TableCell>
                {race.name ?? <span className="text-subtle-foreground/50">·</span>}
              </TableCell>
              {/* クラスは Badge をやめて素のテキスト。G1 だけ色を持たせて格を出す */}
              <TableCell
                className={
                  race.race_class === 'G1'
                    ? 'font-mono text-[10px] text-primary'
                    : 'font-mono text-[10px] text-subtle-foreground'
                }
              >
                {race.race_class ?? '·'}
              </TableCell>
              <TableCell>{race.surface}</TableCell>
              <TableCell className="cell-num">
                {race.distance}
                <span className="text-unit">m</span>
              </TableCell>
              <TableCell className="cell-num">
                {race.n_runners ?? '·'}
                <span className="text-unit">頭</span>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </section>
  );
}

// ── screen ────────────────────────────────────────────────────────────────────

/**
 * Race — カレンダーで日を選び、その日のレースを見る 1 画面。
 *
 * 「今週末」「Past」のタブは廃止した。カレンダーが開催日と取込状況の両方を
 * 示すので、タブで期間を切り替える意味がなくなったため。初期表示は
 * **今週末の開催日**（無ければ直近の取込済みの日 / 今日）を選んだ状態にする。
 *
 * AI 予想はここでは走らせない。1 レースあたり十数秒かかるうえ、一覧の段階では
 * どの馬をいくらで買うかまでは決められないため、レース詳細を開いて
 *「予想を見る」で実行する（旧「AI 予想を実行」ボタンも廃止）。
 */
export function Races() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const weekend = useThisWeekendRaces();

  // 今週末の開催日 (取込済みのもの)。無ければ今日。
  const weekendDate = useMemo(() => {
    const dates = [...new Set((weekend.data?.races ?? []).map((r) => r.date))].sort();
    if (dates.length === 0) return null;
    const today = todayIso();
    // 今日以降の開催日を優先し、無ければ最後の開催日
    return dates.find((d) => d >= today) ?? dates[dates.length - 1];
  }, [weekend.data]);

  const dateParam = searchParams.get('date');
  const selectedDate = dateParam ?? weekendDate ?? todayIso();

  // 初回に今週末が判明したら URL にも反映する (リロード・共有で同じ日を開ける)
  useEffect(() => {
    if (!dateParam && weekendDate) {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set('date', weekendDate);
        return next;
      }, { replace: true });
    }
  }, [dateParam, weekendDate, setSearchParams]);

  const { data, isPending, isError } = useRacesByDate(selectedDate);
  const sections = data ? groupByCourse(data.races) : [];

  function handleDateChange(value: string) {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      if (value) next.set('date', value);
      else next.delete('date');
      return next;
    }, { replace: true });
  }

  function handleRowClick(race: RaceSummary) {
    navigate(`/races/${race.race_id}?date=${selectedDate}`);
  }

  const raceList = isPending ? (
    <TableSkeleton />
  ) : isError ? (
    <EmptyState
      message="レース情報の取得に失敗しました"
      description="バックエンドが起動しているか確認してください。"
    />
  ) : sections.length === 0 ? (
    <EmptyState
      message="この日のレースはまだ取り込まれていません"
      description="開催が無かった日か、未取得の日です。左のボタンでこの日を取り込めます。"
    />
  ) : (
    <div className="flex flex-col gap-8">
      {sections.map((section) => (
        <RaceTable key={section.course} section={section} onRowClick={handleRowClick} />
      ))}
    </div>
  );

  return (
    <div className="flex flex-col gap-8 p-6">
      <PageHeader
        eyebrow="Race"
        title={formatDayLabel(selectedDate)}
        description="カレンダーから日を選ぶと、その日のレース一覧が出ます。予想は各レースを開いて実行します"
      />

      {/* いま手元にどれだけデータがあるかを最初に出す */}
      <DataCoverageBand />

      <div className="flex flex-col gap-8 lg:flex-row lg:items-start lg:gap-10">
        <div className="flex shrink-0 flex-col gap-4">
          <RaceCalendar value={selectedDate} onChange={handleDateChange} />
          {/* 取込は「どの日が空いているか」が見えるカレンダーの隣に置く */}
          <DayIngestPanel
            date={selectedDate}
            raceCount={sections.reduce((n, s) => n + s.races.length, 0)}
            hasResults={(data?.races ?? []).length > 0 && !isPending}
          />
        </div>
        <div className="min-w-0 flex-1">{raceList}</div>
      </div>
    </div>
  );
}
