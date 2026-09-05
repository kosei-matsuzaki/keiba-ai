import { useEffect, useMemo } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

import { useRacesByDate } from '@/hooks/useRacesByDate';
import { useThisWeekendRaces } from '@/hooks/useThisWeekendRaces';
import { DataCoverageBand } from '@/components/DataCoverageBand';
import { DayIngestPanel } from '@/components/DayIngestPanel';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { RaceCalendar } from '@/components/RaceCalendar';
import { Button } from '@/components/ui/button';
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
        className="mb-2 flex items-center gap-4 font-mono text-2xs tracking-[0.18em] text-subtle-foreground"
      >
        {section.course}
        <span className="h-px flex-1 bg-border" aria-hidden="true" />
        <span className="tabular-nums">{section.races.length} R</span>
      </h3>
      <Table>
        <TableHeader>
          <TableRow>
            {/* レース名以外に固定幅を与え、余りを全部レース名に回す。
                auto 配分だと「第169回天皇賞(春)」が真っ先に削られる。 */}
            <TableHead className="w-10 text-label">R</TableHead>
            <TableHead>レース名</TableHead>
            <TableHead className="w-[5.25rem] whitespace-nowrap">クラス</TableHead>
            {/* 馬場と距離は 1 列。3 場を横に並べると 1 列 460px 前後しか無く、
                6 列だとレース名が折り返して行高が不揃いになる。
                「芝1600m」は番組表の読み方そのものなので、畳んでも読みは落ちない。 */}
            <TableHead className="w-[5.5rem] whitespace-nowrap text-right">馬場・距離</TableHead>
            <TableHead className="w-12 whitespace-nowrap text-right">頭数</TableHead>
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
              <TableCell className="max-w-0 truncate" title={race.name ?? undefined}>
                {race.name ?? <span className="text-subtle-foreground/50">·</span>}
              </TableCell>
              {/* クラスは Badge をやめて素のテキスト。G1 だけ色を持たせて格を出す */}
              <TableCell
                className={
                  race.race_class === 'G1'
                    ? 'whitespace-nowrap font-mono text-2xs text-primary'
                    : 'whitespace-nowrap font-mono text-2xs text-subtle-foreground'
                }
              >
                {race.race_class ?? '·'}
              </TableCell>
              <TableCell className="cell-num whitespace-nowrap">
                <span className="text-unit">{race.surface}</span>
                {race.distance}
                <span className="text-unit">m</span>
              </TableCell>
              <TableCell className="cell-num whitespace-nowrap">
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
/**
 * 今週末のレースがまだ取り込まれていないことを、**取り込む画面**で知らせる。
 *
 * 以前は Dashboard に出していたが、そこから「レース一覧へ」を踏んで日を選んで
 * 取り込む、と動線が長かった。ここなら知らせの下がそのまま取込操作になる。
 * 取り込めているときは何も出さない。
 */
function WeekendIngestNotice({ onJump }: { onJump: (date: string) => void }) {
  const weekend = useThisWeekendRaces();
  if (weekend.isPending || (weekend.data?.races.length ?? 0) > 0) return null;

  // 直近の土曜 (無ければ今日) を取込先の候補にする
  const today = new Date();
  const toSat = (6 - today.getDay() + 7) % 7;
  const sat = new Date(today);
  sat.setDate(today.getDate() + toSat);
  const target = sat.toISOString().slice(0, 10);

  return (
    <div className="flex flex-col items-start gap-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-label-ja">今週末</span>
        <span className="font-mono text-2xs text-subtle-foreground">未取得</span>
      </div>
      <Button size="sm" variant="outline" onClick={() => onJump(target)}>
        {target} を開く
      </Button>
    </div>
  );
}

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
    // 場ごとの表を横に並べる。JRA は 1 日最大 3 場なので、3 列に流すと
    // 12R × 3 場がスクロールなしで一望でき、場をまたいだ比較がそのままできる。
    // 以前 2 列を試したときはレース名と頭数が折り返したが、それは右半分しか
    // 使えていなかったため。カレンダーを上に出して横幅を全部渡し、表側で
    // 馬場と距離を 1 列に畳んだので 1 列あたり 460px 前後を確保できる。
    <div className="grid grid-cols-1 gap-x-8 gap-y-8 md:grid-cols-2 xl:grid-cols-3">
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

      {/* 日を選ぶ道具は、下のレース一覧と**同じ 3 列グリッド**に載せる。
          カレンダーが 2 列ぶん・取込パネルが 3 列目で、左右の端が下の表と
          揃う。中央に細い柱として置くと、左右に 180px ずつ空いたまま
          「余白」ではなく「余り」に見える。 */}
      <div className="grid grid-cols-1 gap-x-8 gap-y-6 md:grid-cols-2 xl:grid-cols-3">
        {/* カレンダーは列だけを占め、上下に何も敷かない。月表示は行数が
            決まっているので、上下に足したものはそのままセルの高さを削る。 */}
        <div className="min-w-0 xl:col-span-2">
          <RaceCalendar value={selectedDate} onChange={handleDateChange} />
        </div>
        {/* カレンダーに添えるものはこの列に縦積みにする。どちらも
            「見出し → 状態 → 操作」の同じ並びで、区切りは余白と見出しだけ —
            罫線を足すと 3 つ目の区切りになり、左のカレンダーにも無い線が
            ここだけ入る。選んだ日が主、今週末の知らせが従 (見ていない日の話)。 */}
        <div className="flex min-w-0 flex-col gap-10">
          <DayIngestPanel
            date={selectedDate}
            raceCount={sections.reduce((n, s) => n + s.races.length, 0)}
            hasResults={(data?.races ?? []).length > 0 && !isPending}
          />
          <WeekendIngestNotice onJump={handleDateChange} />
        </div>
      </div>

      <div className="min-w-0">{raceList}</div>
    </div>
  );
}
