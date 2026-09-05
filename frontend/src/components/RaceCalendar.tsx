import { useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

import { useRacesCalendar } from '@/hooks/useRacesCalendar';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/cn';
import type { CalendarDay } from '@/types/api';

interface RaceCalendarProps {
  /** 選択中の日 (ISO "YYYY-MM-DD")。 */
  value: string | undefined;
  /** 日をクリックしたときに ISO 文字列を返す。 */
  onChange: (date: string) => void;
  className?: string;
}

const DOW = ['日', '月', '火', '水', '木', '金', '土'] as const;

/** 重賞だけ色を持たせる。平場は無彩色 (色は情報なので安売りしない)。 */
function gradeClass(raceClass: string | null): string {
  if (!raceClass) return 'text-subtle-foreground';
  if (raceClass.includes('G1')) return 'text-primary';
  if (raceClass.includes('G2') || raceClass.includes('G3')) return 'text-foreground';
  return 'text-subtle-foreground';
}

/**
 * 「第169回天皇賞(春)」→「天皇賞(春)」。
 *
 * カレンダーで見たいのは「その日に何があるか」で、開催回数は要らない。
 * 付けたままだと幅の半分を「第169回」が占め、肝心の名前から先に省略される。
 */
function stripEdition(name: string): string {
  return name.replace(/^第\s*\d+\s*回/, '');
}

/** 曜日ヘッダと日セルで共有する列幅。日・土だけ広い (JRA は基本 土日開催)。 */
const DOW_GRID = 'grid grid-cols-[1.9fr_1fr_1fr_1fr_1fr_1fr_1.9fr]';

function pad(n: number): string {
  return String(n).padStart(2, '0');
}

function iso(y: number, m: number, d: number): string {
  return `${y}-${pad(m)}-${pad(d)}`;
}

function daysInMonth(y: number, m: number): number {
  return new Date(y, m, 0).getDate();
}

/** 「2026-08-23」から {y, m} を取る。不正なら今日の年月。 */
function parseMonth(value: string | undefined): { y: number; m: number } {
  const now = new Date();
  if (!value) return { y: now.getFullYear(), m: now.getMonth() + 1 };
  const [ys, ms] = value.split('-');
  const y = Number(ys);
  const m = Number(ms);
  if (!Number.isFinite(y) || !Number.isFinite(m)) {
    return { y: now.getFullYear(), m: now.getMonth() + 1 };
  }
  return { y, m };
}

/**
 * 月表示のレースカレンダー。
 *
 * 年・月・日のセレクトを 3 つ並べるより、月を一望して日をクリックするほうが
 * 「いつ開催があったか」「どこまで取り込めているか」が同時に読める。
 *
 * 各日のセルは 3 つの状態を持つ:
 *   - 未取得 … 薄く表示。選ぶと「この日を取り込む」導線が出る
 *   - 出馬表のみ … レース数を琥珀で (結果がまだ)
 *   - 結果あり … レース数を通常色で + 主要レース名
 */
export function RaceCalendar({ value, onChange, className }: RaceCalendarProps) {
  const [month, setMonth] = useState(() => parseMonth(value));

  const from = iso(month.y, month.m, 1);
  const to = iso(month.y, month.m, daysInMonth(month.y, month.m));
  const { data, isPending } = useRacesCalendar(from, to);

  const byDate = useMemo(() => {
    const map = new Map<string, CalendarDay>();
    for (const d of data?.days ?? []) map.set(d.date, d);
    return map;
  }, [data]);

  const todayIso = useMemo(() => {
    const t = new Date();
    return iso(t.getFullYear(), t.getMonth() + 1, t.getDate());
  }, []);

  const firstDow = new Date(month.y, month.m - 1, 1).getDay();
  const total = daysInMonth(month.y, month.m);
  // 先頭の空白 + 各日。週の区切りは grid が自動で折り返す。
  const cells: (number | null)[] = [
    ...Array.from({ length: firstDow }, () => null),
    ...Array.from({ length: total }, (_, i) => i + 1),
  ];

  function shift(delta: number) {
    setMonth((prev) => {
      const m = prev.m + delta;
      if (m < 1) return { y: prev.y - 1, m: 12 };
      if (m > 12) return { y: prev.y + 1, m: 1 };
      return { y: prev.y, m };
    });
  }

  const monthRaces = (data?.days ?? []).reduce((acc, d) => acc + d.race_count, 0);
  const monthDays = data?.days.length ?? 0;

  return (
    <div className={cn('w-full', className)}>
      {/* 月送り */}
      <div className="mb-2 flex items-center justify-between gap-2">
        <Button variant="ghost" size="sm" onClick={() => shift(-1)} aria-label="前の月">
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <div className="flex items-baseline gap-3">
          <span
            className="font-mono text-lg font-bold tabular-nums"
            aria-label="表示中の月"
          >
            {month.y}
            <span className="mx-0.5 text-subtle-foreground">/</span>
            {pad(month.m)}
          </span>
          <span className="font-mono text-xs tabular-nums text-subtle-foreground">
            {monthDays} 開催日 / {monthRaces} R
          </span>
        </div>
        <Button variant="ghost" size="sm" onClick={() => shift(1)} aria-label="次の月">
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>

      {/* 曜日ヘッダ */}
      {/* JRA は基本 土日開催。平日に同じ幅を割く理由がないので、
          日・土だけ約 2 倍取って重賞名を入る幅にする。列幅はヘッダと
          セルで同じ定義を使う (ずれると曜日と日付が噛み合わなくなる)。 */}
      <div className={cn(DOW_GRID, 'border-b border-border-strong')}>
        {DOW.map((d, i) => (
          <div
            key={d}
            className={cn(
              'pb-1 text-center font-mono text-xs',
              i === 0 && 'text-destructive',
              i === 6 && 'text-info',
              i !== 0 && i !== 6 && 'text-subtle-foreground'
            )}
          >
            {d}
          </div>
        ))}
      </div>

      {isPending ? (
        <Skeleton className="mt-2 h-64 w-full" />
      ) : (
        <div className={DOW_GRID}>
          {cells.map((day, i) => {
            if (day === null) return <div key={`pad-${i}`} className="border-b border-border" />;
            const dateIso = iso(month.y, month.m, day);
            const info = byDate.get(dateIso);
            const selected = value === dateIso;
            const isToday = todayIso === dateIso;
            const dow = (firstDow + day - 1) % 7;
            const hasData = info != null;
            const resultsPending = hasData && info.result_count === 0;

            return (
              <button
                key={dateIso}
                type="button"
                onClick={() => onChange(dateIso)}
                aria-label={
                  hasData
                    ? `${month.m}月${day}日 ${info.race_count}レース${
                        resultsPending ? '（結果未取得）' : ''
                      }`
                    : `${month.m}月${day}日 データなし`
                }
                aria-pressed={selected}
                title={
                  hasData
                    ? [
                        `${info.courses.join('・')} / ${info.race_count}R（結果 ${info.result_count}R）`,
                        info.highlight_name,
                      ]
                        .filter(Boolean)
                        .join('\n')
                    : undefined
                }
                className={cn(
                  // min-w-0 が要る: grid item は既定で min-width:auto なので、
                  // これが無いと重賞名が truncate されずセルごと横に膨らみ、
                  // 隣の列 (狭い画面では右のパネル) を押し出す。
                  'flex h-14 min-w-0 flex-col items-start gap-0.5 border-b border-r border-border px-3 py-1.5 text-left transition-colors',
                  '[&:nth-child(7n)]:border-r-0',
                  'cursor-pointer hover:bg-card-elevated',
                  // 未取得の日も選べる (選ぶと右側に取込ボタンが出る)
                  !hasData && 'opacity-40',
                  selected && 'bg-primary/15 ring-1 ring-inset ring-primary',
                  'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary'
                )}
              >
                {/* 両端揃えにしない。セルが広い日 (3 場グリッドに載せると 200px 前後)
                    だと「7 ……… 36R」と離れ、1 つの情報に見えなくなる。 */}
                <span className="flex items-baseline gap-2">
                  <span
                    className={cn(
                      'font-mono text-sm tabular-nums',
                      isToday && 'underline underline-offset-2',
                      dow === 0 && 'text-destructive',
                      dow === 6 && 'text-info',
                      dow !== 0 && dow !== 6 && 'text-muted-foreground'
                    )}
                  >
                    {day}
                  </span>
                  {hasData && (
                    <span
                      className={cn(
                        'font-mono text-2xs tabular-nums',
                        // 出馬表だけ = 結果がまだ、を琥珀で示す
                        resultsPending ? 'text-primary' : 'text-muted-foreground'
                      )}
                    >
                      {info.race_count}R
                    </span>
                  )}
                </span>
                {/* 重賞名。土日の列を約 2 倍取ってあるので入る。
                    「第169回」は落とす (stripEdition) — 開催回数は
                    カレンダーで見たい情報ではなく、付けたままだと幅の半分を
                    食って肝心の名前から先に省略される。 */}
                {hasData && info.highlight_name && (
                  <span
                    className={cn(
                      'w-full truncate text-2xs leading-tight',
                      gradeClass(info.highlight_class)
                    )}
                  >
                    {stripEdition(info.highlight_name)}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}

    </div>
  );
}
