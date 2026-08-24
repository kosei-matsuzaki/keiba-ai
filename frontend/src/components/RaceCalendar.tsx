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

/** 重賞だけ色を持たせる。平場は無彩色 (色は情報なので安売りしない)。 */
function gradeClass(raceClass: string | null): string {
  if (!raceClass) return 'text-subtle-foreground';
  if (raceClass.includes('G1')) return 'text-primary';
  if (raceClass.includes('G2') || raceClass.includes('G3')) return 'text-foreground';
  return 'text-subtle-foreground';
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
    <div className={cn('w-full max-w-md', className)}>
      {/* 月送り */}
      <div className="mb-2 flex items-center justify-between gap-2">
        <Button variant="ghost" size="sm" onClick={() => shift(-1)} aria-label="前の月">
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <div className="flex items-baseline gap-3">
          <span
            className="font-mono text-[15px] font-bold tabular-nums"
            aria-label="表示中の月"
          >
            {month.y}
            <span className="mx-0.5 text-subtle-foreground">/</span>
            {pad(month.m)}
          </span>
          <span className="font-mono text-[10px] tabular-nums text-subtle-foreground">
            {monthDays} 開催日 / {monthRaces} R
          </span>
        </div>
        <Button variant="ghost" size="sm" onClick={() => shift(1)} aria-label="次の月">
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>

      {/* 曜日ヘッダ */}
      <div className="grid grid-cols-7 border-b border-border-strong">
        {DOW.map((d, i) => (
          <div
            key={d}
            className={cn(
              'pb-1 text-center font-mono text-[10px]',
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
        <Skeleton className="mt-2 h-56 w-full" />
      ) : (
        <div className="grid grid-cols-7">
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
                    ? `${info.courses.join('・')} / ${info.race_count}R（結果 ${info.result_count}R）`
                    : undefined
                }
                className={cn(
                  'flex h-14 flex-col items-start gap-0.5 border-b border-r border-border px-1.5 py-1 text-left transition-colors',
                  '[&:nth-child(7n)]:border-r-0',
                  'cursor-pointer hover:bg-card-elevated',
                  // 未取得の日も選べる (選ぶと右側に取込ボタンが出る)
                  !hasData && 'opacity-40',
                  selected && 'bg-primary/15 ring-1 ring-inset ring-primary',
                  'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary'
                )}
              >
                <span
                  className={cn(
                    'font-mono text-[11px] tabular-nums',
                    isToday && 'underline underline-offset-2',
                    dow === 0 && 'text-destructive',
                    dow === 6 && 'text-info',
                    dow !== 0 && dow !== 6 && 'text-muted-foreground'
                  )}
                >
                  {day}
                </span>
                {hasData && (
                  <>
                    <span
                      className={cn(
                        'font-mono text-[10px] tabular-nums',
                        // 出馬表だけ = 結果がまだ、を琥珀で示す
                        resultsPending ? 'text-primary' : 'text-foreground'
                      )}
                    >
                      {info.race_count}R
                    </span>
                    {info.highlight_name && (
                      <span
                        className={cn(
                          'w-full truncate text-[9px] leading-tight',
                          gradeClass(info.highlight_class)
                        )}
                      >
                        {info.highlight_name}
                      </span>
                    )}
                  </>
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* 凡例 — セルの色が何を意味するかを明示する */}
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10px] text-subtle-foreground">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 border border-border-strong bg-card" />
          結果あり
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 border border-primary/60 bg-primary/20" />
          出馬表のみ
        </span>
        <span className="flex items-center gap-1 opacity-50">
          <span className="inline-block h-2 w-2 border border-border" />
          未取得
        </span>
      </div>
    </div>
  );
}
