import { useMemo } from 'react';
import { Link } from 'react-router-dom';

import { useRacesByDate } from '@/hooks/useRacesByDate';
import { cn } from '@/lib/cn';
import { groupByCourse, isGraded, raceNumber, splitGrade } from '@/lib/races';

interface DayRacePickerProps {
  /** 表示中のレースの開催日 (ISO)。この日の全レースを出す。 */
  date: string;
  /** いま開いているレース。ここだけ印を付ける。 */
  currentRaceId: string;
}

/**
 * その日の全レースへ飛ぶための 1 行。**場をまたいで移動できる唯一の導線**。
 *
 * 1 日は最大 3 場あり、race_id の末尾を ±1 する前後移動では同じ開催の中しか
 * 動けない。「東京の 11R を見たあと京都の 11R を見る」ができないと、
 * そのたび一覧へ戻ることになる。
 *
 * 場を縦に積まず**横に並べる**。3 行取ると本文の前に厚い帯ができて、
 * 何の画面か分かる前にナビが目に入る。1 行なら余白だけで済む。
 *
 * 罫線は引かない。親が gap-8 で離しており、すぐ下の PageHeader も自前の罫線を
 * 持つので、足すと線が 2 本続く。
 *
 * 番号だけを並べるのは、レース名まで出すと 36 本で画面が埋まるため。ただし
 * **重賞だけは色を持たせる** — 別の場へ飛ぶ動機はたいてい「そっちのメイン」なので、
 * どれがメインか分からないと結局一覧へ戻ることになる。名前は hover に置く。
 */
export function DayRacePicker({ date, currentRaceId }: DayRacePickerProps) {
  const { data } = useRacesByDate(date);

  const sections = useMemo(() => groupByCourse(data?.races ?? []), [data]);

  if (sections.length === 0) return null;

  return (
    <nav
      aria-label="この日のレース"
      className="flex flex-wrap items-baseline gap-x-8 gap-y-2"
    >
      {sections.map(({ course, races }) => {
        const here = races.some((r) => r.race_id === currentRaceId);
        return (
          <div key={course} className="flex min-w-0 items-baseline gap-2">
            {/* いま見ている場だけ前景色。どこにいるかを場の側でも示す */}
            <span
              className={cn(
                'shrink-0 text-2xs',
                here ? 'text-foreground' : 'text-subtle-foreground'
              )}
            >
              {course}
            </span>
            <div className="flex min-w-0 flex-wrap items-baseline gap-x-0.5">
              {races.map((r) => {
                const current = r.race_id === currentRaceId;
                const graded = isGraded(splitGrade(r.name ?? '', r.race_class ?? '').grade);
                const n = raceNumber(r.race_id);
                return (
                  <Link
                    key={r.race_id}
                    to={`/races/${r.race_id}?date=${date}`}
                    title={r.name ?? undefined}
                    aria-current={current ? 'page' : undefined}
                    className={cn(
                      // 余白は現在地かどうかに関わらず同じ。印の有無で番号が
                      // 左右にずれると、並びを目で追えなくなる。
                      'rounded-sm px-1 font-mono text-2xs tabular-nums transition-colors',
                      current && 'bg-primary/20 font-medium text-primary',
                      !current && graded && 'font-medium text-foreground hover:bg-card-elevated',
                      !current &&
                        !graded &&
                        'text-subtle-foreground hover:bg-card-elevated hover:text-foreground'
                    )}
                  >
                    {n}
                  </Link>
                );
              })}
            </div>
          </div>
        );
      })}
    </nav>
  );
}
