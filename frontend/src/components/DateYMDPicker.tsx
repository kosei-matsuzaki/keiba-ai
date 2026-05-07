import { useMemo } from 'react';

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/cn';

interface DateYMDPickerProps {
  /** ISO 形式 "YYYY-MM-DD"。空文字 / undefined のときは「未指定」表示。 */
  value: string | undefined;
  /** 新しい値 (ISO "YYYY-MM-DD") を返す。「未指定」状態は空文字 ""。 */
  onChange: (value: string) => void;
  /** 年セレクトの最小値。default = 当年-5 */
  minYear?: number;
  /** 年セレクトの最大値。default = 当年+1 */
  maxYear?: number;
  /** 「未指定」選択肢を表示するか。default = false (3 列とも常に値あり) */
  allowEmpty?: boolean;
  className?: string;
  /** 各 Select に渡す aria-label */
  ariaLabel?: string;
  disabled?: boolean;
}

const _PLACEHOLDER_YEAR = '----';
const _PLACEHOLDER_MONTH = '--';
const _PLACEHOLDER_DAY = '--';

const _DOW_CHARS = ['日', '月', '火', '水', '木', '金', '土'] as const;

function _daysInMonth(year: number, month: number): number {
  // month: 1-12
  return new Date(year, month, 0).getDate();
}

function _dayOfWeek(year: number, month: number, day: number): number {
  // 0=Sun, 6=Sat
  return new Date(year, month - 1, day).getDay();
}

/** JRA は基本土日開催。土曜=青 / 日曜=赤 で見つけやすくする。 */
function _dowColorClass(dow: number | null): string {
  if (dow === 0) return 'text-red-600 dark:text-red-400';
  if (dow === 6) return 'text-blue-600 dark:text-blue-400';
  return '';
}

function _parse(value: string | undefined): {
  year: number | null;
  month: number | null;
  day: number | null;
} {
  if (!value) return { year: null, month: null, day: null };
  const [y, m, d] = value.split('-');
  const year = Number(y);
  const month = Number(m);
  const day = Number(d);
  return {
    year: Number.isFinite(year) ? year : null,
    month: Number.isFinite(month) ? month : null,
    day: Number.isFinite(day) ? day : null,
  };
}

function _format(year: number, month: number, day: number): string {
  return `${String(year).padStart(4, '0')}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
}

/**
 * 年 / 月 / 日 を 3 つの Select で入力するピッカー。
 * native の <input type="date"> がカレンダー UI 起点で過去日選択がしづらい問題への対応。
 *
 * 出力は ISO "YYYY-MM-DD" 文字列で、`value` も同じ形式を期待する。
 * 月末を超える日は自動的にその月の末日へ clamp する (例: 1/31 → 2/28)。
 */
export function DateYMDPicker({
  value,
  onChange,
  minYear,
  maxYear,
  allowEmpty = false,
  className,
  ariaLabel,
  disabled,
}: DateYMDPickerProps) {
  const today = useMemo(() => new Date(), []);
  const _minYear = minYear ?? today.getFullYear() - 5;
  const _maxYear = maxYear ?? today.getFullYear() + 1;

  const { year, month, day } = _parse(value);

  const years = useMemo(() => {
    const out: number[] = [];
    for (let y = _maxYear; y >= _minYear; y--) out.push(y);
    return out;
  }, [_minYear, _maxYear]);

  const months = useMemo(() => {
    const out: number[] = [];
    for (let m = 1; m <= 12; m++) out.push(m);
    return out;
  }, []);

  const days = useMemo(() => {
    const upper = year && month ? _daysInMonth(year, month) : 31;
    const out: { day: number; dow: number | null }[] = [];
    for (let d = 1; d <= upper; d++) {
      const dow = year && month ? _dayOfWeek(year, month, d) : null;
      out.push({ day: d, dow });
    }
    return out;
  }, [year, month]);

  function _emit(nextY: number | null, nextM: number | null, nextD: number | null) {
    if (nextY === null || nextM === null || nextD === null) {
      if (allowEmpty) onChange('');
      return;
    }
    // 月末を超える日は clamp (例: 2/31 → 2/28)
    const maxDay = _daysInMonth(nextY, nextM);
    const clampedDay = Math.min(nextD, maxDay);
    onChange(_format(nextY, nextM, clampedDay));
  }

  return (
    <div className={cn('flex items-center gap-1.5', className)}>
      <Select
        value={year !== null ? String(year) : ''}
        onValueChange={(v) => _emit(Number(v), month, day)}
        disabled={disabled}
      >
        <SelectTrigger
          className="w-[88px]"
          aria-label={ariaLabel ? `${ariaLabel} 年` : '年'}
        >
          <SelectValue placeholder={_PLACEHOLDER_YEAR} />
        </SelectTrigger>
        <SelectContent>
          {years.map((y) => (
            <SelectItem key={y} value={String(y)}>
              {y}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <span className="text-sm text-muted-foreground">年</span>

      <Select
        value={month !== null ? String(month) : ''}
        onValueChange={(v) => _emit(year, Number(v), day)}
        disabled={disabled}
      >
        <SelectTrigger
          className="w-[68px]"
          aria-label={ariaLabel ? `${ariaLabel} 月` : '月'}
        >
          <SelectValue placeholder={_PLACEHOLDER_MONTH} />
        </SelectTrigger>
        <SelectContent>
          {months.map((m) => (
            <SelectItem key={m} value={String(m)}>
              {m}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <span className="text-sm text-muted-foreground">月</span>

      <Select
        value={day !== null ? String(day) : ''}
        onValueChange={(v) => _emit(year, month, Number(v))}
        disabled={disabled}
      >
        <SelectTrigger
          className="w-[88px]"
          aria-label={ariaLabel ? `${ariaLabel} 日` : '日'}
        >
          <SelectValue placeholder={_PLACEHOLDER_DAY} />
        </SelectTrigger>
        <SelectContent>
          {days.map(({ day: d, dow }) => {
            const dowChar = dow !== null ? _DOW_CHARS[dow] : null;
            return (
              <SelectItem key={d} value={String(d)}>
                <span className={_dowColorClass(dow)}>
                  {d}
                  {dowChar !== null && ` (${dowChar})`}
                </span>
              </SelectItem>
            );
          })}
        </SelectContent>
      </Select>
      <span className="text-sm text-muted-foreground">日</span>
    </div>
  );
}
