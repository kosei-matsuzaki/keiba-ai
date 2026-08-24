import { cn } from '@/lib/cn';
import { wakuColor, wakuOf } from '@/lib/waku';

interface UmabanProps {
  /** 馬番。null / undefined なら未確定として控えめな枠だけ出す。 */
  n: number | null | undefined;
  /** 出走頭数。枠番の導出に要る (8 頭以下は馬番 = 枠番)。 */
  runners: number;
  size?: 'sm' | 'md';
  className?: string;
}

const SIZE_CLASS = {
  sm: 'h-5 w-5 text-[11px]',
  md: 'h-6 w-6 text-[12px]',
} as const;

/**
 * 枠色付きの馬番チップ。
 *
 * 枠色は競馬固有の語彙で、利用者は枠で馬を識別する。数字だけを並べるより
 * 速く読め、一覧でも詳細でも同じ見え方になる。
 */
export function Umaban({ n, runners, size = 'md', className }: UmabanProps) {
  const base = cn(
    'inline-flex shrink-0 items-center justify-center rounded-[2px] font-mono font-bold tabular-nums',
    SIZE_CLASS[size],
    className
  );

  if (n == null || !Number.isFinite(n)) {
    return (
      <span className={cn(base, 'border border-border text-subtle-foreground/50')} aria-hidden="true">
        ·
      </span>
    );
  }

  const waku = wakuOf(n, runners);
  const color = wakuColor(waku);
  if (!color) {
    return <span className={cn(base, 'border border-border text-foreground')}>{n}</span>;
  }

  return (
    <span
      className={base}
      style={{ background: color.bg, color: color.fg }}
      title={`${waku}枠 ${color.name} / ${n}番`}
    >
      {n}
    </span>
  );
}
