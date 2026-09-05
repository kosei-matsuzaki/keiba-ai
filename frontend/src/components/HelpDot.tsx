import { cn } from '@/lib/cn';

/**
 * 見出しの横に置く「?」。ホバーで説明を出す。
 *
 * この画面群は指標と表が主役なので、注釈を本文として置くと**読む順序が
 * 分からなくなる**。読まなくても操作できる説明はここに畳む。
 * 逆に、読まないと誤解する数字 (回収率が 1.0 未満など) は本文に残す。
 */
export function HelpDot({ text, label, className }: {
  text: string;
  /** スクリーンリーダー向けの見出し。「◯◯ の説明」になる。 */
  label: string;
  className?: string;
}) {
  return (
    <span
      title={text}
      tabIndex={0}
      role="note"
      aria-label={`${label} の説明`}
      className={cn(
        'cursor-help select-none rounded-full border border-border px-1',
        'font-mono text-2xs leading-tight text-subtle-foreground',
        'hover:border-border-strong hover:text-muted-foreground',
        className
      )}
    >
      ?
    </span>
  );
}
