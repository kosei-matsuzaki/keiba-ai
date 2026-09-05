import { useEffect, useRef, useState } from 'react';
import { MoreHorizontal } from 'lucide-react';

import { cn } from '@/lib/cn';

/**
 * 行の操作を三点リーダーに畳むメニュー。
 *
 * ボタンを 5 つ横に並べると、行の主役 (数字) より操作の方が目立ってしまう。
 * **操作は探せば見つかればよい**ので、普段は「⋯」1 つに畳む。
 *
 * ポップオーバーの類は入れていないので、外側クリックと Esc で閉じる素の実装。
 */
export interface RowAction {
  label: string;
  onSelect: () => void;
  disabled?: boolean;
  /** 選べない理由や補足。ホバーで出す。 */
  title?: string;
  /** 破壊的な操作 (削除) は色を変える。 */
  tone?: 'default' | 'destructive';
}

export function RowActionsMenu({ actions, label = '操作' }: {
  actions: RowAction[];
  label?: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDocClick(e: MouseEvent) {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false);
    }
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative flex justify-end">
      <button
        type="button"
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className={cn(
          'inline-flex h-8 w-8 items-center justify-center rounded-sm',
          'text-muted-foreground transition-colors hover:bg-card-elevated hover:text-foreground',
          open && 'bg-card-elevated text-foreground'
        )}
      >
        <MoreHorizontal className="h-4 w-4" />
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 top-9 z-20 min-w-[11rem] rounded-sm border border-border bg-card py-1 shadow-lg"
        >
          {actions.map((a) => (
            <button
              key={a.label}
              type="button"
              role="menuitem"
              title={a.title}
              disabled={a.disabled}
              onClick={() => {
                setOpen(false);
                a.onSelect();
              }}
              className={cn(
                'block w-full px-3 py-2 text-left text-sm transition-colors',
                'hover:bg-card-elevated disabled:cursor-not-allowed disabled:opacity-40',
                a.tone === 'destructive' ? 'text-destructive' : 'text-foreground'
              )}
            >
              {a.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
