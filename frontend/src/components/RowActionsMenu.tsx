import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import { MoreHorizontal } from 'lucide-react';

import { useAnchoredPosition } from '@/hooks/useAnchoredPosition';
import { cn } from '@/lib/cn';

/**
 * 行の操作を三点リーダーに畳むメニュー。
 *
 * ボタンを 5 つ横に並べると、行の主役 (数字) より操作の方が目立ってしまう。
 * **操作は探せば見つかればよい**ので、普段は「⋯」1 つに畳む。
 */
export interface RowAction {
  label: string;
  onSelect: () => void;
  disabled?: boolean;
  /** 選べない理由や補足。項目の下に小さく出す。 */
  hint?: string;
  /** 破壊的な操作 (削除) は色を変え、区切りの下に置く。 */
  tone?: 'default' | 'destructive';
}

export function RowActionsMenu({
  actions,
  label = '操作',
}: {
  actions: RowAction[];
  label?: string;
}) {
  const [open, setOpen] = useState(false);
  /** キーボードで動かしている項目。-1 = まだ触っていない。 */
  const [active, setActive] = useState(-1);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const hintId = useId();
  // 右端に寄せる。表の右端の列に出るので、左揃えだと画面外へ出る。
  // 画面下寄りの行では上に出る (下に描くと見切れる)。
  const pos = useAnchoredPosition(triggerRef, menuRef, open, { align: 'end', gap: 4 });

  // 破壊的な操作は必ず最後にまとめる。並び順を呼び出し側に委ねると、
  // 「削除」が真ん中に来た行と来ない行ができて手が滑る。
  const ordered = [
    ...actions.filter((a) => a.tone !== 'destructive'),
    ...actions.filter((a) => a.tone === 'destructive'),
  ];
  const firstDestructive = ordered.findIndex((a) => a.tone === 'destructive');

  function close(focusTrigger = true) {
    setOpen(false);
    setActive(-1);
    // 閉じたあとフォーカスが body に落ちると、Tab の位置が先頭に戻る。
    if (focusTrigger) triggerRef.current?.focus();
  }

  /** 使える項目だけを行き来する (無効な項目で止まると進めなくなる)。 */
  function move(from: number, step: number) {
    const n = ordered.length;
    for (let i = 1; i <= n; i += 1) {
      const next = (from + step * i + n * n) % n;
      if (!ordered[next].disabled) return next;
    }
    return from;
  }

  useEffect(() => {
    if (!open) return;
    function onPointerDown(e: MouseEvent) {
      const t = e.target as Node;
      if (menuRef.current?.contains(t) || triggerRef.current?.contains(t)) return;
      close(false);
    }
    document.addEventListener('mousedown', onPointerDown);
    return () => document.removeEventListener('mousedown', onPointerDown);
  }, [open]);

  // 矢印で選んだ項目に実際のフォーカスを移す。role="menu" の読み上げは
  // 「n 個中 k 個目」を出すので、視覚的な強調だけでは足りない。
  //
  useEffect(() => {
    if (open && active >= 0) itemRefs.current[active]?.focus();
  }, [open, active]);

  function onTriggerKeyDown(e: KeyboardEvent) {
    if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      setOpen(true);
      setActive(move(-1, 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setOpen(true);
      setActive(move(0, -1));
    }
  }

  function onMenuKeyDown(e: KeyboardEvent) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((i) => move(i, 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((i) => move(i, -1));
    } else if (e.key === 'Home') {
      e.preventDefault();
      setActive(move(-1, 1));
    } else if (e.key === 'End') {
      e.preventDefault();
      setActive(move(0, -1));
    } else if (e.key === 'Escape' || e.key === 'Tab') {
      e.preventDefault();
      close();
    }
  }

  return (
    <div className="flex justify-end">
      <button
        ref={triggerRef}
        type="button"
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => (open ? close() : setOpen(true))}
        onKeyDown={onTriggerKeyDown}
        className={cn(
          'inline-flex h-8 w-8 items-center justify-center rounded-sm',
          'text-muted-foreground transition-colors hover:bg-card-elevated hover:text-foreground',
          'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
          open && 'bg-card-elevated text-foreground'
        )}
      >
        <MoreHorizontal className="h-4 w-4" />
      </button>

      {open &&
        createPortal(
          <div
            ref={menuRef}
            role="menu"
            aria-label={label}
            onKeyDown={onMenuKeyDown}
            // fixed + portal。表は横スクロールする枠の中にあるので、absolute だと
            // 右端の行で切れる。画面下寄りの行では上に出る。寸法を測ってから
            // 位置を決めるので、決まるまでは隠したまま描く。
            style={{ top: pos.top, left: pos.left, visibility: pos.ready ? undefined : 'hidden' }}
            className={cn(
              'fixed z-50 min-w-[12rem] overflow-hidden rounded-sm border border-border',
              'bg-popover py-1 text-popover-foreground shadow-lg'
            )}
          >
            {ordered.map((a, i) => (
              <div key={a.label}>
                {/* 破壊的な操作の手前にだけ区切りを 1 本。誤って押す距離を作る。 */}
                {i === firstDestructive && i > 0 && (
                  <div role="separator" className="my-1 h-px bg-border" />
                )}
                <button
                  ref={(el) => (itemRefs.current[i] = el)}
                  type="button"
                  role="menuitem"
                  tabIndex={-1}
                  disabled={a.disabled}
                  // 名前は操作そのものだけ。hint まで名前に入れると
                  // 「Activate このモデルで買い目を決めるようにする」になる。
                  aria-label={a.label}
                  aria-describedby={a.hint ? `${hintId}-${i}` : undefined}
                  onClick={() => {
                    close();
                    a.onSelect();
                  }}
                  className={cn(
                    'flex w-full flex-col items-start gap-0.5 px-3 py-2 text-left transition-colors',
                    'focus-visible:outline-none',
                    'hover:bg-card-elevated focus:bg-card-elevated',
                    'disabled:cursor-not-allowed disabled:bg-transparent disabled:opacity-40',
                    a.tone === 'destructive' ? 'text-destructive' : 'text-foreground'
                  )}
                >
                  <span className="text-sm">{a.label}</span>
                  {/* 選べない理由は畳まずその場に出す。ここを `title` に入れていた
                      ころは、押せない理由がホバーしないと分からなかった。 */}
                  {a.hint && (
                    <span
                      id={`${hintId}-${i}`}
                      className="text-2xs leading-snug text-subtle-foreground"
                    >
                      {a.hint}
                    </span>
                  )}
                </button>
              </div>
            ))}
          </div>,
          document.body
        )}
    </div>
  );
}
