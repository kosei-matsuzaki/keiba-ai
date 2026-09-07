import { useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { useAnchoredPosition } from '@/hooks/useAnchoredPosition';
import { cn } from '@/lib/cn';

interface InfoTipProps {
  /** 何の説明かを読み上げに渡す。「◯◯ の説明」になる。 */
  label: string;
  /** 吹き出しの中身。読まなくても操作できることだけを入れる。 */
  text: string;
  className?: string;
}

/**
 * ラベルの右に置く「?」。ホバーとキーボードフォーカスで説明を出す。
 *
 * **数字の定義はここに畳む。**「予想1位が1着」「上位3頭のうち1頭以上が3着以内」
 * のような**量の定義**は、読まないと誤解するが常時出ていると値より先に目に入る。
 * 一方で**窓 (出所・期間・レース数) は畳まない** — 回収率は測り直すたびに動くので、
 * 見えていないと数字そのものが読めなくなる。
 *
 * `title` 属性ではなく自前で出しているのは 3 つの理由から:
 *   - ネイティブの吹き出しは**キーボードでは出ない** (フォーカスでは出ず、ホバー限定)
 *   - 遅延が OS 任せで、和文が折り返さない
 *   - 配色がテーマに追従しない
 *
 * portal で `position: fixed` に出すのは、`MetricCard` が `overflow-hidden` で、
 * 中に置くと**吹き出しが下辺で切れる**ため。画面の下寄りでは上に出す
 * (`useAnchoredPosition`)。
 */
export function InfoTip({ label, text, className }: InfoTipProps) {
  const [open, setOpen] = useState(false);
  const anchorRef = useRef<HTMLButtonElement>(null);
  const tipRef = useRef<HTMLSpanElement>(null);
  const pos = useAnchoredPosition(anchorRef, tipRef, open);
  const id = useId();

  return (
    <>
      <button
        ref={anchorRef}
        type="button"
        aria-label={`${label} の説明`}
        aria-expanded={open}
        aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => e.key === 'Escape' && setOpen(false)}
        // クリックでも閉じない/開かないと、触ったのに反応が無いように見える
        onClick={() => setOpen((v) => !v)}
        className={cn(
          'inline-flex h-4 w-4 shrink-0 cursor-help items-center justify-center rounded-full',
          'border border-border font-mono text-2xs leading-none text-subtle-foreground',
          'transition-colors hover:border-border-strong hover:text-muted-foreground',
          'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
          className
        )}
      >
        ?
      </button>
      {open &&
        createPortal(
          // 寸法を測ってから位置を決めるので、決まるまでは隠したまま描く。
          <span
            ref={tipRef}
            id={id}
            role="tooltip"
            style={{ top: pos.top, left: pos.left, visibility: pos.ready ? undefined : 'hidden' }}
            className={cn(
              'pointer-events-none fixed z-50 max-w-[18rem] rounded-sm border border-border',
              'bg-popover px-3 py-2 text-2xs leading-relaxed text-popover-foreground shadow-lg'
            )}
          >
            {text}
          </span>,
          document.body
        )}
    </>
  );
}
