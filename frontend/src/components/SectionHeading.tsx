import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';

interface SectionHeadingProps {
  /** 2 = 画面直下の節、3 = 節の中の小見出し。 */
  level?: 2 | 3;
  /**
   * 見出しから右へ罫線を伸ばす。**既定は無し。**
   *
   * 引くのは「どこまでが 1 つの塊か」が余白では読めないときだけ — いまは
   * 横に並べた列の分かれ目（モデル画面の運用中 2 列）に使っている。罫線は
   * 縦に積むと縞になって内容より先に目に付くので、節ごとに引かない。
   */
  rule?: boolean;
  /** 見出しの文字。アイコンやバッジを添えるならここに並べる。 */
  children: ReactNode;
  /** 見出しの右端に置くもの（操作ボタン・件数・開閉の印など）。 */
  aside?: ReactNode;
  className?: string;
}

/**
 * 節の見出し。**この形以外で節の見出しを書かない。**
 *
 * **この部品を作ったのは、見出しが 3 通りに割れていたから** (2026-09-06)。
 * `CardTitle` は `text-lg font-semibold` (18px) で、呼び出し側が 11 箇所すべてで
 * `className="text-label-ja"` を足していた。ところが `.text-label-ja` は
 * `globals.css` の `@layer base` にあり、Tailwind の utilities より**先**に出る。
 * 後勝ちなので `text-lg` の 18px が残り、`.text-label-ja` からは色 (subtle) だけが
 * 効いていた。**エラーは出ず、Ledger の小見出しだけが大きく灰色**になっていた。
 * 段は `docs/ui-style.md`「字の尺度」の h2 / h3 に従う。
 *
 * **`aside` は見出しタグの外に置く。**中に入れると、ボタンの文字まで見出しの
 * 読み上げ名に混ざる（「手持ちのモデル ID を詰める 再学習を実行」になっていた）。
 */
export function SectionHeading({
  level = 2,
  rule = false,
  children,
  aside,
  className,
}: SectionHeadingProps) {
  const Tag = level === 3 ? 'h3' : 'h2';
  return (
    // flex-1 は、CardHeader のように親が flex-row のときに幅を取るため。
    // 親が block なら無視される。
    <div className={cn('flex min-w-0 flex-1 items-center gap-3', className)}>
      <Tag
        className={cn(
          'flex min-w-0 items-center gap-2',
          level === 3 ? 'text-sm font-medium' : 'text-base font-semibold tracking-tight'
        )}
      >
        {children}
      </Tag>
      {/* 罫線を引かないときも同じ隙間を置く。aside が右端に寄る位置は
          rule の有無で変わらないほうがよい。 */}
      <span className={cn('flex-1', rule && 'h-px bg-border')} aria-hidden="true" />
      {aside}
    </div>
  );
}
