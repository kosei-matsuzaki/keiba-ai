import { clsx, type ClassValue } from 'clsx';
import { extendTailwindMerge } from 'tailwind-merge';

/**
 * `globals.css` が持つ「型」クラス。Tailwind のユーティリティではなく、
 * 大きさ・書体・太さ・字間・数字の揃えまで含んだ 1 つの型 (docs/ui-style.md「字の尺度」)。
 *
 * **tailwind-merge に教えないと黙って消える。**既定の設定は `text-*` を
 * 「t シャツサイズか数値なら font-size、それ以外は文字色」と読むので、
 * `text-kpi` は文字色に分類され、同じ `cn()` に入った `text-foreground` に
 * 上書きされて**クラスごと落ちる**。エラーは出ず、26px の KPI が本文サイズで
 * 出るだけなので画面を見るまで気づけない (実際 `MetricCard` / `MetricBand` の
 * KPI がずっとこの状態だった)。
 *
 * 1 クラス 1 グループにしてあるのは、これらが互いに排他ではないから
 * (`text-num` と `text-unit` は同時に当てられる)。
 */
const KEIBA_TYPE_CLASSES = ['text-kpi', 'text-label', 'text-label-ja', 'text-num', 'text-unit'];

const twMerge = extendTailwindMerge({
  extend: {
    classGroups: Object.fromEntries(
      KEIBA_TYPE_CLASSES.map((name) => [`keiba-${name}`, [name]])
    ),
  },
});

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
