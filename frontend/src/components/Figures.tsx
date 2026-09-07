import { InfoTip } from '@/components/InfoTip';
import { cn } from '@/lib/cn';

export interface Figure {
  label: string;
  /** 整形済みの値。未算出は '未算出' のような文字列で渡す。 */
  value: string;
  /** 値の**右にかっこ書き**で添えるもの (比べる相手など)。 */
  paren?: string;
  /** ラベル横の「?」に畳む定義。量の言い換えはこちら。 */
  hint?: string;
}

/**
 * カードにしない数値の並び。
 *
 * **答えは `MetricCard` 1 枚、残りはここ。**囲うのは「その画面の答えになる指標」
 * だけで（`docs/ui-style.md`「領域の作り方」）、補助の数値まで囲うと答えが埋もれる。
 * 枠も面も付けず、ラベルの下に値を置くだけにしてある。
 *
 * **3 画面で同じ形にするために切り出した** (2026-09-07)。収支台帳・シミュレーション
 * の結果・モデル画面は、どれも「収支の答え 1 つ + 補助の数値」という同じ構造なのに、
 * 5 枚のカード / 18px の `<dl>` / 12px の `<dl>` と 3 通りに書かれていた。
 */
export function Figures({ items, className }: { items: Figure[]; className?: string }) {
  return (
    <dl className={cn('flex flex-wrap gap-x-8 gap-y-3', className)}>
      {items.map(({ label, value, paren, hint }) => (
        <div key={label} className="flex min-w-0 flex-col gap-1">
          {/* 量の定義 (「予想1位が1着」など) は「?」に畳む。読まないと誤解するが、
              常時出ていると値より先に目に入る。 */}
          <dt className="flex items-center gap-1 text-label-ja">
            {label}
            {hint && <InfoTip label={label} text={hint} />}
          </dt>
          <dd className="text-num font-mono text-sm text-foreground">
            {value}
            {paren && <span className="ml-1 text-subtle-foreground">({paren})</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}
