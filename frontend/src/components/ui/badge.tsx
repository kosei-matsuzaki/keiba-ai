import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/cn';

/**
 * バッジは「形 3 種 × 意味 4 種」で組む。
 *
 * 形 (variant):
 *   solid   … BUY など「結論」を示すもの (1 画面に 1 種類まで)
 *   soft    … 状態の表示 (実行中・完了・失敗)
 *   outline … 分類の表示 (クラス・券種)。色を持たない
 *
 * 意味 (tone) は solid / soft にだけ効く。outline は常に無彩色。
 * tone="default" は --primary を引くので【測った値】を指すバッジになる
 * (OOS・実測・95% 区間)。学習時の値には付けない — docs/design.md「色の 3 層」。
 * 以前は default / secondary / destructive / outline / success / warning / info
 * ＋ soft-* 6 種で 13 通りあり、どれを使うかが場当たりになっていた。
 *
 * 角丸はピルの例外として rounded-full のまま (「基本は直角、意図があるときだけ
 * 完全な丸」というリズムを作る)。
 */
const badgeVariants = cva(
  'inline-flex items-center rounded-full border px-2 py-0.5 font-mono text-2xs tracking-[0.14em] transition-colors',
  {
    variants: {
      variant: {
        solid: 'border-transparent',
        soft: 'border-transparent',
        outline: 'border-border-strong text-muted-foreground',
      },
      tone: {
        default: '',
        success: '',
        destructive: '',
        warning: '',
      },
    },
    compoundVariants: [
      { variant: 'solid', tone: 'default', class: 'bg-primary text-primary-foreground' },
      { variant: 'solid', tone: 'success', class: 'bg-success text-success-foreground' },
      { variant: 'solid', tone: 'destructive', class: 'bg-destructive text-destructive-foreground' },
      { variant: 'solid', tone: 'warning', class: 'bg-warning text-warning-foreground' },
      { variant: 'soft', tone: 'default', class: 'bg-primary/15 text-primary' },
      { variant: 'soft', tone: 'success', class: 'bg-success/15 text-success' },
      { variant: 'soft', tone: 'destructive', class: 'bg-destructive/15 text-destructive' },
      { variant: 'soft', tone: 'warning', class: 'bg-warning/15 text-warning' },
    ],
    defaultVariants: {
      variant: 'soft',
      tone: 'default',
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, tone, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant, tone }), className)} {...props} />;
}

export { Badge, badgeVariants };
