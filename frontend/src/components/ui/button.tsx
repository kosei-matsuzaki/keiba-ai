import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/cn';

const buttonVariants = cva(
  // base: 角丸 2px・等幅・字間広め。影は使わない (ダークで効かず、箱をやめる方針とも矛盾する)
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-sm ' +
    'font-mono text-xs tracking-[0.08em] ' +
    'transition-[background-color,border-color,color] duration-150 ' +
    'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary ' +
    'disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        // Primary CTA: **無彩色の反転** (地と文字を入れ替える)。
        // アクセント (--primary) は「測れているか」を指す色なので、CTA に使うと
        // 意味が薄まり、面で出た瞬間に画面でいちばん強いものが「実行」になる
        // (煽る見た目に寄る)。強さは色ではなくコントラストで出す。
        default: 'bg-foreground text-background hover:bg-foreground/85',
        // Destructive: red solid
        destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        // Outline (副ボタン): 枠だけ。面を持たせない
        outline:
          'border border-border-strong bg-transparent text-foreground hover:border-foreground',
        // Secondary: card-elevated bg, used for less-prominent actions
        secondary:
          'bg-card-elevated text-foreground hover:bg-card-elevated/70',
        // Ghost: no bg, just hover tint
        ghost:
          'text-muted-foreground hover:bg-card-elevated hover:text-foreground',
        // Soft: 淡い面 + 前景色。ここもアクセントを使わない
        soft:
          'bg-card-elevated text-foreground hover:bg-card-elevated/70',
        // Link: underline on hover
        link:
          'text-primary underline-offset-4 hover:underline h-auto p-0',
      },
      size: {
        default: 'h-9 px-4 py-2',
        sm: 'h-8 px-3 text-xs',
        lg: 'h-10 px-5',
        icon: 'h-9 w-9 p-0',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button';
    return (
      <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
    );
  }
);
Button.displayName = 'Button';

export { Button, buttonVariants };
