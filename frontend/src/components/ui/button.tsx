import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/cn';

const buttonVariants = cva(
  // base: 角丸 2px・等幅・字間広め。影は使わない (ダークで効かず、箱をやめる方針とも矛盾する)
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-sm ' +
    'font-mono text-[12px] tracking-[0.08em] ' +
    'transition-[background-color,border-color,color] duration-150 ' +
    'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary ' +
    'disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        // Primary CTA: アクセント 1 色の面
        default: 'bg-primary text-primary-foreground hover:bg-primary/90',
        // Destructive: red solid
        destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        // Outline (副ボタン): 枠だけ。面を持たせない
        outline:
          'border border-border-strong bg-transparent text-foreground hover:border-primary hover:text-primary',
        // Secondary: card-elevated bg, used for less-prominent actions
        secondary:
          'bg-card-elevated text-foreground hover:bg-card-elevated/70',
        // Ghost: no bg, just hover tint
        ghost:
          'text-muted-foreground hover:bg-card-elevated hover:text-foreground',
        // Soft primary: tinted bg + saturated fg (= dashboard pill 風 button)
        soft:
          'bg-primary/15 text-primary hover:bg-primary/25',
        // Link: underline on hover
        link:
          'text-primary underline-offset-4 hover:underline h-auto p-0',
      },
      size: {
        default: 'h-9 px-3.5 py-2',
        sm: 'h-8 px-2.5 text-xs',
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
