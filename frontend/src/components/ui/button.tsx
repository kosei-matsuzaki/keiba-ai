import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/cn';

const buttonVariants = cva(
  // base: tighter (h-9 default), font-medium, gap-2 for icons
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium ' +
    'transition-[background-color,border-color,color,box-shadow,transform] duration-150 ' +
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background ' +
    'disabled:pointer-events-none disabled:opacity-50 active:scale-[0.98]',
  {
    variants: {
      variant: {
        // Primary CTA: indigo solid + subtle inner highlight + lift on hover
        default:
          'bg-primary text-primary-foreground shadow-[inset_0_1px_0_0_hsl(var(--primary-foreground)/0.15)] hover:bg-primary/90 hover:shadow-lg hover:shadow-primary/20',
        // Destructive: red solid
        destructive:
          'bg-destructive text-destructive-foreground hover:bg-destructive/90 hover:shadow-lg hover:shadow-destructive/20',
        // Outline: dark surface + visible border, hover bg subtle
        outline:
          'border border-border-strong bg-card text-foreground hover:bg-card-elevated hover:border-border-strong',
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
