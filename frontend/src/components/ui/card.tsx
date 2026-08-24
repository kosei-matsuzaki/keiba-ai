import * as React from 'react';
import { cn } from '@/lib/cn';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  /**
   * 罫線と背景を持つ「箱」にする。既定は false。
   *
   * この UI は面ではなく **罫線と余白** で領域を作る方針なので、Card は既定で
   * ただの領域になる。囲うのはダイアログ・ポップオーバー・本当に浮かせたい
   * ものだけ。ダッシュボードの KPI・グラフ・表は boxed を付けず、呼び出し側が
   * `border-t` / `divide-y` で区切る。
   *
   * 囲いを外して散らかって見えるときは、箱を戻すのではなくセクション間の
   * 余白 (space-y-12 = 48px) を広げること。
   */
  boxed?: boolean;
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, boxed = false, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'text-card-foreground',
        boxed && 'rounded-sm border border-border bg-card',
        className
      )}
      {...props}
    />
  )
);
Card.displayName = 'Card';

const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    // p-6 は箱ありきの余白だった。区切りは罫線に任せ、上下はセクション間で作る。
    <div ref={ref} className={cn('flex flex-col space-y-1.5 pb-4', className)} {...props} />
  )
);
CardHeader.displayName = 'CardHeader';

const CardTitle = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn('text-base font-semibold leading-tight tracking-tight', className)}
      {...props}
    />
  )
);
CardTitle.displayName = 'CardTitle';

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p ref={ref} className={cn('text-sm text-muted-foreground', className)} {...props} />
));
CardDescription.displayName = 'CardDescription';

const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => <div ref={ref} className={cn(className)} {...props} />
);
CardContent.displayName = 'CardContent';

const CardFooter = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('flex items-center pt-4', className)} {...props} />
  )
);
CardFooter.displayName = 'CardFooter';

export { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter };
