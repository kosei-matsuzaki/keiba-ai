import * as React from 'react';
import { cn } from '@/lib/cn';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  /**
   * 罫線と背景を持つ「箱」にする。既定は false。
   *
   * この UI は面ではなく **罫線と余白** で領域を作るのが基本。ただし
   * **1 画面に別種の情報が 5〜6 個並ぶところ (Dashboard / シミュレーション) は
   * 画面直下の塊にだけ箱を付ける**。罫線と余白だけだと、どこまでが 1 つの話か
   * 読み取れなかったため。
   *
   * 箱は入れ子にしない。箱の中は従来どおり罫線と余白で仕切る (箱の中に箱が
   * あると、外側の意味が消える)。表の 1 行・KPI の 1 項目のような小さい単位にも
   * 付けない。
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
