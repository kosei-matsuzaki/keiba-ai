import type { HTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

function Skeleton({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('animate-skeleton-shimmer rounded-md bg-muted', className)}
      {...props}
    />
  );
}

export { Skeleton };
