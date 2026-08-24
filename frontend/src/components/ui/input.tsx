import * as React from 'react';
import { cn } from '@/lib/cn';

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          // 黒地に対して入力欄がわずかに明るいほうが「入力できる場所」が分かる。
          // フォーカスリングは太いと角丸 2px の直角さを殺すので 1px + 枠線の着色。
          'flex h-9 w-full rounded-sm border border-input bg-card px-3 py-2 text-sm',
          'file:border-0 file:bg-transparent file:text-sm file:font-medium',
          'placeholder:text-subtle-foreground',
          'focus-visible:outline-none focus-visible:border-primary focus-visible:ring-1 focus-visible:ring-primary',
          'disabled:cursor-not-allowed disabled:opacity-50',
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Input.displayName = 'Input';

export { Input };
