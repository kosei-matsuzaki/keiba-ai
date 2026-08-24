import * as React from 'react';
import * as TabsPrimitive from '@radix-ui/react-tabs';
import { cn } from '@/lib/cn';

const Tabs = TabsPrimitive.Root;

/**
 * 見出し風のタブ。
 *
 * 「ボタンが並んでいる」のではなく「見出しが並んでいる」見た目にする:
 * - 幅は文字数に任せる (旧実装の min-w-32 = 128px 固定を撤去。2 文字のタブまで
 *   128px 占有し、5 タブで 640px をラベルの並びに費やしていた)
 * - 区切りはタブ同士の間隔 (gap-6) で作る
 * - ラベルは等幅・字間広め。Topbar・表ヘッダ・KPI ラベルと揃い、
 *   「分類ラベルは等幅」という規則が全画面で通る
 * - active の指示子は角丸をやめ、タブ幅いっぱいの 1px 直線
 *
 * 高さは固定せず中身に任せる (旧 h-10)。下端の罫線は TabsList が持ち、
 * 指示子はその罫線の上にちょうど重なる。
 */
const TabsList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn(
      'inline-flex items-stretch gap-6 border-b border-border text-muted-foreground',
      className
    )}
    {...props}
  />
));
TabsList.displayName = TabsPrimitive.List.displayName;

const TabsTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      'relative inline-flex items-center justify-center whitespace-nowrap',
      'px-0.5 pb-3 pt-2 font-mono text-[12px] tracking-[0.1em] transition-colors',
      'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary',
      'disabled:pointer-events-none disabled:opacity-50',
      'text-subtle-foreground hover:text-foreground',
      'data-[state=active]:text-primary',
      // 指示子: TabsList の罫線にちょうど重なる 1px の直線
      'data-[state=active]:after:absolute data-[state=active]:after:inset-x-0',
      'data-[state=active]:after:bottom-[-1px] data-[state=active]:after:h-px',
      'data-[state=active]:after:bg-primary',
      className
    )}
    {...props}
  />
));
TabsTrigger.displayName = TabsPrimitive.Trigger.displayName;

const TabsContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    ref={ref}
    className={cn(
      // 注: 上 margin はデフォルト 0。親 (Tabs) の flex gap で TabsList との
      // 間隔を制御し、タブごとに余分な余白が入らないようにする。
      'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary',
      className
    )}
    {...props}
  />
));
TabsContent.displayName = TabsPrimitive.Content.displayName;

export { Tabs, TabsList, TabsTrigger, TabsContent };
