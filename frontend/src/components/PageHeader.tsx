import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';

interface PageHeaderProps {
  icon: LucideIcon;
  title: string;
  /** Optional muted-foreground subtitle shown below the title */
  description?: string;
  /** Right-side slot for actions (buttons, dialogs) — laid out flex-row gap-2 */
  children?: ReactNode;
}

/**
 * Shared page header — used at the top of every route.
 *
 * Visual:
 *   [icon-tile]  Title (text-3xl)
 *                description (text-sm muted)              [actions →]
 *
 * The icon tile uses bg-primary/10 + text-primary for a subtle brand accent
 * that still picks up dark/light theme automatically.
 */
export function PageHeader({ icon: Icon, title, description, children }: PageHeaderProps) {
  return (
    <header className="flex items-start justify-between gap-4">
      <div className="flex items-center gap-3">
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
          <Icon className="h-5 w-5" strokeWidth={2} />
        </div>
        <div className="min-w-0">
          <h1 className="text-3xl font-bold leading-tight tracking-tight">{title}</h1>
          {description && (
            <p className="mt-0.5 text-sm text-muted-foreground">{description}</p>
          )}
        </div>
      </div>
      {children && <div className="flex shrink-0 items-center gap-2">{children}</div>}
    </header>
  );
}
