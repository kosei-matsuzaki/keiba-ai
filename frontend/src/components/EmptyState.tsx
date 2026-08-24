import { InboxIcon, type LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';

interface EmptyStateProps {
  message: string;
  description?: string;
  /** Override the default Inbox icon — pick a more situational lucide icon
   *  for empty model lists, no-results searches, network failures, etc. */
  icon?: LucideIcon;
  /** Optional action slot rendered below the description (e.g. a trigger button). */
  children?: ReactNode;
}

export function EmptyState({
  message,
  description,
  icon: Icon = InboxIcon,
  children,
}: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      {/* 40px。64px だと「何もない」ことを強調しすぎる。線幅は .lucide で統一。 */}
      <Icon className="h-10 w-10 text-muted-foreground/25" />
      <p className="text-base font-medium text-muted-foreground">{message}</p>
      {description && <p className="max-w-md text-sm text-muted-foreground/70">{description}</p>}
      {children && <div className="mt-2">{children}</div>}
    </div>
  );
}
