import { InboxIcon, type LucideIcon } from 'lucide-react';

interface EmptyStateProps {
  message: string;
  description?: string;
  /** Override the default Inbox icon — pick a more situational lucide icon
   *  for empty model lists, no-results searches, network failures, etc. */
  icon?: LucideIcon;
}

export function EmptyState({ message, description, icon: Icon = InboxIcon }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      <Icon className="h-16 w-16 text-muted-foreground/30" strokeWidth={1.5} />
      <p className="text-base font-medium text-muted-foreground">{message}</p>
      {description && <p className="max-w-md text-sm text-muted-foreground/70">{description}</p>}
    </div>
  );
}
