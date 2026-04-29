import { InboxIcon } from 'lucide-react';

interface EmptyStateProps {
  message: string;
  description?: string;
}

export function EmptyState({ message, description }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      <InboxIcon className="h-12 w-12 text-muted-foreground/40" />
      <p className="text-base font-medium text-muted-foreground">{message}</p>
      {description && <p className="text-sm text-muted-foreground/70">{description}</p>}
    </div>
  );
}
