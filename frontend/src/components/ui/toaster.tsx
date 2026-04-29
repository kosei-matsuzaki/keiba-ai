import { Toaster as SonnerToaster } from 'sonner';

/**
 * Mount once in the app root (main.tsx or App.tsx) to enable toast notifications.
 */
export function Toaster() {
  return (
    <SonnerToaster
      position="top-right"
      richColors
      closeButton
    />
  );
}
