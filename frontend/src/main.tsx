import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider } from 'react-router-dom';
import { QueryClientProvider } from '@tanstack/react-query';
import { router } from './router';
import { queryClient } from './lib/query-client';
import { Toaster } from '@/components/ui/toaster';
import './globals.css';

// First-paint テーマ適用 (FOUC 回避): React レンダリング前に localStorage から
// 復元して html.classList に反映する。useTheme hook は以後の切替を担当。
(() => {
  try {
    const saved = window.localStorage.getItem('keiba-ai:theme');
    const theme = saved === 'light' ? 'light' : 'dark';
    if (theme === 'dark') document.documentElement.classList.add('dark');
  } catch {
    document.documentElement.classList.add('dark');
  }
})();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      <Toaster />
    </QueryClientProvider>
  </React.StrictMode>
);
