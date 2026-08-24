import { useEffect } from 'react';
import { Outlet } from 'react-router-dom';

import { Topbar } from '@/components/Topbar';
import { fetchHealth } from '@/lib/api';

/**
 * AppShell: Topbar (top) + scrollable main area (below).
 * The Outlet renders the current route's component.
 */
export function App() {
  // ky API client は lazy 初期化なので最初の fetch で TLS handshake が起きる。
  // App マウント直後に fetchHealth() を fire-and-forget しておくと、
  // 各ページのポーリング群が走る前に client を warm up できる。
  useEffect(() => {
    fetchHealth().catch(() => {
      // 起動直後に backend が ready でない可能性があるので無視。
      // 本来のページの useQuery が改めて retry する。
    });
  }, []);

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <Topbar />
      {/* 1920px 幅でカードが伸びきって視線移動が長くなるのを防ぐため、
          コンテンツ幅に上限を置く。横に広げたい表は、その表側で
          max-w-none / overflow-x-auto を指定する。 */}
      <main aria-label="メインコンテンツ" className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[1440px]">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
