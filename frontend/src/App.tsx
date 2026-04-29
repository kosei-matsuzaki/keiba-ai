import { Outlet } from 'react-router-dom';
import { Sidebar } from '@/components/Sidebar';

/**
 * AppShell: Sidebar (left) + scrollable main area (right).
 * The Outlet renders the current route's component.
 */
export function App() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
