import type { ComponentType } from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, CalendarDays, History, Brain, Download, Settings, Wallet } from 'lucide-react';
import { cn } from '@/lib/cn';

interface NavItem {
  to: string;
  label: string;
  icon: ComponentType<{ className?: string }>;
}

const navItems: NavItem[] = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/upcoming', label: 'Upcoming Races', icon: CalendarDays },
  { to: '/results', label: 'Recent Races', icon: History },
  { to: '/models', label: 'Models', icon: Brain },
  { to: '/ingest', label: 'Ingest', icon: Download },
  { to: '/ledger', label: 'Ledger', icon: Wallet },
  { to: '/settings', label: 'Settings', icon: Settings },
];

export function Sidebar() {
  return (
    <aside aria-label="サイドナビゲーション" className="flex h-full w-60 flex-col border-r bg-card">
      <div className="flex h-14 items-center gap-2.5 border-b px-5">
        <img src="/logo.svg" alt="" aria-hidden="true" className="h-6 w-6 shrink-0" />
        <span className="text-base font-bold tracking-wide text-primary">KEIBA AI</span>
      </div>
      <nav aria-label="主要画面" className="flex-1 space-y-0.5 p-2">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary/10 text-primary shadow-[inset_2px_0_0_0_hsl(var(--primary))]'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )
            }
          >
            <Icon className="h-4 w-4 shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
