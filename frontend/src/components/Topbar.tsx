import type { ComponentType } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  CalendarDays,
  Brain,
  Settings,
  Wallet,
} from 'lucide-react';

import { cn } from '@/lib/cn';

interface NavItem {
  to: string;
  label: string;
  icon: ComponentType<{ className?: string }>;
  /** RaceDetail (/races/:id) も Race tab を active にする */
  activeMatch?: (pathname: string) => boolean;
}

const navItems: NavItem[] = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  {
    to: '/races',
    label: 'Race',
    icon: CalendarDays,
    activeMatch: (p) => p.startsWith('/races') || p.startsWith('/upcoming') || p.startsWith('/past'),
  },
  { to: '/ledger', label: 'Ledger', icon: Wallet },
  {
    to: '/models',
    label: 'Models',
    icon: Brain,
    activeMatch: (p) => p.startsWith('/models'),
  },
  {
    to: '/settings',
    label: 'Settings',
    icon: Settings,
    activeMatch: (p) => p.startsWith('/settings') || p.startsWith('/ingest'),
  },
];

export function Topbar() {
  const { pathname } = useLocation();
  return (
    <header
      aria-label="トップナビゲーション"
      className="sticky top-0 z-30 flex h-14 items-center gap-6 border-b border-border bg-background/80 px-6 backdrop-blur-md"
    >
      {/* Logo */}
      <div className="flex items-center gap-2.5">
        <img src="/logo.svg" alt="" aria-hidden="true" className="h-6 w-6 shrink-0" />
        <span className="text-base font-bold tracking-wide text-foreground">
          KEIBA <span className="text-primary">AI</span>
        </span>
      </div>

      {/* Nav links */}
      <nav aria-label="主要画面" className="flex flex-1 items-center gap-1">
        {navItems.map(({ to, label, icon: Icon, activeMatch }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => {
              const active = activeMatch ? activeMatch(pathname) : isActive;
              return cn(
                'flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
                active
                  ? 'bg-primary/15 text-primary'
                  : 'text-muted-foreground hover:bg-card-elevated hover:text-foreground',
              );
            }}
          >
            <Icon className="h-4 w-4 shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
