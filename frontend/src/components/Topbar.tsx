import { NavLink, useLocation } from 'react-router-dom';
import { Sun, Moon } from 'lucide-react';

import { BrandMark } from '@/components/BrandMark';
import { Button } from '@/components/ui/button';
import { useTheme } from '@/hooks/useTheme';
import { cn } from '@/lib/cn';

interface NavItem {
  to: string;
  label: string;
  /** RaceDetail (/races/:id) も Race tab を active にする */
  activeMatch?: (pathname: string) => boolean;
}

// 並びは週次の流れ順 (レースを見る → 収支を突き合わせる → モデルを測り直す → 設定)。
// 「DASHBOARD」は置かない — 最初に出る画面が RACE そのものなので、
// 別名を与えると同じ画面に 2 つの呼び名ができる。
const navItems: NavItem[] = [
  {
    to: '/race',
    label: 'RACE',
    // '/races' も startsWith('/race') に入る (旧ブックマークの一瞬の滞在)。
    activeMatch: (p) =>
      p === '/' || p.startsWith('/race') || p.startsWith('/upcoming') || p.startsWith('/past'),
  },
  { to: '/ledger', label: 'LEDGER' },
  // モデル詳細 (/models/:id) からもここが active に見えるようにする。
  { to: '/models', label: 'MODEL', activeMatch: (p) => p.startsWith('/models') },
  {
    to: '/settings',
    label: 'SETTINGS',
    activeMatch: (p) => p.startsWith('/settings') || p.startsWith('/ingest'),
  },
];

/**
 * ナビは等幅の英字だけにする (アイコンも番号も外す)。
 * 選択中は面ではなく色 (primary) で示す。
 */
export function Topbar() {
  const { pathname } = useLocation();
  const [theme, , toggleTheme] = useTheme();
  return (
    <header
      aria-label="トップナビゲーション"
      className="sticky top-0 z-30 flex h-16 items-center gap-8 border-b border-border bg-background/80 px-8 backdrop-blur-md"
    >
      {/* Logo */}
      <div className="flex items-center gap-2">
        <BrandMark className="h-[18px] w-[18px] text-primary" />
        <span className="font-mono text-xs tracking-[0.2em] text-foreground">
          KEIBA <span className="text-primary">AI</span>
        </span>
      </div>

      {/* Nav links */}
      <nav aria-label="主要画面" className="flex flex-1 items-center gap-6">
        {navItems.map(({ to, label, activeMatch }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => {
              const active = activeMatch ? activeMatch(pathname) : isActive;
              return cn(
                'font-mono text-xs tracking-[0.12em] transition-colors',
                active ? 'text-primary' : 'text-subtle-foreground hover:text-foreground'
              );
            }}
          >
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Theme toggle */}
      <Button
        variant="ghost"
        size="icon"
        onClick={toggleTheme}
        aria-label={theme === 'dark' ? 'ライトモードに切替' : 'ダークモードに切替'}
        title={theme === 'dark' ? 'ライトモードに切替' : 'ダークモードに切替'}
      >
        {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
      </Button>
    </header>
  );
}
