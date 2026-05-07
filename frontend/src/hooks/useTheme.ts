import { useCallback, useEffect, useState } from 'react';

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'keiba-ai:theme';

function _readInitial(): Theme {
  if (typeof window === 'undefined') return 'dark';
  const saved = window.localStorage.getItem(STORAGE_KEY);
  if (saved === 'light' || saved === 'dark') return saved;
  // 初期値は dark (CamuQuotes-dark がプロジェクトの基本トーン)。
  return 'dark';
}

/**
 * テーマを localStorage に保存しつつ html.classList を切り替える hook。
 * Topbar の Sun/Moon トグルと連動する。
 *
 * shadcn 標準の class-based 戦略 (.dark を html に付ける) に従う。
 */
export function useTheme(): [Theme, (next: Theme) => void, () => void] {
  const [theme, setThemeState] = useState<Theme>(_readInitial);

  useEffect(() => {
    const html = document.documentElement;
    if (theme === 'dark') {
      html.classList.add('dark');
    } else {
      html.classList.remove('dark');
    }
    window.localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  const setTheme = useCallback((next: Theme) => setThemeState(next), []);
  const toggle = useCallback(
    () => setThemeState((prev) => (prev === 'dark' ? 'light' : 'dark')),
    [],
  );

  return [theme, setTheme, toggle];
}
