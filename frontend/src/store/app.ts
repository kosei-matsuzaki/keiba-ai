import { create } from 'zustand';

interface AppStore {
  /** FastAPI port — overridden by Tauri invoke in M8 */
  apiPort: number;
  setApiPort: (port: number) => void;

  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
}

export const useAppStore = create<AppStore>((set) => ({
  apiPort: 8765,
  setApiPort: (port) => set({ apiPort: port }),

  sidebarOpen: true,
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
}));

interface ScraperStore {
  /** True while a manual scrape job is running — controls polling interval */
  isRunning: boolean;
  setRunning: (running: boolean) => void;
}

export const useScraperStore = create<ScraperStore>((set) => ({
  isRunning: false,
  setRunning: (running) => set({ isRunning: running }),
}));
