import { createBrowserRouter, Navigate } from 'react-router-dom';
import { App } from './App';
import { Dashboard } from './routes/Dashboard';
import { Races } from './routes/Races';
import { RaceDetail } from './routes/RaceDetail';
import { Models } from './routes/Models';
import { ModelDetail } from './routes/ModelDetail';
import { Settings } from './routes/Settings';
import { Ledger } from './routes/Ledger';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Dashboard /> },
      // Race tab: Upcoming + Past を 1 ページに統合
      { path: 'races', element: <Races /> },
      { path: 'races/:race_id', element: <RaceDetail /> },
      // 既存ブックマーク互換: 旧 /upcoming /past は /races へ redirect
      { path: 'upcoming', element: <Navigate to="/races" replace /> },
      { path: 'past', element: <Navigate to="/races" replace /> },
      // Models: 一覧。各モデルの詳細 (/models/:id) でバックテストを実行する。
      { path: 'models', element: <Models /> },
      { path: 'models/:model_id', element: <ModelDetail /> },
      // Settings: 一般 + Ingest を内部タブで統合
      { path: 'settings', element: <Settings /> },
      // 旧 /ingest は /settings (Ingest タブ) へ redirect
      { path: 'ingest', element: <Navigate to="/settings" replace /> },
      { path: 'ledger', element: <Ledger /> },
    ],
  },
]);
