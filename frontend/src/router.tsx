import { createBrowserRouter, Navigate } from 'react-router-dom';
import { App } from './App';
import { Dashboard } from './routes/Dashboard';
import { Races } from './routes/Races';
import { RaceDetail } from './routes/RaceDetail';
import { ModelDetail } from './routes/ModelDetail';
import { Settings } from './routes/Settings';
import { Ledger } from './routes/Ledger';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Dashboard /> },
      // Race: カレンダーで日を選ぶ 1 画面 (今週末/Past のタブは廃止)
      { path: 'races', element: <Races /> },
      { path: 'races/:race_id', element: <RaceDetail /> },
      // 既存ブックマーク互換: 旧 /upcoming /past は /races へ redirect
      { path: 'upcoming', element: <Navigate to="/races" replace /> },
      { path: 'past', element: <Navigate to="/races" replace /> },
      // モデルの一覧・学習・役割の割り当ては Dashboard に統合済み。
      // 旧ブックマーク互換で /models は Dashboard へ送る。
      { path: 'models', element: <Navigate to="/" replace /> },
      { path: 'models/:model_id', element: <ModelDetail /> },
      // Settings: 全レース共通の予想パラメータ + スクレイパー動作設定
      { path: 'settings', element: <Settings /> },
      // 取込は Race > 過去のレース (カレンダー横) へ移設したので、旧 /ingest はそこへ
      { path: 'ingest', element: <Navigate to="/races?tab=past" replace /> },
      { path: 'ledger', element: <Ledger /> },
    ],
  },
]);
