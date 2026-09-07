import { createBrowserRouter, Navigate, redirect } from 'react-router-dom';
import { App } from './App';
import { Races } from './routes/Races';
import { RaceDetail } from './routes/RaceDetail';
import { Models } from './routes/Models';
import { ModelDetail } from './routes/ModelDetail';
import { Settings } from './routes/Settings';
import { StyleGuide } from './routes/StyleGuide';
import { Ledger } from './routes/Ledger';

/**
 * 旧 `/races*` を `/race*` へ送る。
 *
 * `?date=` を落とすと**選んでいた日が消える**ので search を持ち越す
 * (素の `<Navigate>` はクエリを捨てる)。loader にしてあるのは、描画前に返せば
 * 画面が一瞬出ないため。
 */
export function racesToRaceLoader({
  request,
  params,
}: {
  request: Request;
  params: { race_id?: string };
}) {
  const { search } = new URL(request.url);
  return redirect(params.race_id ? `/race/${params.race_id}${search}` : `/race${search}`);
}

export const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      // 最初に出るのはレース。この道具を開く理由は「その日のレースを見る」で、
      // モデルの成績を見に来るのは学習し直したときだけなので `/models` に置く。
      // `/` を画面にせず `/race` へ送るのは、**画面に呼び名を 2 つ作らないため**。
      { index: true, element: <Navigate to="/race" replace /> },
      { path: 'race', element: <Races /> },
      { path: 'race/:race_id', element: <RaceDetail /> },
      // 既存ブックマーク互換: 旧 /races* /upcoming /past /ingest は /race へ
      { path: 'races', loader: racesToRaceLoader },
      { path: 'races/:race_id', loader: racesToRaceLoader },
      { path: 'upcoming', element: <Navigate to="/race" replace /> },
      { path: 'past', element: <Navigate to="/race" replace /> },
      { path: 'ingest', element: <Navigate to="/race" replace /> },
      // モデルの成績・一覧・学習・役割の割り当ての 1 画面
      { path: 'models', element: <Models /> },
      { path: 'models/:model_id', element: <ModelDetail /> },
      // Settings: 全レース共通の予想パラメータ + スクレイパー動作設定
      { path: 'settings', element: <Settings /> },
      { path: 'ledger', element: <Ledger /> },
      // 見た目の規定を実物で確かめる 1 画面。Topbar には出さない
      // (毎日使う画面ではない)。docs/ui-style.md から指す。
      { path: 'style', element: <StyleGuide /> },
    ],
  },
]);
