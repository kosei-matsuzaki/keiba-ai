import { createBrowserRouter } from 'react-router-dom';
import { App } from './App';
import { Dashboard } from './routes/Dashboard';
import { UpcomingRaces } from './routes/UpcomingRaces';
import { RaceDetail } from './routes/RaceDetail';
import { Models } from './routes/Models';
import { Ingest } from './routes/Ingest';
import { Settings } from './routes/Settings';
import { Ledger } from './routes/Ledger';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'upcoming', element: <UpcomingRaces /> },
      { path: 'races/:race_id', element: <RaceDetail /> },
      { path: 'models', element: <Models /> },
      { path: 'ingest', element: <Ingest /> },
      { path: 'settings', element: <Settings /> },
      { path: 'ledger', element: <Ledger /> },
    ],
  },
]);
