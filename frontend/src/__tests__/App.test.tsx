import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { QueryClientProvider, QueryClient } from '@tanstack/react-query';
import { App } from '../App';
import { Dashboard } from '../routes/Dashboard';
import { UpcomingRaces } from '../routes/UpcomingRaces';
import { RaceDetail } from '../routes/RaceDetail';
import { Models } from '../routes/Models';
import { Ingest } from '../routes/Ingest';
import { Settings } from '../routes/Settings';

// Suppress console.error from React Query error boundaries during tests
vi.spyOn(console, 'error').mockImplementation(() => {});

function makeRouter(initialPath: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  const router = createMemoryRouter(
    [
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
        ],
      },
    ],
    { initialEntries: [initialPath] }
  );

  return { router, client };
}

function renderAt(path: string) {
  const { router, client } = makeRouter(path);
  return render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}

// Mock fetch so React Query queries don't throw unhandled errors
beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500, json: async () => ({}) }));
});

describe('Routing', () => {
  it('renders Dashboard at /', async () => {
    renderAt('/');
    expect(await screen.findByRole('heading', { name: 'Dashboard' })).toBeInTheDocument();
  });

  it('renders UpcomingRaces at /upcoming', async () => {
    renderAt('/upcoming');
    expect(await screen.findByRole('heading', { name: 'Upcoming Races' })).toBeInTheDocument();
  });

  it('renders RaceDetail placeholder at /races/:id', async () => {
    renderAt('/races/202406010101');
    expect(await screen.findByRole('heading', { name: 'Race Detail' })).toBeInTheDocument();
  });

  it('renders Models placeholder at /models', async () => {
    renderAt('/models');
    expect(await screen.findByRole('heading', { name: 'Models' })).toBeInTheDocument();
  });

  it('renders Ingest placeholder at /ingest', async () => {
    renderAt('/ingest');
    expect(await screen.findByRole('heading', { name: 'Ingest' })).toBeInTheDocument();
  });

  it('renders Settings placeholder at /settings', async () => {
    renderAt('/settings');
    expect(await screen.findByRole('heading', { name: 'Settings' })).toBeInTheDocument();
  });

  it('sidebar contains all navigation links', async () => {
    renderAt('/');
    expect(await screen.findByRole('link', { name: /Dashboard/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Upcoming Races/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Models/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Ingest/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Settings/i })).toBeInTheDocument();
  });
});
