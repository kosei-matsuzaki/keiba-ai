import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Recharts uses ResizeObserver which is not available in jsdom
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal('ResizeObserver', ResizeObserverStub);
