import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Recharts uses ResizeObserver which is not available in jsdom
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal('ResizeObserver', ResizeObserverStub);

// Radix UI (Select 等) が jsdom 未実装の pointer capture / scrollIntoView を
// 要求するためスタブする。これが無いと Select の open / item 選択が動かない。
window.HTMLElement.prototype.scrollIntoView = vi.fn();
window.HTMLElement.prototype.hasPointerCapture = vi.fn().mockReturnValue(false);
window.HTMLElement.prototype.setPointerCapture = vi.fn();
window.HTMLElement.prototype.releasePointerCapture = vi.fn();
