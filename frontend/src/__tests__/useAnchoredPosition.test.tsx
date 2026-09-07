import { describe, it, expect, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { useRef } from 'react';

import { useAnchoredPosition } from '../hooks/useAnchoredPosition';

// jsdom は寸法を持たないので、基準要素の矩形と浮遊物の高さだけを差し込む。
// 実測できないと「下に入るか」を判定しようがないため、ここだけは stub する。
const realRect = Element.prototype.getBoundingClientRect;
const realHeight = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'offsetHeight');

function stubLayout({ top, bottom, height }: { top: number; bottom: number; height: number }) {
  Element.prototype.getBoundingClientRect = () =>
    ({ top, bottom, left: 40, right: 140, width: 100, height: bottom - top }) as DOMRect;
  Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
    configurable: true,
    get: () => height,
  });
}

afterEach(() => {
  Element.prototype.getBoundingClientRect = realRect;
  if (realHeight) Object.defineProperty(HTMLElement.prototype, 'offsetHeight', realHeight);
});

function Harness() {
  const anchor = useRef<HTMLDivElement>(null);
  const floating = useRef<HTMLDivElement>(null);
  const pos = useAnchoredPosition(anchor, floating, true, { gap: 6 });
  return (
    <>
      <div ref={anchor} />
      <div ref={floating} />
      <output data-testid="pos">{`${pos.placement} top=${pos.top} ready=${pos.ready}`}</output>
    </>
  );
}

describe('useAnchoredPosition', () => {
  it('下に入るならそのまま下に出す', () => {
    // window.innerHeight は jsdom 既定の 768。
    stubLayout({ top: 100, bottom: 120, height: 200 });
    render(<Harness />);
    expect(screen.getByTestId('pos')).toHaveTextContent('bottom top=126 ready=true');
  });

  it('下に入らず上に入るなら上に出す (画面下の行で見切れていた)', () => {
    stubLayout({ top: 680, bottom: 700, height: 200 });
    render(<Harness />);
    // 680 - gap 6 - 高さ 200
    expect(screen.getByTestId('pos')).toHaveTextContent('top top=474 ready=true');
  });

  it('上下どちらにも入らないときは、下辺で切らずに収まる位置まで押し上げる', () => {
    // 高さ 700 は 768 の画面にほぼ収まらない。上に反転しても入らないので、
    // 下向きのまま viewport の内側へ寄せる (上端が隠れるほうが読める)。
    stubLayout({ top: 300, bottom: 320, height: 700 });
    render(<Harness />);
    expect(screen.getByTestId('pos')).toHaveTextContent('bottom top=60 ready=true');
  });
});
