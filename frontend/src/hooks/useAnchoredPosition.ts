import { useLayoutEffect, useState, type RefObject } from 'react';

export type AnchorAlign = 'start' | 'end';

export interface AnchoredPosition {
  top: number;
  left: number;
  /** 下に入らなければ上に出す。呼び出し側が矢印などを反転させたいとき用。 */
  placement: 'bottom' | 'top';
  /** 座標が決まったか。決まるまでは隠したまま描いて寸法を測る。 */
  ready: boolean;
}

const HIDDEN: AnchoredPosition = { top: 0, left: 0, placement: 'bottom', ready: false };

/**
 * 開いている間だけ、基準要素に貼り付く浮遊物の座標 (viewport 座標) を返す。
 *
 * **`position: fixed` + portal で出すため。**素の `absolute` だと、
 * `MetricCard` (`overflow-hidden`) や表の横スクロール枠の中で**切れる**。
 *
 * **入らなければ上に出す。**画面下寄りの行で「⋯」を開くと、下に描いたメニューが
 * 画面外にはみ出して見切れていた。横も同じで、viewport の内側に収まるよう左右を詰める。
 *
 * 寸法は描いてからでないと測れないので 2 段構え — `ready:false` の間は
 * 呼び出し側が `visibility: hidden` で描き、`ResizeObserver` が実寸を返した時点で
 * 位置を決めて見せる。スクロールとリサイズでも測り直す (`capture: true` は
 * 内側のスクロール枠で起きた scroll も拾うため)。
 */
export function useAnchoredPosition(
  anchorRef: RefObject<HTMLElement>,
  floatingRef: RefObject<HTMLElement>,
  open: boolean,
  { align = 'start', gap = 6, margin = 8 }: { align?: AnchorAlign; gap?: number; margin?: number } = {}
): AnchoredPosition {
  const [pos, setPos] = useState<AnchoredPosition>(HIDDEN);

  useLayoutEffect(() => {
    if (!open) {
      setPos(HIDDEN);
      return;
    }
    function update() {
      const anchor = anchorRef.current;
      if (!anchor) return;
      const a = anchor.getBoundingClientRect();
      const floating = floatingRef.current;
      const h = floating?.offsetHeight ?? 0;
      const w = floating?.offsetWidth ?? 0;

      // 下に入らず、上には入るときだけ反転する。どちらも入らないなら下のまま
      // 収まる位置まで押し上げる (下辺で切るより、上端が隠れるほうが読める)。
      const below = window.innerHeight - a.bottom - gap - margin;
      const above = a.top - gap - margin;
      const flip = h > 0 && below < h && above >= h;
      let top = flip ? a.top - gap - h : a.bottom + gap;
      if (h > 0) top = Math.min(top, Math.max(margin, window.innerHeight - h - margin));
      top = Math.max(margin, top);

      let left = align === 'end' ? a.right - w : a.left;
      if (w > 0) left = Math.min(left, Math.max(margin, window.innerWidth - w - margin));
      left = Math.max(margin, left);

      setPos({ top, left, placement: flip ? 'top' : 'bottom', ready: true });
    }

    update();
    // observe した瞬間にも 1 回発火するので、これが実寸での測り直しになる。
    const ro =
      typeof ResizeObserver !== 'undefined' && floatingRef.current
        ? new ResizeObserver(update)
        : null;
    if (ro && floatingRef.current) ro.observe(floatingRef.current);
    window.addEventListener('scroll', update, true);
    window.addEventListener('resize', update);
    return () => {
      ro?.disconnect();
      window.removeEventListener('scroll', update, true);
      window.removeEventListener('resize', update);
    };
  }, [anchorRef, floatingRef, open, align, gap, margin]);

  return pos;
}
