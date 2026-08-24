import { cn } from '@/lib/cn';

/**
 * ブランドマーク (馬蹄)。Topbar のロゴ。
 *
 * `<img src="/logo.svg" />` ではなく inline SVG にしてあるのは、色を `currentColor` に
 * 預けてテーマ (dark / light) に追従させるため。`<img>` 経由だと CSS 変数も
 * currentColor も解決されず、HSL 直書きで固定するしかない。
 *
 * viewBox は 32x32 のタイル版 (`public/favicon.svg`) から馬蹄だけを正方形に切り出した
 * もので、パスの座標は favicon と同一。地 (タイル) は持たない — ヘッダに app タイルを
 * 置かないのは globals.css の「箱をやめて罫線で区切る」に従うため。
 */
export function BrandMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="8 10.6 16 16"
      fill="none"
      aria-hidden="true"
      focusable="false"
      className={cn('shrink-0', className)}
    >
      <path d="M10 21V14a6 6 0 0 1 12 0v7" stroke="currentColor" strokeWidth="2.4" />
      <rect x="8.8" y="22.1" width="2.4" height="2.4" fill="currentColor" />
      <rect x="20.8" y="22.1" width="2.4" height="2.4" fill="currentColor" />
    </svg>
  );
}
