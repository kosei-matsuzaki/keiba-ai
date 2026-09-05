import type { ReactNode } from 'react';

interface PageHeaderProps {
  /**
   * 見出しの左に置く印。数字を持つのは**レースだけ**なので、レース系の画面が
   * `11R` を渡す。Dashboard / Ledger / Models / Settings は渡さない
   * (章番号は読み物の語彙で、この題材のものではない)。
   */
  marker?: ReactNode;
  /** 等幅の小さな英字ラベル。日本語は title 側に置く。 */
  eyebrow: string;
  title: string;
  /** Optional muted-foreground subtitle shown below the title */
  description?: string;
  /** Right-side slot for actions (buttons, dialogs) — laid out flex-row gap-2 */
  children?: ReactNode;
}

/**
 * 全画面共通のページ見出し。
 *
 * 見出しの右へ罫線を伸ばす「競馬新聞の罫」で区切る (方眼紙はやめた)。
 *
 *   RACE DETAIL
 *   11R  日本ダービー ───────────────────────   [actions →]
 *        2024-06-01・芝2400m
 */
export function PageHeader({
  marker,
  eyebrow,
  title,
  description,
  children,
}: PageHeaderProps) {
  return (
    <header className="flex items-start justify-between gap-4">
      <div className="min-w-0 flex-1">
        <span className="font-mono text-2xs tracking-[0.04em] text-subtle-foreground">
          {eyebrow}
        </span>
        <div className="mt-1 flex items-center gap-3">
          {marker}
          <h1 className="truncate text-lg font-semibold tracking-tight">{title}</h1>
          <span className="h-px flex-1 bg-border" aria-hidden="true" />
        </div>
        {description && (
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{description}</p>
        )}
      </div>
      {children && <div className="flex shrink-0 items-center gap-2 pt-5">{children}</div>}
    </header>
  );
}
