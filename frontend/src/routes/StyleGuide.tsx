import { useLayoutEffect, useRef, useState } from 'react';
import { AlertTriangle, SearchX } from 'lucide-react';

import { EmptyState } from '@/components/EmptyState';
import { InfoTip } from '@/components/InfoTip';
import { MetricCard } from '@/components/MetricCard';
import { PageHeader } from '@/components/PageHeader';
import { RowActionsMenu } from '@/components/RowActionsMenu';
import { SectionHeading } from '@/components/SectionHeading';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { MetricTone } from '@/lib/metricStyles';

/**
 * `/style` — 規定と実装が合っているかを目で確かめる 1 画面。
 *
 * **値を書き写さない。**ここに出る数字はすべて、実際に描いた要素を
 * `getComputedStyle` で測ったもの。だから `docs/ui-style.md` のように
 * 古くなることがない — 古くなったらそれは実装が変わったということ。
 *
 * これを作ったのは、**画面を見ないと気づけない壊れ方を 3 回続けて踏んだ**から
 * (2026-09-06〜07):
 *   - `.text-kpi` が `cn()` の tailwind-merge でクラスごと消えていた
 *   - `.text-label-ja` がレイヤ順で `text-lg` に負けていた
 *   - `font-semibold` (600) を使っているのに 600 を読み込んでいない
 * どれもエラーが出ず、テストも通る。並べて見れば一目で分かるものだった。
 *
 * Topbar には出さない (毎日使う画面ではない)。`docs/ui-style.md` から指す。
 */

// ── 測る ──────────────────────────────────────────────────────────────────

/** rgb(74, 189, 207) → #4ABDCF。ブラウザが返すのは rgb 形式なので変換する。 */
function toHex(rgb: string): string {
  const m = rgb.match(/\d+/g);
  if (!m || m.length < 3) return rgb;
  return (
    '#' +
    m
      .slice(0, 3)
      .map((n) => Number(n).toString(16).padStart(2, '0'))
      .join('')
      .toUpperCase()
  );
}

interface TokenRow {
  name: string;
  hsl: string;
  hex: string;
}

/** :root に載っている CSS 変数の値と、実際に描画される色を読む。 */
function useTokens(names: string[]): TokenRow[] {
  const probeRef = useRef<HTMLDivElement>(null);
  const [rows, setRows] = useState<TokenRow[]>([]);

  useLayoutEffect(() => {
    const root = getComputedStyle(document.documentElement);
    const probe = probeRef.current;
    if (!probe) return;
    setRows(
      names.map((name) => {
        probe.style.backgroundColor = `hsl(var(${name}))`;
        return {
          name,
          hsl: root.getPropertyValue(name).trim(),
          hex: toHex(getComputedStyle(probe).backgroundColor),
        };
      })
    );
    probe.style.backgroundColor = '';
    // テーマを切り替えたら測り直したいので、html の class を監視する。
    const obs = new MutationObserver(() => {
      const r = getComputedStyle(document.documentElement);
      setRows(
        names.map((name) => {
          probe.style.backgroundColor = `hsl(var(${name}))`;
          return {
            name,
            hsl: r.getPropertyValue(name).trim(),
            hex: toHex(getComputedStyle(probe).backgroundColor),
          };
        })
      );
      probe.style.backgroundColor = '';
    });
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => obs.disconnect();
  }, [names]);

  return rows;
}

// ── 色 ────────────────────────────────────────────────────────────────────

const LAYERS: { layer: string; note: string; tokens: string[] }[] = [
  {
    layer: 'chrome',
    note: '地・面・罫・文字。**--primary と --warning の 2 色だけ**で、ここを増やさない',
    tokens: [
      '--background',
      '--card',
      '--card-elevated',
      '--foreground',
      '--muted-foreground',
      '--subtle-foreground',
      '--border',
      '--border-strong',
      '--primary',
    ],
  },
  {
    layer: '値の向き',
    note: '損益・的中・成否。面積が小さくても常に色を持たせる',
    tokens: ['--success', '--destructive', '--warning'],
  },
  {
    layer: 'データ',
    note: '暦の慣習。--info は土曜日だけ (日曜は --destructive)',
    tokens: ['--info'],
  },
];

function Colors() {
  const probeRef = useRef<HTMLDivElement>(null);
  const all = LAYERS.flatMap((l) => l.tokens);
  const rows = useTokens(all);
  const byName = new Map(rows.map((r) => [r.name, r]));

  return (
    <section className="flex flex-col gap-6">
      {/* 色を実際に描いて読み返すための、見えない probe。 */}
      <div ref={probeRef} aria-hidden="true" className="pointer-events-none fixed h-0 w-0" />
      {LAYERS.map(({ layer, note, tokens }) => (
        <div key={layer} className="flex flex-col gap-3">
          <SectionHeading level={3}>{layer}</SectionHeading>
          <p className="text-2xs text-muted-foreground">{note}</p>
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {tokens.map((name) => {
              const t = byName.get(name);
              return (
                <div key={name} className="flex items-center gap-3">
                  <span
                    className="h-8 w-8 shrink-0 rounded-sm border border-border"
                    style={{ backgroundColor: `hsl(var(${name}))` }}
                  />
                  <div className="flex min-w-0 flex-col">
                    <dt className="truncate font-mono text-2xs text-foreground">{name}</dt>
                    <dd className="text-num font-mono text-2xs text-subtle-foreground">
                      {t ? `${t.hex} · ${t.hsl}` : '測定中'}
                    </dd>
                  </div>
                </div>
              );
            })}
          </dl>
        </div>
      ))}
    </section>
  );
}

// ── 字 ────────────────────────────────────────────────────────────────────

const STEPS = [
  { cls: 'text-2xs', use: 'ラベル・単位・バッジ・補足' },
  { cls: 'text-xs', use: '表のセル、小さい本文' },
  { cls: 'text-sm', use: '本文 (body の既定)' },
  { cls: 'text-base', use: '節の見出し (h2)' },
  { cls: 'text-lg', use: 'ページ見出し (h1)' },
  { cls: 'text-kpi', use: '数値。等幅 + tabular-nums' },
];

function TypeScale() {
  const refs = useRef<(HTMLSpanElement | null)[]>([]);
  const [sizes, setSizes] = useState<string[]>([]);

  useLayoutEffect(() => {
    setSizes(
      STEPS.map((_, i) => {
        const el = refs.current[i];
        if (!el) return '';
        const c = getComputedStyle(el);
        return `${parseFloat(c.fontSize)}px / ${c.fontWeight}`;
      })
    );
  }, []);

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-28">クラス</TableHead>
          <TableHead className="w-32">実測</TableHead>
          <TableHead>見え方</TableHead>
          <TableHead className="w-56">用途</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {STEPS.map(({ cls, use }, i) => (
          <TableRow key={cls}>
            <TableCell className="font-mono text-2xs">{cls}</TableCell>
            <TableCell className="cell-num text-2xs">{sizes[i] ?? '—'}</TableCell>
            <TableCell>
              <span ref={(el) => (refs.current[i] = el)} className={cls}>
                回収率 0.907
              </span>
            </TableCell>
            <TableCell className="text-2xs text-muted-foreground">{use}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

/** 読み込めているウェイトを FontFaceSet から直に読む。 */
function LoadedFonts() {
  const [faces, setFaces] = useState<{ family: string; weights: string[] }[]>([]);
  const [missing, setMissing] = useState<string[]>([]);

  useLayoutEffect(() => {
    void document.fonts.ready.then(() => {
      const map = new Map<string, Set<string>>();
      document.fonts.forEach((f) => {
        const fam = f.family.replace(/['"]/g, '');
        if (!map.has(fam)) map.set(fam, new Set());
        map.get(fam)!.add(String(f.weight));
      });
      setFaces(
        [...map.entries()]
          .map(([family, w]) => ({ family, weights: [...w].sort() }))
          .sort((a, b) => a.family.localeCompare(b.family))
      );
      // 画面で使っているウェイトのうち、読み込めていないもの
      const used = ['400', '500', '600', '700'];
      setMissing(
        used.filter((w) => !document.fonts.check(`${w} 16px Inter`))
      );
    });
  }, []);

  return (
    <div className="flex flex-col gap-3">
      <dl className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {faces.map(({ family, weights }) => (
          <div key={family} className="flex min-w-0 flex-col gap-1">
            <dt className="text-label-ja">{family}</dt>
            <dd className="text-num font-mono text-xs text-foreground">{weights.join(' / ')}</dd>
          </div>
        ))}
      </dl>
      {missing.length > 0 && (
        <div className="flex items-start gap-3 border border-warning/30 bg-warning/[0.06] px-4 py-3">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" aria-hidden="true" />
          <p className="text-2xs leading-relaxed">
            <span className="font-medium">Inter の {missing.join(' / ')} を読み込んでいない。</span>
            使っているのにここに出ないウェイトは、ブラウザが近い太さへ落とすか合成する。
            和文 (Noto Sans JP) と別々にずれるので、欧文と和文が混ざる行で太さが揃わない。
          </p>
        </div>
      )}
    </div>
  );
}

// ── 余白・角丸 ────────────────────────────────────────────────────────────

const GAPS = [
  { cls: 'w-0.5', px: '2px', use: '隣接を示すだけの詰め' },
  { cls: 'w-1', px: '4px', use: '表のセル内のアイコンとラベル' },
  { cls: 'w-2', px: '8px', use: '既定の詰め' },
  { cls: 'w-3', px: '12px', use: '' },
  { cls: 'w-4', px: '16px', use: '' },
  { cls: 'w-6', px: '24px', use: '面の内側 (.block-surface)' },
  { cls: 'w-8', px: '32px', use: '節と節' },
  { cls: 'w-12', px: '48px', use: '横に並べた列のあいだ' },
];

function Spacing() {
  const refs = useRef<(HTMLSpanElement | null)[]>([]);
  const [widths, setWidths] = useState<string[]>([]);
  useLayoutEffect(() => {
    setWidths(GAPS.map((_, i) => `${refs.current[i]?.getBoundingClientRect().width ?? 0}px`));
  }, []);

  return (
    <dl className="flex flex-col gap-2">
      {GAPS.map(({ cls, px, use }, i) => (
        <div key={cls} className="flex items-center gap-4">
          <dt className="w-16 font-mono text-2xs text-subtle-foreground">{cls}</dt>
          <dd className="flex items-center gap-3">
            <span ref={(el) => (refs.current[i] = el)} className={`${cls} h-3 bg-primary`} />
            <span className="cell-num text-2xs text-muted-foreground">
              {widths[i] ?? px}
            </span>
            {use && <span className="text-2xs text-subtle-foreground">{use}</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}

// ── 部品 ──────────────────────────────────────────────────────────────────

const TONES: MetricTone[] = ['default', 'positive', 'negative', 'muted'];

function Components() {
  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-3">
        <SectionHeading level={3}>Badge — 形 × 意味</SectionHeading>
        <div className="flex flex-col gap-2">
          {(['solid', 'soft', 'outline'] as const).map((variant) => (
            <div key={variant} className="flex flex-wrap items-center gap-2">
              <span className="w-16 font-mono text-2xs text-subtle-foreground">{variant}</span>
              {(['default', 'success', 'destructive', 'warning'] as const).map((tone) => (
                <Badge key={tone} variant={variant} tone={tone}>
                  {tone}
                </Badge>
              ))}
            </div>
          ))}
        </div>
        <p className="text-2xs text-muted-foreground">
          outline は常に無彩色 (tone が効かない)。solid は「結論」で 1 画面 1 種類まで。
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <SectionHeading level={3}>Button</SectionHeading>
        <div className="flex flex-wrap items-center gap-2">
          {(['default', 'destructive', 'outline', 'secondary', 'ghost'] as const).map((v) => (
            <Button key={v} variant={v} size="sm">
              {v}
            </Button>
          ))}
        </div>
        <p className="text-2xs text-muted-foreground">
          既定は**無彩色の反転**。アクセント (--primary) は「測れているか」を指す色なので
          CTA に使わない。
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <SectionHeading level={3}>MetricCard — 1 画面 1〜3 枚まで</SectionHeading>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {TONES.map((tone) => (
            <MetricCard
              key={tone}
              label={`単勝回収率 (${tone})`}
              value="0.907"
              tone={tone}
              hint="1.00 = 収支トントン。控除率 20% があるので 1.0 未満は平均で負け越し。"
              note={<span className="font-mono">実測 / 2024-10-28 〜 2026-08-23 / 6,244 レース</span>}
            />
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <SectionHeading level={3}>SectionHeading — 罫線は既定で引かない</SectionHeading>
        <div className="flex flex-col gap-4">
          <SectionHeading>罫線なし (既定)</SectionHeading>
          <SectionHeading rule>罫線あり — 横に並べた列の分かれ目だけ</SectionHeading>
          <SectionHeading level={3}>level 3 の小見出し</SectionHeading>
          <SectionHeading aside={<Button size="sm" variant="outline">操作</Button>}>
            aside は見出しタグの外
          </SectionHeading>
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <SectionHeading level={3}>InfoTip / RowActionsMenu</SectionHeading>
        <div className="flex flex-wrap items-center gap-6">
          <span className="flex items-center gap-1 text-label-ja">
            複勝的中率
            <InfoTip
              label="複勝的中率"
              text="上位3頭のうち1頭以上が3着以内だった割合。実測と学習時で数え方が違うので、出所と一緒に読む。"
            />
          </span>
          <RowActionsMenu
            label="見本の操作"
            actions={[
              { label: 'Activate', hint: 'このモデルで買い目を決めるようにする', onSelect: () => {} },
              { label: '計測', hint: '実運用の賭けルールで測り直す', onSelect: () => {} },
              { label: '名称を編集', onSelect: () => {} },
              {
                label: '削除',
                tone: 'destructive',
                hint: 'Active モデルは削除できません',
                disabled: true,
                onSelect: () => {},
              },
            ]}
          />
        </div>
        <p className="text-2xs text-muted-foreground">
          どちらも画面の下端で開くと**上に反転**する (`useAnchoredPosition`)。
        </p>
      </div>
    </div>
  );
}

// ── 状態 ──────────────────────────────────────────────────────────────────

function States() {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      <div className="flex flex-col gap-2">
        <span className="text-label-ja">空 — 既定の状態</span>
        <EmptyState
          message="学習済みモデルはありません"
          description="「再学習を実行」から最初のモデルを学習してください。"
        />
      </div>
      <div className="flex flex-col gap-2">
        <span className="text-label-ja">読み込み中</span>
        <div className="flex flex-col gap-2 py-4">
          <Skeleton className="h-8 w-full rounded-sm" />
          <Skeleton className="h-8 w-full rounded-sm" />
          <Skeleton className="h-8 w-2/3 rounded-sm" />
        </div>
      </div>
      <div className="flex flex-col gap-2">
        <span className="text-label-ja">エラー</span>
        <EmptyState
          icon={SearchX}
          message="メトリクス取得に失敗しました"
          description="バックエンドが起動しているか確認してください。"
        />
      </div>
    </div>
  );
}

// ── 画面 ──────────────────────────────────────────────────────────────────

export function StyleGuide() {
  return (
    <div className="flex flex-col gap-10 p-6">
      <PageHeader
        eyebrow="Style"
        title="見た目の規定"
        description="ここに出る数字は実際に描いた要素を測ったもの。docs/ui-style.md と食い違ったら、実装が変わったということ"
      />

      <section aria-label="色" className="flex flex-col gap-4">
        <SectionHeading>色の 3 層</SectionHeading>
        <Colors />
      </section>

      <section aria-label="字" className="flex flex-col gap-4">
        <SectionHeading>字の尺度</SectionHeading>
        <TypeScale />
        <LoadedFonts />
      </section>

      <section aria-label="余白と角丸" className="flex flex-col gap-4">
        <SectionHeading>余白の尺度</SectionHeading>
        <Spacing />
        <p className="text-2xs text-muted-foreground">
          角丸は 2px の 1 値だけ。ピル (バッジ) の rounded-full が唯一の例外。
        </p>
      </section>

      <section aria-label="部品" className="flex flex-col gap-4">
        <SectionHeading>部品</SectionHeading>
        <Components />
      </section>

      <section aria-label="状態" className="flex flex-col gap-4">
        <SectionHeading>状態の扱い</SectionHeading>
        <States />
      </section>
    </div>
  );
}
