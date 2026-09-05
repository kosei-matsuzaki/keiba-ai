import { useState } from 'react';
import type { ReactNode } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Umaban } from '@/components/Umaban';
import { formatPercent, formatRatio, formatYen } from '@/lib/formatters';
import { buildPurchaseGroups } from '@/lib/purchaseGroups';
import type { PurchaseGroup } from '@/lib/purchaseGroups';
import { useCreateBetsBulk } from '@/hooks/useCreateBetsBulk';
import type { BetType, RecommendationCandidate } from '@/types/api';

/**
 * 買い目を **窓口で買う単位** で出す表。
 *
 * 1 点ずつの表は「何を買うか」は分かるが、そのまま投票するには操作が多すぎる。
 * 同じ集合を流し・ボックス・フォーメーションに畳んで、1 行 = 1 回の投票操作にする。
 * 畳めない集合を畳んだことにすると**買う点数が変わる**ので、形が一致しないときは
 * 素直に「個別」として点数を出す (`lib/purchaseGroups.ts`)。
 *
 * 行を開くと 1 点ずつの明細が出る。金額の変更と単発購入はそこで行う。
 */
interface PurchaseTableProps {
  candidates: RecommendationCandidate[];
  runners: number;
  raceId: string;
  /** 明細行に出す購入 UI。1 点ずつ買う導線は既存のものを渡す。 */
  renderBuy: (candidate: RecommendationCandidate) => ReactNode;
}

function HorseList({ horses, runners }: { horses: number[]; runners: number }) {
  return (
    <span className="inline-flex flex-wrap items-center gap-0.5">
      {horses.map((h) => (
        <Umaban key={h} n={h} runners={runners} size="sm" />
      ))}
    </span>
  );
}

/** 式を馬番チップで組み立てる。窓口の記入欄と同じ並びにする。 */
function Formula({ group, runners }: { group: PurchaseGroup; runners: number }) {
  const { shape } = group;
  if (shape.kind === 'nagashi') {
    return (
      <span className="inline-flex flex-wrap items-center gap-1">
        <HorseList horses={shape.axes} runners={runners} />
        <span className="text-xs text-subtle-foreground">軸から</span>
        <HorseList horses={shape.partners} runners={runners} />
      </span>
    );
  }
  if (shape.kind === 'box') {
    return (
      <span className="inline-flex flex-wrap items-center gap-1">
        <span className="text-xs text-subtle-foreground">BOX</span>
        <HorseList horses={shape.horses} runners={runners} />
      </span>
    );
  }
  if (shape.kind === 'formation') {
    return (
      <span className="inline-flex flex-wrap items-center gap-1">
        {shape.legs.map((leg, i) => (
          <span key={i} className="inline-flex items-center gap-1">
            {i > 0 && <span className="text-xs text-subtle-foreground">→</span>}
            <HorseList horses={leg} runners={runners} />
          </span>
        ))}
      </span>
    );
  }
  if (shape.kind === 'single') {
    return <HorseList horses={shape.horses} runners={runners} />;
  }
  return <span className="text-xs text-muted-foreground">{group.formula}</span>;
}

export function PurchaseTable({ candidates, runners, raceId, renderBuy }: PurchaseTableProps) {
  const groups = buildPurchaseGroups(candidates);
  const [open, setOpen] = useState<Set<string>>(new Set());
  const bulk = useCreateBetsBulk();

  /** 1 グループ = 1 券種なので、そのまま bulk API の 1 リクエストになる。 */
  function buyGroup(group: PurchaseGroup) {
    bulk.mutate({
      race_id: raceId,
      bet_type: group.betType as BetType,
      source: 'recommendation',
      combos: group.candidates.map((c) => ({ combo: c.combo, stake: c.stake })),
    });
  }

  function buyAll() {
    for (const g of groups) buyGroup(g);
  }

  function toggle(key: string) {
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  const total = groups.reduce((n, g) => n + g.totalStake, 0);
  const points = groups.reduce((n, g) => n + g.points, 0);

  return (
    <section aria-label="購入用の買い目" className="flex flex-col gap-3">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-8" />
            <TableHead>券種</TableHead>
            <TableHead>買い方</TableHead>
            <TableHead>買い目</TableHead>
            <TableHead className="text-right">点数</TableHead>
            <TableHead className="text-right">合計</TableHead>
            <TableHead className="text-right">購入</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {groups.map((g) => {
            const isOpen = open.has(g.key);
            return [
              <TableRow
                key={g.key}
                className="cursor-pointer"
                onClick={() => toggle(g.key)}
                title="1 点ずつの明細を開く"
              >
                <TableCell className="text-muted-foreground">
                  {isOpen ? (
                    <ChevronDown className="h-4 w-4" aria-label="閉じる" />
                  ) : (
                    <ChevronRight className="h-4 w-4" aria-label="開く" />
                  )}
                </TableCell>
                <TableCell className="font-medium">{g.betType}</TableCell>
                <TableCell className="text-sm text-muted-foreground">{g.patternLabel}</TableCell>
                <TableCell>
                  <Formula group={g} runners={runners} />
                </TableCell>
                <TableCell className="text-right tabular-nums">{g.points}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {formatYen(g.totalStake)}
                </TableCell>
                {/* 行クリックは開閉なので、購入は伝播を止める */}
                <TableCell className="text-right" onClick={(e) => e.stopPropagation()}>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={bulk.isPending}
                    onClick={() => buyGroup(g)}
                    title={`${g.betType} ${g.points} 点 (${formatYen(g.totalStake)}) をまとめて記録します`}
                  >
                    {g.points} 点を買う
                  </Button>
                </TableCell>
              </TableRow>,
              ...(isOpen
                ? [
                    <TableRow key={`${g.key}-detail`} className="bg-muted/20">
                      <TableCell />
                      <TableCell colSpan={6} className="py-2">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>組合せ</TableHead>
                              <TableHead
                                className="text-right"
                                title="その組合せが当たる確率。買う順序はこれで決めている"
                              >
                                確信度
                              </TableHead>
                              <TableHead
                                className="text-right"
                                title="連系の確信度は確率モデルが直接出しているので、左と同じ数字"
                              >
                                確率モデル
                              </TableHead>
                              <TableHead className="text-right">オッズ</TableHead>
                              <TableHead className="text-right">賭け金 / 購入</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {g.candidates.map((c, i) => (
                              <TableRow key={`${c.combo}-${i}`}>
                                <TableCell>
                                  <span className="inline-flex items-center gap-0.5">
                                    {c.post_positions.map((p, j) => (
                                      <span key={j} className="inline-flex items-center gap-0.5">
                                        {j > 0 && (
                                          <span className="text-2xs text-subtle-foreground">
                                            -
                                          </span>
                                        )}
                                        <Umaban n={p} runners={runners} size="sm" />
                                      </span>
                                    ))}
                                  </span>
                                </TableCell>
                                <TableCell className="text-right tabular-nums">
                                  {formatPercent(c.confidence)}
                                </TableCell>
                                <TableCell className="text-right tabular-nums">
                                  {c.probability_model_confidence == null ? '—' : formatPercent(c.probability_model_confidence)}
                                </TableCell>
                                <TableCell className="text-right tabular-nums">
                                  {c.est_odds === null ? '—' : formatRatio(c.est_odds)}
                                </TableCell>
                                <TableCell className="text-right">{renderBuy(c)}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableCell>
                    </TableRow>,
                  ]
                : []),
            ];
          })}
        </TableBody>
      </Table>

      <div className="flex flex-wrap items-baseline gap-x-4 text-sm">
        <span className="text-muted-foreground">
          合計 <span className="font-medium text-foreground">{formatYen(total)}</span>
        </span>
        <span className="text-muted-foreground">{points} 点</span>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 px-2"
          onClick={() =>
            setOpen((prev) => (prev.size === groups.length ? new Set() : new Set(groups.map((g) => g.key))))
          }
        >
          {open.size === groups.length && groups.length > 0 ? 'すべて閉じる' : 'すべて開く'}
        </Button>
        <Button
          size="sm"
          className="ml-auto h-7"
          disabled={bulk.isPending || groups.length === 0}
          onClick={buyAll}
          title="表示されている買い目をすべて購入記録に入れます"
        >
          {bulk.isPending ? '記録中…' : `全部買う (${formatYen(total)})`}
        </Button>
      </div>
    </section>
  );
}
