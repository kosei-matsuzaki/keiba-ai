import { useState } from 'react';
import { Wallet } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { EmptyState } from '@/components/EmptyState';
import { formatErrorMessageSync, isNotFoundError, isServiceUnavailableError } from '@/lib/api';
import { formatPercent, formatRatio, formatYen } from '@/lib/formatters';
import { useCreateBet } from '@/hooks/useCreateBet';
import type { RecommendationCandidate, RecommendationsResponse, BetType } from '@/types/api';

// ── Odds source badge ─────────────────────────────────────────────────────────

/**
 * est_odds の出所を視覚的に区別するためのバッジ。
 * - confirmed (確定): 緑系の控えめなアウトラインバッジ
 * - implied  (推定): 黄系で「推」表記。tooltip に詳細
 * - unknown  : なし（— が表示されている前提）
 */
function OddsSourceBadge({ source }: { source: 'confirmed' | 'scraped' | 'implied' | 'unknown' }) {
  if (source === 'confirmed') {
    return (
      <Badge
        variant="outline"
        className="ml-1 border-emerald-300 px-1 text-[10px] font-normal text-emerald-700 dark:border-emerald-700 dark:text-emerald-400"
        title="確定オッズ（payouts / entries.odds_win 由来）"
      >
        確定
      </Badge>
    );
  }
  if (source === 'scraped') {
    return (
      <Badge
        variant="outline"
        className="ml-1 border-sky-300 px-1 text-[10px] font-normal text-sky-700 dark:border-sky-700 dark:text-sky-400"
        title="実市場オッズ（odds.db に取り込んだ全 combo 確定オッズ）"
      >
        実
      </Badge>
    );
  }
  if (source === 'implied') {
    return (
      <Badge
        variant="outline"
        className="ml-1 border-amber-300 px-1 text-[10px] font-normal text-amber-700 dark:border-amber-700 dark:text-amber-400"
        title="単勝オッズから Plackett-Luce で推定したオッズ"
      >
        推定
      </Badge>
    );
  }
  return null;
}

// ── EV coloring ───────────────────────────────────────────────────────────────

function evClass(ev: number | null): string {
  if (ev === null) return 'text-muted-foreground';
  if (ev >= 1.5) return 'text-green-600 font-semibold';
  if (ev >= 1.2) return 'text-yellow-600';
  return 'text-muted-foreground';
}

// ── StakeInputAndBuy ──────────────────────────────────────────────────────────

interface StakeInputAndBuyProps {
  candidate: RecommendationCandidate;
  raceId: string;
}

/**
 * 賭け金の入力フィールドと「買う」ボタンを横並びで表示する。
 *
 * - default は AI 推奨 stake (`candidate.stake`)
 * - ユーザは 100 円単位で自由に変更可能 (例: 推奨 0 でも 100 円で勝負試したい等)
 * - 0 円 / 空欄 / 100 円未満は「買う」を disable
 * - 入力は 100 円刻みに自動 round (snap)
 */
function StakeInputAndBuy({ candidate, raceId }: StakeInputAndBuyProps) {
  const [stake, setStake] = useState<number>(candidate.stake);
  const { mutate, isPending } = useCreateBet();

  function handleStakeChange(e: React.ChangeEvent<HTMLInputElement>) {
    const raw = Number(e.target.value);
    if (Number.isNaN(raw) || raw < 0) {
      setStake(0);
      return;
    }
    // Snap to 100 円 単位
    setStake(Math.floor(raw / 100) * 100);
  }

  function handleBuy() {
    if (stake < 100) return;
    mutate({
      race_id: raceId,
      bet_type: candidate.bet_type as BetType,
      combo: candidate.combo,
      stake,
      source: 'recommendation',
    });
  }

  return (
    <div className="flex items-center justify-end gap-1">
      <Input
        type="number"
        min={0}
        step={100}
        value={stake}
        onChange={handleStakeChange}
        className="h-8 w-24 text-right text-sm"
        aria-label="賭け金 (円, 100 円単位)"
      />
      <Button
        size="sm"
        variant="outline"
        disabled={isPending || stake < 100}
        onClick={handleBuy}
      >
        買う
      </Button>
    </div>
  );
}

// ── Candidate sorting ─────────────────────────────────────────────────────────

/**
 * Sort candidates: stake desc → ev desc (null last) → prob desc.
 * This ensures recommended (stake > 0) candidates appear above zero-stake ones,
 * and candidates with null ev/est_odds are pinned to the bottom.
 */
function sortCandidates(candidates: RecommendationCandidate[]): RecommendationCandidate[] {
  return [...candidates].sort((a, b) => {
    if (b.stake !== a.stake) return b.stake - a.stake;
    // null ev rows sink to the bottom
    if (a.ev === null && b.ev === null) return b.prob - a.prob;
    if (a.ev === null) return 1;
    if (b.ev === null) return -1;
    if (b.ev !== a.ev) return b.ev - a.ev;
    return b.prob - a.prob;
  });
}

// ── Main component ────────────────────────────────────────────────────────────

interface RecommendationsCardProps {
  raceId: string;
  data: RecommendationsResponse | undefined;
  isPending: boolean;
  isError: boolean;
  error: unknown;
}

export function RecommendationsCard({
  raceId,
  data,
  isPending,
  isError,
  error,
}: RecommendationsCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">推奨買目</CardTitle>
      </CardHeader>
      <CardContent>
        {isPending ? (
          <Skeleton className="h-40 w-full" />
        ) : isError ? (
          <EmptyState
            message="推奨買目を取得できません"
            description={
              isServiceUnavailableError(error)
                ? 'active モデルが見つかりません。Models 画面から train を実行してください。'
                : isNotFoundError(error)
                  ? 'このレースの推奨買目はありません。'
                  : formatErrorMessageSync(error)
            }
          />
        ) : !data || data.candidates.length === 0 ? (
          <EmptyState
            icon={Wallet}
            message="現在のフィルタで推奨候補がありません"
            description="EV が十分な組合せがないか、enabled_bet_types で絞り込まれています。"
          />
        ) : (
          <>
            <div className="mb-3 flex flex-wrap items-baseline gap-x-4 gap-y-1">
              <p className="text-sm text-muted-foreground">
                判断時バンクロール:{' '}
                <span className="font-medium text-foreground">
                  {formatYen(data.bankroll_at_decision)}
                </span>
              </p>
              <p className="text-xs text-muted-foreground">
                {data.candidates.length} 候補
                （うち {data.candidates.filter((c) => c.stake > 0).length} 件が推奨）
              </p>
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>券種</TableHead>
                  <TableHead>組合せ</TableHead>
                  <TableHead className="text-right">確率</TableHead>
                  <TableHead className="text-right">推定オッズ</TableHead>
                  <TableHead className="text-right">EV</TableHead>
                  <TableHead className="text-right">推奨 stake</TableHead>
                  <TableHead className="text-right">賭け金 / 購入</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortCandidates(data.candidates).map((c, idx) => {
                  const isZeroStake = c.stake === 0;
                  const rowClass = isZeroStake ? 'opacity-60' : '';
                  return (
                    <TableRow key={`${c.bet_type}-${c.combo}-${idx}`} className={rowClass}>
                      <TableCell className="font-medium">{c.bet_type}</TableCell>
                      <TableCell className="font-mono text-xs">{c.combo}</TableCell>
                      <TableCell className="text-right">{formatPercent(c.prob)}</TableCell>
                      <TableCell className="text-right">
                        {c.est_odds === null ? (
                          <span className="text-muted-foreground">—</span>
                        ) : (
                          <span className="inline-flex items-center justify-end">
                            {formatRatio(c.est_odds)}
                            <OddsSourceBadge source={c.est_odds_source ?? 'unknown'} />
                          </span>
                        )}
                      </TableCell>
                      <TableCell className={`text-right ${evClass(c.ev)}`}>
                        {c.ev === null ? (
                          <span className="text-muted-foreground">—</span>
                        ) : (
                          formatRatio(c.ev)
                        )}
                      </TableCell>
                      <TableCell className="text-right text-muted-foreground">
                        {isZeroStake ? '—' : formatYen(c.stake)}
                      </TableCell>
                      <TableCell className="text-right">
                        <StakeInputAndBuy candidate={c} raceId={raceId} />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
            <p className="mt-2 text-xs text-muted-foreground">
              {data.odds_source === 'live'
                ? '※ 当日の市場オッズ（単勝由来）。'
                : data.odds_source === 'past'
                  ? '※ 確定オッズ。'
                  : '※ オッズ取得待ち or 該当データなし。'}
              <span className="ml-1">
                未取得の combo は単勝由来 Plackett-Luce 推定で補完
                (バッジ「推定」、控除率込み)。
              </span>
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
