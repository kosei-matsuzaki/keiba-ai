import { Wallet } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
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

// ── Pattern badge variant mapping ─────────────────────────────────────────────

const PATTERN_LABELS: Record<string, string> = {
  nagashi: '流し',
  box:     'ボックス',
  formation: 'フォーメーション',
};

function PatternBadge({ pattern }: { pattern: string }) {
  return (
    <Badge variant="outline" className="text-xs">
      {PATTERN_LABELS[pattern] ?? pattern}
    </Badge>
  );
}

// ── EV coloring ───────────────────────────────────────────────────────────────

function evClass(ev: number): string {
  if (ev >= 1.5) return 'text-green-600 font-semibold';
  if (ev >= 1.2) return 'text-yellow-600';
  return 'text-muted-foreground';
}

// ── BuyButton ─────────────────────────────────────────────────────────────────

interface BuyButtonProps {
  candidate: RecommendationCandidate;
  raceId: string;
}

function BuyButton({ candidate, raceId }: BuyButtonProps) {
  const { mutate, isPending } = useCreateBet();

  function handleBuy() {
    mutate({
      race_id: raceId,
      bet_type: candidate.bet_type as BetType,
      combo: candidate.combo,
      stake: candidate.stake,
      source: 'recommendation',
    });
  }

  return (
    <Button
      size="sm"
      variant="outline"
      disabled={isPending || candidate.stake === 0}
      onClick={handleBuy}
    >
      買う
    </Button>
  );
}

// ── Candidate sorting ─────────────────────────────────────────────────────────

/**
 * Sort candidates: stake desc → ev desc → prob desc.
 * This ensures recommended (stake > 0) candidates appear above zero-stake ones.
 */
function sortCandidates(candidates: RecommendationCandidate[]): RecommendationCandidate[] {
  return [...candidates].sort((a, b) => {
    if (b.stake !== a.stake) return b.stake - a.stake;
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
                  <TableHead>パターン</TableHead>
                  <TableHead className="text-right">確率</TableHead>
                  <TableHead className="text-right">
                    推定オッズ
                  </TableHead>
                  <TableHead className="text-right">EV</TableHead>
                  <TableHead className="text-right">推奨 stake</TableHead>
                  <TableHead />
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
                      <TableCell>
                        <PatternBadge pattern={c.pattern} />
                      </TableCell>
                      <TableCell className="text-right">{formatPercent(c.prob)}</TableCell>
                      <TableCell className="text-right">{formatRatio(c.est_odds)}</TableCell>
                      <TableCell className={`text-right ${evClass(c.ev)}`}>
                        {formatRatio(c.ev)}
                      </TableCell>
                      <TableCell className="text-right">
                        {isZeroStake ? (
                          <span className="text-muted-foreground">賭けない</span>
                        ) : (
                          formatYen(c.stake)
                        )}
                      </TableCell>
                      <TableCell className="text-right">
                        <BuyButton candidate={c} raceId={raceId} />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
            <p className="mt-2 text-xs text-muted-foreground">
              ※ 推定オッズは過去払戻の平均値（暫定）。当日オッズ未対応
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
