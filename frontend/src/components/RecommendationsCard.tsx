import { useState } from 'react';
import type { ChangeEvent } from 'react';
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
import { Umaban } from '@/components/Umaban';
import {
  RecommendationParamsBar,
  type RecommendationOverrides,
} from '@/components/RecommendationParamsBar';
import { formatErrorMessageSync, isNotFoundError, isServiceUnavailableError } from '@/lib/api';
import { formatPercent, formatRatio, formatYen } from '@/lib/formatters';
import { useCreateBet } from '@/hooks/useCreateBet';
import type { RecommendationCandidate, RecommendationsResponse, BetType } from '@/types/api';

// ── Odds source badge ─────────────────────────────────────────────────────────

/**
 * est_odds の出所を視覚的に区別するためのバッジ。
 *
 * 出所は「外から取ってきた事実」のメタ情報なので色相を使わない
 * (緑は「買い」、赤は「マイナス」、青は「AI の出力」に予約している)。
 * 確定 / 実 は無彩色の実体バッジ、推定だけ枠線バッジで弱める。
 */
function OddsSourceBadge({ source }: { source: 'confirmed' | 'scraped' | 'implied' | 'unknown' }) {
  if (source === 'confirmed') {
    return (
      <Badge
        variant="outline"
        className="ml-1 px-1"
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
        className="ml-1 px-1"
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
        className="ml-1 px-1 text-subtle-foreground"
        title="単勝オッズから Plackett-Luce で推定したオッズ"
      >
        推定
      </Badge>
    );
  }
  return null;
}

/**
 * 買い目 (`3-7` `1-5-8` `4` など) を枠色の馬番チップで並べる。
 * 区切り記号はそのまま残し、数字だけをチップにする。
 */
function ComboMarks({ combo, runners }: { combo: string; runners: number }) {
  const parts = combo.split(/([^0-9]+)/).filter((t) => t !== '');
  return (
    <span className="inline-flex items-center gap-0.5" title={combo}>
      {parts.map((part, i) =>
        /^[0-9]+$/.test(part) ? (
          <Umaban key={i} n={Number(part)} runners={runners} size="sm" />
        ) : (
          <span key={i} className="px-0.5 font-mono text-[11px] text-subtle-foreground">
            {part.trim() === '' ? '\u00a0' : part}
          </span>
        )
      )}
    </span>
  );
}

// ── EV coloring ───────────────────────────────────────────────────────────────

function evClass(ev: number | null): string {
  if (ev === null) return 'text-muted-foreground';
  // 緑 = 「買い」推奨。強弱は色相ではなくウェイトで付ける。
  if (ev >= 1.5) return 'font-medium text-success';
  if (ev >= 1.2) return 'text-success';
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

  function handleStakeChange(e: ChangeEvent<HTMLInputElement>) {
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
  /** 枠色の導出に使う出走頭数。0 なら枠色なしの素のチップになる。 */
  runners?: number;
  /** このレースだけの上書き条件 (券種・予算)。 */
  overrides?: RecommendationOverrides;
  onOverridesChange?: (next: RecommendationOverrides) => void;
}

export function RecommendationsCard({
  raceId,
  data,
  isPending,
  isError,
  error,
  runners = 0,
  overrides,
  onOverridesChange,
}: RecommendationsCardProps) {
  return (
    <Card className="border-t border-border pt-6">
      <CardHeader>
        <CardTitle className="text-label-ja">推奨買目</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {overrides != null && onOverridesChange != null && (
          <RecommendationParamsBar value={overrides} onChange={onOverridesChange} />
        )}
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
                このレースの予算:{' '}
                <span className="font-medium text-foreground">
                  {formatYen(data.race_budget)}
                </span>
                <span className="ml-2 text-xs">
                  （実際の合計 {formatYen(data.candidates.reduce((n, c) => n + c.stake, 0))}）
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
                      <TableCell>
                        <ComboMarks combo={c.combo} runners={runners} />
                      </TableCell>
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
            <div className="mt-2 space-y-1 text-xs text-muted-foreground">
              <p>
                {data.odds_source === 'live'
                  ? '※ 当日のライブ市場オッズ（全馬券の実オッズ。未公開の組合せは単勝由来で推定）。'
                  : data.odds_source === 'past'
                    ? '※ 確定オッズ。外れ combo は確定払戻が無いため推定で補完。'
                    : '※ オッズ取得待ち or 該当データなし。'}
                <span className="ml-1">
                  未取得の combo は単勝由来 Plackett-Luce 推定で補完
                  (バッジ「推定」、控除率込み)。
                </span>
              </p>
              {data.place_confidence != null && (
                <p>
                  ※ この予想の確信度 {(data.place_confidence * 100).toFixed(1)}%
                  {data.place_confidence_threshold != null && (
                    <>
                      （複勝を買う下限 {(data.place_confidence_threshold * 100).toFixed(0)}%）。
                      {data.place_confidence < data.place_confidence_threshold
                        ? '下回るため複勝は見送っています。'
                        : '上回るため複勝も買います。'}
                    </>
                  )}
                </p>
              )}
              <p>
                ※ 買い目は期待値ではなく<strong>的中確率の高い順</strong>に選んでいます
                （期待値で絞ると大穴に寄り、実測で回収率が落ちるため）。
                実測の回収率は単勝 0.93 / 複勝 0.89（確信度で絞ると 0.92）/ 連系 0.85〜0.88 で、
                <strong>いずれも 1.0 未満</strong>です。控除率の内側なので実買いは慎重に。
              </p>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
