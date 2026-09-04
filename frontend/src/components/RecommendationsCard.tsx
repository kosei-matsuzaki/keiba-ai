import { useState } from 'react';
import type { ChangeEvent, ReactNode } from 'react';
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
import { BettingRuleDetails } from '@/components/BettingRuleDetails';
import { isComboBetType } from '@/lib/betTypes';
import { ResultReviewTab } from '@/components/ResultReviewTab';
import { EmptyState } from '@/components/EmptyState';
import { PurchaseTable } from '@/components/PurchaseTable';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Umaban } from '@/components/Umaban';
import {
  RecommendationParamsBar,
  type RecommendationOverrides,
} from '@/components/RecommendationParamsBar';
import { formatErrorMessageSync, isNotFoundError, isServiceUnavailableError } from '@/lib/api';
import { formatPercent, formatRatio, formatYen } from '@/lib/formatters';
import { useCreateBet } from '@/hooks/useCreateBet';
import type {
  BetType,
  PayoutEntry,
  RecommendationCandidate,
  RecommendationsResponse,
} from '@/types/api';

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

/** 券種の優先度。エンジン (assign_flat_stakes) と同じ順序。 */
const BET_TYPE_PRIORITY: Record<string, number> = { 単勝: 0, 複勝: 1 };

/**
 * **エンジンが賭ける順序をそのまま出す**: 買う → 単勝 → 複勝 → 連系 → 確信度の高い順。
 *
 * 以前は EV の降順で並べていたが、エンジンは EV を見ていない。表の順序と実際の
 * 買い方が違うと「上から順に買えばよい」という読み方ができなくなる。EV 順は
 * とくに危険で、較正済みの確率だと単勝の EV が連系より低く出るため、
 * 回収率の推定が最も確かな単複が下に沈む。
 */
function sortCandidates(candidates: RecommendationCandidate[]): RecommendationCandidate[] {
  return [...candidates].sort((a, b) => {
    if ((b.stake > 0 ? 1 : 0) !== (a.stake > 0 ? 1 : 0)) return b.stake > 0 ? 1 : -1;
    const pa = BET_TYPE_PRIORITY[a.bet_type] ?? 2;
    const pb = BET_TYPE_PRIORITY[b.bet_type] ?? 2;
    if (pa !== pb) return pa - pb;
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
  /** 全券種の確定払戻。あれば「答え合わせ」タブを出す。 */
  payouts?: PayoutEntry[];
  /**
   * 出走馬一覧。渡すと 1 つ目のタブになる。
   *
   * **タブが出るのは買い目が取れているときだけ** なので、取れないときは
   * 呼び出し側が出走馬一覧を単独で描くこと (ここに渡すと消えてしまう)。
   */
  entriesTab?: ReactNode;
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
  payouts = [],
  entriesTab,
}: RecommendationsCardProps) {
  // 1 点あたりの金額。賭け金は必ずこの倍数なので、点数は stake / unit で戻せる。
  const unit = data?.stake_unit || 100;
  return (
    <Card className="border-t border-border pt-6">
      <CardHeader>
        <CardTitle className="text-label-ja">推奨買目</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {overrides != null && onOverridesChange != null && (
          <>
            <RecommendationParamsBar value={overrides} onChange={onOverridesChange} />
            {/* 買い方は「買う馬券」の直下に置く。何を買うかを決めたすぐ後に
                「どういう条件で買うのか」を読めるようにする。 */}
            <BettingRuleDetails />
          </>
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
            message="このレースの買い目がありません"
            description="オッズが取れないか、買う馬券の設定で絞り込まれています。"
          />
        ) : (
          <>
            <Tabs defaultValue="detail">
              <TabsList>
                <TabsTrigger value="detail">1 点ずつ</TabsTrigger>
                <TabsTrigger value="purchase">購入用</TabsTrigger>
                {/* 出走馬は参照するデータなので買い目の後ろ。既定は「1 点ずつ」の
                    まま — 買い目を先に見せる方針は変えない */}
                {entriesTab != null && <TabsTrigger value="entries">出走馬</TabsTrigger>}
                {/* 結果が出ているレースだけ。買い目と同じ場所で振り返れるようにする */}
                {payouts.length > 0 && <TabsTrigger value="review">答え合わせ</TabsTrigger>}
              </TabsList>

              {entriesTab != null && (
                <TabsContent value="entries" className="pt-3">
                  {entriesTab}
                </TabsContent>
              )}

              <TabsContent value="review" className="pt-3">
                <ResultReviewTab candidates={data.candidates} payouts={payouts} />
              </TabsContent>

              <TabsContent value="purchase" className="pt-3">
                <PurchaseTable
                  candidates={data.candidates}
                  raceId={raceId}
                  runners={runners}
                  renderBuy={(c) => <StakeInputAndBuy candidate={c} raceId={raceId} />}
                />
              </TabsContent>

              <TabsContent value="detail" className="pt-3">
            <Table aria-label="推奨買目の一覧">
              <TableHeader>
                <TableRow>
                  <TableHead>券種</TableHead>
                  <TableHead>組合せ</TableHead>
                  <TableHead
                    className="text-right"
                    title="その買い目が当たる確率。単勝=1着 / 複勝=3着以内 / 連系=その組合せ。買う順序はこれで決めている"
                  >
                    確信度
                  </TableHead>
                  <TableHead
                    className="text-right"
                    title="同じ確信度を、確率専用モデルが答えたもの。複勝を買うかと厚みはこれで決める。連系はもともと確率モデルが出しているので「同じ」と表示する"
                  >
                    確率モデル
                  </TableHead>
                  <TableHead className="text-right">推定オッズ</TableHead>
                  <TableHead
                    className="text-right"
                    title="この買い目に賭ける金額。1 点あたりの金額 × 点数。単勝・複勝は基準 5 点で、複勝だけ確信度に応じて 1〜15 点に増減する。連系は 1 点ずつ"
                  >
                    賭け金
                  </TableHead>
                  <TableHead
                    className="text-right text-subtle-foreground"
                    title="参考値。確信度 × オッズ。買う / 買わないの判定には使っていない (使うと大穴に寄り、実測で回収率が落ちる)"
                  >
                    参考 EV
                  </TableHead>
                  <TableHead className="text-right">購入</TableHead>
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
                      <TableCell className="text-right tabular-nums">
                        {isComboBetType(c.bet_type) ? (
                          // 連系の確信度は確率モデルが直接出したもの。左の列と同じ値
                          <span
                            className="text-subtle-foreground"
                            title="連系の確信度は確率モデルが直接出しているので、左と同じ数字です"
                          >
                            同じ
                          </span>
                        ) : c.confidence == null ? (
                          <span className="text-subtle-foreground">—</span>
                        ) : (
                          formatPercent(c.confidence)
                        )}
                      </TableCell>
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
                      <TableCell className="text-right">
                        {isZeroStake ? (
                          <span className="text-muted-foreground">—</span>
                        ) : (
                          <span className="inline-flex items-baseline gap-1 tabular-nums">
                            <span className="font-medium">{formatYen(c.stake)}</span>
                            <span className="text-xs text-subtle-foreground">
                              {Math.round(c.stake / unit)} 点
                            </span>
                          </span>
                        )}
                      </TableCell>
                      <TableCell className="text-right text-subtle-foreground">
                        {c.ev === null ? '—' : formatRatio(c.ev)}
                      </TableCell>
                      <TableCell className="text-right">
                        <StakeInputAndBuy candidate={c} raceId={raceId} />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
              </TabsContent>
            </Tabs>
          </>
        )}
      </CardContent>
    </Card>
  );
}
