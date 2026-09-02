import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { formatPercent, formatSignedYen, formatYen } from '@/lib/formatters';
import type { PayoutEntry, RecommendationCandidate } from '@/types/api';

/**
 * 答え合わせ — **推奨買目をそのまま全部買った場合**の収支。
 *
 * 推奨買目のタブの 1 つとして出す。買い目と結果が別の場所にあると、
 * 「どの買い目がどうなったか」を目で往復しないと分からない。
 *
 * 払戻は `payouts` (100 円あたり) の厳密マッチ。表記ゆれはバックエンドで
 * 正規化済み (`core.bet_types.normalize_combo`)。
 */
interface BetTypeReview {
  betType: string;
  points: number;
  invested: number;
  returned: number;
  hits: number;
}

function reviewCandidates(
  candidates: RecommendationCandidate[],
  payouts: PayoutEntry[]
): { rows: BetTypeReview[]; invested: number; returned: number } | null {
  const buying = candidates.filter((c) => c.stake > 0);
  if (buying.length === 0) return null;
  const key = (betType: string, combo: string) => `${betType}|${combo}`;
  const paid = new Map(payouts.map((p) => [key(p.bet_type, p.combo), p.amount]));
  const byType = new Map<string, BetTypeReview>();
  for (const c of buying) {
    const row = byType.get(c.bet_type) ?? {
      betType: c.bet_type,
      points: 0,
      invested: 0,
      returned: 0,
      hits: 0,
    };
    // payouts.amount は 100 円あたりの払戻。賭け金に比例させる。
    const amount = paid.get(key(c.bet_type, c.combo)) ?? 0;
    row.points += 1;
    row.invested += c.stake;
    row.returned += (amount * c.stake) / 100;
    if (amount > 0) row.hits += 1;
    byType.set(c.bet_type, row);
  }
  const rows = [...byType.values()];
  return {
    rows,
    invested: rows.reduce((n, r) => n + r.invested, 0),
    returned: rows.reduce((n, r) => n + r.returned, 0),
  };
}

interface ResultReviewTabProps {
  candidates: RecommendationCandidate[];
  payouts: PayoutEntry[];
}

export function ResultReviewTab({ candidates, payouts }: ResultReviewTabProps) {
  const full = payouts.length > 0 ? reviewCandidates(candidates, payouts) : null;

  if (!full) {
    return (
      <p className="py-6 text-sm text-muted-foreground">
        まだ結果が出ていません。確定するとここに、この買い目をそのまま買った場合の収支が出ます。
      </p>
    );
  }

  const roi = full.invested > 0 ? full.returned / full.invested : null;
  const profit = Math.round(full.returned) - full.invested;
  const hits = full.rows.reduce((n, r) => n + r.hits, 0);
  const points = full.rows.reduce((n, r) => n + r.points, 0);

  return (
    <section aria-label="答え合わせ" className="flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline gap-x-8 gap-y-2 text-sm">
        <span>
          <span className="text-label">収支</span>{' '}
          <span
            className={`font-mono text-lg font-medium tabular-nums ${
              profit >= 0 ? 'text-success' : 'text-destructive'
            }`}
          >
            {formatSignedYen(profit)}
          </span>
        </span>
        <span>
          <span className="text-label">回収率</span>{' '}
          <span
            className={`font-mono tabular-nums ${
              roi != null && roi >= 1 ? 'text-success' : 'text-destructive'
            }`}
          >
            {roi == null ? '—' : formatPercent(roi, 0)}
          </span>
        </span>
        <span>
          <span className="text-label">投資</span>{' '}
          <span className="font-mono tabular-nums">{formatYen(full.invested)}</span>
          <span className="mx-1 text-subtle-foreground">→</span>
          <span className="text-label">払戻</span>{' '}
          <span className="font-mono tabular-nums">{formatYen(Math.round(full.returned))}</span>
        </span>
        <span>
          <span className="text-label">的中</span>{' '}
          <span className="font-mono tabular-nums">
            {hits} / {points}
          </span>{' '}
          点
        </span>
      </div>

      <Table aria-label="券種別の答え合わせ">
        <TableHeader>
          <TableRow>
            <TableHead>券種</TableHead>
            <TableHead className="text-right">的中 / 点数</TableHead>
            <TableHead className="text-right">投資</TableHead>
            <TableHead className="text-right">払戻</TableHead>
            <TableHead className="text-right">回収率</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {full.rows.map((r) => {
            const rowRoi = r.invested > 0 ? r.returned / r.invested : null;
            return (
              <TableRow key={r.betType}>
                <TableCell className="font-medium">{r.betType}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {r.hits} / {r.points}
                </TableCell>
                <TableCell className="text-right tabular-nums">{formatYen(r.invested)}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {formatYen(Math.round(r.returned))}
                </TableCell>
                <TableCell
                  className={`text-right tabular-nums ${
                    rowRoi != null && rowRoi >= 1 ? 'text-success' : 'text-muted-foreground'
                  }`}
                >
                  {rowRoi == null ? '—' : formatPercent(rowRoi, 0)}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>

    </section>
  );
}
