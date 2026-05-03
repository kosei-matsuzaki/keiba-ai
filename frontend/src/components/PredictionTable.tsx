import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { formatPercent, formatScore } from '@/lib/formatters';
import type { HorsePrediction, EntrySummary } from '@/types/api';

interface PredictionTableProps {
  predictions: HorsePrediction[];
  entries: EntrySummary[];
}

/** Indicates BUY when single-win expected value > 1.1 */
function isBuy(pred: HorsePrediction, entries: EntrySummary[]): boolean {
  const entry = entries.find((e) => e.horse_id === pred.horse_id);
  if (!entry?.odds_win) return false;
  return pred.win_prob * entry.odds_win > 1.1;
}

export function PredictionTable({ predictions, entries }: PredictionTableProps) {
  const sorted = [...predictions].sort((a, b) => b.score - a.score);

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>馬 ID</TableHead>
          <TableHead className="text-right">スコア</TableHead>
          <TableHead className="text-right">単勝確率</TableHead>
          <TableHead className="text-right">複勝確率</TableHead>
          <TableHead className="text-center">推奨</TableHead>
          <TableHead>SHAP</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {sorted.map((pred) => (
          <TableRow key={pred.horse_id}>
            <TableCell className="font-mono text-xs">{pred.horse_id}</TableCell>
            <TableCell className="text-right">{formatScore(pred.score)}</TableCell>
            <TableCell className="text-right">{formatPercent(pred.win_prob)}</TableCell>
            <TableCell className="text-right">{formatPercent(pred.place_prob)}</TableCell>
            <TableCell className="text-center">
              {isBuy(pred, entries) && (
                <Badge variant="success">BUY</Badge>
              )}
            </TableCell>
            <TableCell className="text-xs text-muted-foreground italic">
              SHAP 寄与は M9 以降
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
