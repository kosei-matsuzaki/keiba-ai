import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { formatDateTime, formatRatio, formatScore } from '@/lib/formatters';
import type { ModelMeta } from '@/types/api';

interface ModelTableProps {
  models: ModelMeta[];
  onActivate: (id: number) => void;
  activatingId: number | null;
}

const PLACEHOLDER = '—';

/** Pull a numeric metric from the loose `metrics` JSON and render it via the
 *  shared formatter. ndcg-like values use 3 digits; ratio-like values 2.
 */
function extractMetric(
  metrics: Record<string, unknown> | null,
  key: string,
  format: 'score' | 'ratio',
): string {
  if (!metrics) return PLACEHOLDER;
  const v = metrics[key];
  if (typeof v !== 'number') return PLACEHOLDER;
  return format === 'ratio' ? formatRatio(v) : formatScore(v);
}

export function ModelTable({ models, onActivate, activatingId }: ModelTableProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
          <TableHead>作成日時</TableHead>
          <TableHead>学習期間</TableHead>
          <TableHead>検証期間</TableHead>
          <TableHead className="text-right">NDCG@3</TableHead>
          <TableHead className="text-right">単勝回収率</TableHead>
          <TableHead className="text-center">状態</TableHead>
          <TableHead></TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {models.map((model) => (
          <TableRow
            key={model.id}
            className={model.is_active ? 'bg-emerald-500/5' : undefined}
          >
            <TableCell>{model.id}</TableCell>
            <TableCell className="text-xs">{formatDateTime(model.created_at)}</TableCell>
            <TableCell className="text-xs">{model.train_range ?? PLACEHOLDER}</TableCell>
            <TableCell className="text-xs">{model.valid_range ?? PLACEHOLDER}</TableCell>
            <TableCell className="text-right">{extractMetric(model.metrics, 'ndcg3', 'score')}</TableCell>
            <TableCell className="text-right">{extractMetric(model.metrics, 'payback_win', 'ratio')}</TableCell>
            <TableCell className="text-center">
              {model.is_active ? (
                <Badge className="bg-emerald-600 text-white">Active</Badge>
              ) : (
                <Badge variant="outline">非アクティブ</Badge>
              )}
            </TableCell>
            <TableCell>
              {!model.is_active && (
                <Button
                  size="sm"
                  variant="outline"
                  disabled={activatingId !== null}
                  onClick={() => onActivate(model.id)}
                >
                  {activatingId === model.id ? '切り替え中…' : 'Activate'}
                </Button>
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
