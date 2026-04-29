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
import type { ModelMeta } from '@/types/api';

interface ModelTableProps {
  models: ModelMeta[];
  onActivate: (id: number) => void;
  activatingId: number | null;
}

function extractMetric(metrics: Record<string, unknown> | null, key: string): string {
  if (!metrics) return '—';
  const v = metrics[key];
  if (typeof v === 'number') return v.toFixed(4);
  return '—';
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
          <TableRow key={model.id}>
            <TableCell>{model.id}</TableCell>
            <TableCell className="text-xs">{model.created_at.slice(0, 19).replace('T', ' ')}</TableCell>
            <TableCell className="text-xs">{model.train_range ?? '—'}</TableCell>
            <TableCell className="text-xs">{model.valid_range ?? '—'}</TableCell>
            <TableCell className="text-right">{extractMetric(model.metrics, 'ndcg3')}</TableCell>
            <TableCell className="text-right">{extractMetric(model.metrics, 'payback_win')}</TableCell>
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
