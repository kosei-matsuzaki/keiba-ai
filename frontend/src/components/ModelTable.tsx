import { useNavigate } from 'react-router-dom';
import { Pencil, Trash2 } from 'lucide-react';

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
  onEdit: (model: ModelMeta) => void;
  onDelete: (model: ModelMeta) => void;
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

export function ModelTable({
  models,
  onActivate,
  onEdit,
  onDelete,
  activatingId,
}: ModelTableProps) {
  const navigate = useNavigate();
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
          <TableHead>名称</TableHead>
          <TableHead>作成日時</TableHead>
          <TableHead>学習期間</TableHead>
          <TableHead>検証期間</TableHead>
          <TableHead className="text-right">NDCG@3</TableHead>
          <TableHead className="text-right">単勝回収率</TableHead>
          <TableHead className="text-center">状態</TableHead>
          <TableHead className="text-right">操作</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {models.map((model) => (
          <TableRow
            key={model.id}
            className={`cursor-pointer ${model.is_active ? 'bg-success/5' : ''}`}
            onClick={() => navigate(`/models/${model.id}`)}
            title="詳細 / バックテストを開く"
          >
            <TableCell>{model.id}</TableCell>
            <TableCell>
              {model.name?.trim() ? (
                model.name
              ) : (
                <span className="text-muted-foreground">{PLACEHOLDER}</span>
              )}
            </TableCell>
            <TableCell className="text-xs">{formatDateTime(model.created_at)}</TableCell>
            <TableCell className="text-xs">{model.train_range ?? PLACEHOLDER}</TableCell>
            <TableCell className="text-xs">{model.valid_range ?? PLACEHOLDER}</TableCell>
            <TableCell className="text-right">{extractMetric(model.metrics, 'ndcg3', 'score')}</TableCell>
            <TableCell className="text-right">{extractMetric(model.metrics, 'payback_win', 'ratio')}</TableCell>
            <TableCell className="text-center">
              {/* 役割は 2 つある: Active = 買う馬を決める / 確率 = 確からしさを出す。
                  両方を兼ねることもあるので併記する。 */}
              <div className="flex flex-wrap items-center justify-center gap-1">
                {model.is_active ? (
                  <Badge tone="success">Active</Badge>
                ) : (
                  !model.is_probability_model && <Badge variant="outline">非アクティブ</Badge>
                )}
                {model.is_probability_model && (
                  <Badge title="複勝の確信度と連系の確率に使われています">
                    確率
                  </Badge>
                )}
              </div>
            </TableCell>
            <TableCell onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-end gap-2">
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
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label="名称を編集"
                  title="名称を編集"
                  onClick={() => onEdit(model)}
                >
                  <Pencil className="h-4 w-4" />
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label="削除"
                  title={model.is_active ? 'Active モデルは削除できません' : '削除'}
                  disabled={model.is_active}
                  onClick={() => onDelete(model)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
