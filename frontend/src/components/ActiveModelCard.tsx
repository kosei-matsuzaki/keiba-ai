import { Link } from 'react-router-dom';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { formatDateTime, formatRatio, formatScore } from '@/lib/formatters';
import type { ModelMeta } from '@/types/api';

interface ActiveModelCardProps {
  model: ModelMeta | null;
  /** Set to false on the Models page itself so the card doesn't link back to itself. */
  linkToModels?: boolean;
}

const PLACEHOLDER = '—';

function metric(metrics: Record<string, unknown> | null, key: string, fmt: 'score' | 'ratio') {
  if (!metrics) return PLACEHOLDER;
  const v = metrics[key];
  if (typeof v !== 'number') return PLACEHOLDER;
  return fmt === 'score' ? formatScore(v) : formatRatio(v);
}

export function ActiveModelCard({ model, linkToModels = true }: ActiveModelCardProps) {
  if (!model) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
            Active モデル
            <Badge variant="outline">未設定</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          学習済モデルがありません。Models 画面から再学習を実行してください。
        </CardContent>
      </Card>
    );
  }

  const body = (
    <Card className={linkToModels ? 'hover:border-primary/40 transition-colors' : ''}>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-3 text-sm font-medium text-muted-foreground">
          Active モデル
          <Badge className="bg-emerald-600 text-white">ID {model.id}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-4">
          <Field label="作成" value={formatDateTime(model.created_at)} />
          <Field label="学習期間" value={model.train_range ?? PLACEHOLDER} />
          <Field label="NDCG@3" value={metric(model.metrics, 'ndcg3', 'score')} />
          <Field label="単勝回収率" value={metric(model.metrics, 'payback_win', 'ratio')} />
        </div>
      </CardContent>
    </Card>
  );

  if (!linkToModels) return body;
  return (
    <Link to="/models" className="block">
      {body}
    </Link>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono text-xs">{value}</div>
    </div>
  );
}
