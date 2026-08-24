import { Link } from 'react-router-dom';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { formatDateTime, formatRatio, formatScore } from '@/lib/formatters';
import { labelClass } from '@/lib/labels';
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
    <Card className={linkToModels ? 'group transition-colors' : ''}>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-3 text-sm font-medium text-muted-foreground">
          Active モデル
          {/* ID は意味を持たない識別子なので色を付けない (緑は「買い」専用)。 */}
          <span className="font-mono text-xs text-muted-foreground">ID {model.id}</span>
          {model.name?.trim() && (
            <span className="text-sm font-normal text-foreground">{model.name}</span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* メタ情報 (作成 / 学習期間) は左に小さく、指標 (NDCG / 回収率) は
            右に大きく。全部同じ大きさだと何も強調していないのと同じ。 */}
        <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
          <div className="grid grid-cols-2 gap-x-6 gap-y-1">
            <Meta label="作成" value={formatDateTime(model.created_at)} />
            <Meta label="学習期間" value={model.train_range ?? PLACEHOLDER} />
          </div>
          <div className="flex gap-8">
            <Metric label="NDCG@3" value={metric(model.metrics, 'ndcg3', 'score')} />
            <Metric label="単勝回収率" value={metric(model.metrics, 'payback_win', 'ratio')} />
          </div>
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

/** メタ情報 (いつ / どの期間で学習したか) — 小さく、控えめに。 */
function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono text-xs text-muted-foreground">{value}</div>
    </div>
  );
}

/** 指標 — このカードで一番見たい値なので、右寄せで一段大きく出す。 */
function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-right">
      <div className={labelClass(label)}>{label}</div>
      <div className="text-num text-xl font-medium leading-tight">{value}</div>
    </div>
  );
}
