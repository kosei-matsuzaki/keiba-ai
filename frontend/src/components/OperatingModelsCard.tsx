import { Link } from 'react-router-dom';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { formatDateTime, formatRatio, formatScore } from '@/lib/formatters';
import type { ModelMeta } from '@/types/api';

const PLACEHOLDER = '—';

interface OperatingModelsCardProps {
  models: ModelMeta[] | undefined;
  /** Models 画面自身では自己リンクを張らない */
  linkToModels?: boolean;
}

function metric(metrics: Record<string, unknown> | null, key: string, fmt: 'score' | 'ratio') {
  if (!metrics) return PLACEHOLDER;
  const v = metrics[key];
  if (typeof v !== 'number') return PLACEHOLDER;
  return fmt === 'score' ? formatScore(v) : formatRatio(v);
}

/**
 * 予想に使っている **2 つのモデル**を並べて出す。
 *
 * 役割が分かれている:
 *   - 買い目を決める (active)     … どの馬・どの組を買うか
 *   - 確からしさを出す (確率モデル) … 複勝を買うかの判定と、連系の確率
 *
 * 元は「Active モデル」1 枚のカードだったが、それだと確率モデルが動いていることが
 * どこにも出ず、「設定したのに効いているのか分からない」状態になる。片方だけの
 * 表示に戻さないこと。
 */
export function OperatingModelsCard({ models, linkToModels = true }: OperatingModelsCardProps) {
  const active = models?.find((m) => m.is_active) ?? null;
  const probability = models?.find((m) => m.is_probability_model) ?? null;

  const body = (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          予想に使っているモデル
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 md:grid-cols-2">
          <Slot
            role="買い目を決める"
            help="どの馬・どの組を買うかを決めます"
            model={active}
            emptyMessage="学習済モデルがありません。Models 画面から学習を実行してください。"
            metrics={[
              ['NDCG@3', metric(active?.metrics ?? null, 'ndcg3', 'score')],
              ['単勝回収率', metric(active?.metrics ?? null, 'payback_win', 'ratio')],
            ]}
          />
          <Slot
            role="確からしさを出す"
            help="複勝を買うかの判定と、連系の確率に使います"
            model={probability}
            emptyMessage="未設定。Models 画面で選ぶと、複勝の絞り込みと連系の確率が変わります。"
            metrics={[['NDCG@3', metric(probability?.metrics ?? null, 'ndcg3', 'score')]]}
          />
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

interface SlotProps {
  role: string;
  help: string;
  model: ModelMeta | null;
  emptyMessage: string;
  metrics: [string, string][];
}

function Slot({ role, help, model, emptyMessage, metrics }: SlotProps) {
  return (
    <div className="flex flex-col gap-2 border-t border-border pt-3 md:border-l md:border-t-0 md:pl-4 md:pt-0 md:first:border-l-0 md:first:pl-0">
      <div className="flex items-center gap-2">
        <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
          {role}
        </span>
        {model ? (
          <span className="font-mono text-xs text-muted-foreground">ID {model.id}</span>
        ) : (
          <Badge variant="outline">未設定</Badge>
        )}
      </div>
      {model ? (
        <>
          {model.name?.trim() && <div className="text-sm text-foreground">{model.name}</div>}
          <div className="grid grid-cols-2 gap-x-6 gap-y-1">
            <Meta label="作成" value={formatDateTime(model.created_at)} />
            <Meta label="学習期間" value={model.train_range ?? PLACEHOLDER} />
          </div>
          <div className="flex gap-6">
            {metrics.map(([label, value]) => (
              <Metric key={label} label={label} value={value} />
            ))}
          </div>
        </>
      ) : (
        <p className="text-sm text-muted-foreground">{emptyMessage}</p>
      )}
      <p className="text-xs text-subtle-foreground">{help}</p>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono text-xs text-muted-foreground">{value}</div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono text-lg tabular-nums text-foreground">{value}</div>
    </div>
  );
}
