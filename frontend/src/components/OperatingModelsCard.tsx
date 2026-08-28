import { Link } from 'react-router-dom';

import { Badge } from '@/components/ui/badge';
import { formatDate, formatRatio, formatScore } from '@/lib/formatters';
import { labelClass } from '@/lib/labels';
import type { ModelMeta } from '@/types/api';

const PLACEHOLDER = '—';

interface OperatingModelsCardProps {
  models: ModelMeta[] | undefined;
  /** Models 画面自身では自己リンクを張らない */
  linkToModels?: boolean;
}

function metric(model: ModelMeta | null, key: string, fmt: 'score' | 'ratio') {
  const v = model?.metrics?.[key];
  if (typeof v !== 'number') return PLACEHOLDER;
  return fmt === 'score' ? formatScore(v) : formatRatio(v);
}

/** "2015-01-04/2024-04-28" → "2015-01-04 〜 2024-04-28" */
function trainRange(model: ModelMeta | null): string {
  const raw = model?.train_range;
  if (!raw) return PLACEHOLDER;
  const [from, to] = raw.split('/');
  return to ? `${formatDate(from)} 〜 ${formatDate(to)}` : formatDate(raw);
}

/**
 * 予想に使っている **2 つのモデル**を、役割ごとに 1 本の帯で見せる。
 *
 * 役割が分かれている:
 *   - 買い目を決める (active)     … どの馬・どの組を買うか
 *   - 確からしさを出す (確率モデル) … 複勝を買うかの判定と、連系の確率
 *
 * 情報の順序は「**何をする役か → どのモデルか → どれだけ確かか**」。役割名を
 * 一番上に置くのは、ここを読む人の問いが「いま何が動いているか」だから。
 * モデル名や ID から始めると、識別子を読んでから用途を推測することになる。
 *
 * 指標は役割ごとに変える。買い目側は回収率 (実際に得る数字)、確からしさ側は
 * 順位精度。**確率モデルに回収率を出さない**のは、それが賭けに使われていない
 * 数字だから — 並べると「0.82 の方が悪いモデル」と誤読される。
 */
export function OperatingModelsCard({ models, linkToModels = true }: OperatingModelsCardProps) {
  const active = models?.find((m) => m.is_active) ?? null;
  const probability = models?.find((m) => m.is_probability_model) ?? null;

  const body = (
    <section className="border-y border-border">
      <h3 className={labelClass('mb-0 px-4 pt-3 sm:px-6')}>予想に使っているモデル</h3>
      <div className="grid grid-cols-1 divide-y divide-border sm:grid-cols-2 sm:divide-x sm:divide-y-0">
        <Slot
          role="買い目を決める"
          purpose="どの馬・どの組を買うかを決めます"
          model={active}
          empty="学習済モデルがありません"
          emptyHint="Models 画面から学習を実行してください"
          metrics={[
            ['単勝回収率', metric(active, 'payback_win', 'ratio')],
            ['NDCG@3', metric(active, 'ndcg3', 'score')],
          ]}
        />
        <Slot
          role="確からしさを出す"
          purpose="複勝を買うかの判定と、連系の確率に使います"
          model={probability}
          empty="未設定"
          emptyHint="Models 画面で選ぶと、複勝の絞り込みと連系の確率が変わります"
          metrics={[['NDCG@3', metric(probability, 'ndcg3', 'score')]]}
        />
      </div>
    </section>
  );

  if (!linkToModels) return body;
  return (
    <Link to="/models" className="block transition-colors hover:bg-muted/30">
      {body}
    </Link>
  );
}

interface SlotProps {
  role: string;
  purpose: string;
  model: ModelMeta | null;
  empty: string;
  emptyHint: string;
  metrics: [string, string][];
}

function Slot({ role, purpose, model, empty, emptyHint, metrics }: SlotProps) {
  return (
    <div className="flex flex-col gap-3 px-4 py-4 sm:px-6">
      {/* 1. 何をする役か */}
      <div className="flex items-baseline gap-2">
        <span className="text-sm font-medium text-foreground">{role}</span>
        <span className="text-xs text-subtle-foreground">{purpose}</span>
      </div>

      {/* 2. どのモデルか */}
      {model ? (
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="truncate text-sm text-foreground">
            {model.name?.trim() || `モデル ${model.id}`}
          </span>
          <span className="font-mono text-xs text-muted-foreground">ID {model.id}</span>
          <span className="font-mono text-xs text-subtle-foreground">
            学習 {trainRange(model)}
          </span>
        </div>
      ) : (
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <Badge variant="outline">{empty}</Badge>
          <span className="text-xs text-subtle-foreground">{emptyHint}</span>
        </div>
      )}

      {/* 3. どれだけ確かか */}
      {model && (
        <dl className="flex gap-8">
          {metrics.map(([label, value]) => (
            <div key={label}>
              <dt className="text-xs text-muted-foreground">{label}</dt>
              <dd className="font-mono text-xl tabular-nums text-foreground">{value}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
