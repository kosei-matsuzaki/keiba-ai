import { useNavigate } from 'react-router-dom';

import { Badge } from '@/components/ui/badge';
import { RowActionsMenu } from '@/components/RowActionsMenu';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { formatDateTime, formatRatio, formatScore } from '@/lib/formatters';
import {
  inSampleWarning,
  readModelMeta,
  sourceDescription,
  sourceLabel,
} from '@/lib/modelMetrics';
import type { ModelMeta } from '@/types/api';

interface ModelTableProps {
  models: ModelMeta[];
  onActivate: (id: number) => void;
  /** 実運用の賭けルールで測り直す (backtest --persist)。「未算出」を埋める手段。 */
  onEvaluate: (model: ModelMeta) => void;
  /** 確率モデルに設定 / 解除する。役割の割り当ては Settings ではなくここで行う */
  onSetProbability: (model: ModelMeta | null) => void;
  onEdit: (model: ModelMeta) => void;
  onDelete: (model: ModelMeta) => void;
  activatingId: number | null;
  settingProbability: boolean;
  /** 評価ジョブを投入中の ID。二重投入を防ぐ。 */
  evaluatingId: number | null;
}

const PLACEHOLDER = '—';

function ratio(value: number | null): string {
  return value != null ? formatRatio(value) : PLACEHOLDER;
}

function score(value: number | null): string {
  return value != null ? formatScore(value) : PLACEHOLDER;
}

export function ModelTable({
  models,
  onActivate,
  onEvaluate,
  onSetProbability,
  onEdit,
  onDelete,
  activatingId,
  settingProbability,
  evaluatingId,
}: ModelTableProps) {
  const navigate = useNavigate();
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>ID</TableHead>
          <TableHead>名称</TableHead>
          <TableHead>学習期間</TableHead>
          {/* 評価窓を出さないと、違う期間で測った回収率が同じ列に並んで比較できてしまう */}
          <TableHead>評価窓</TableHead>
          <TableHead className="text-right">単勝回収率</TableHead>
          <TableHead className="text-right">複勝回収率</TableHead>
          <TableHead
            className="text-right"
            title="本命の二値 log-loss。市場 (1/オッズ) より小さければ市場より正確"
          >
            log-loss
          </TableHead>
          <TableHead className="text-right" title="上位3頭の並びの正確さ。回収率とは別の量">
            順位精度
          </TableHead>
          <TableHead className="text-center">状態</TableHead>
          <TableHead className="text-right">操作</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {models.map((model) => {
          const m = readModelMeta(model);
          return (
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
              <div className="text-xs text-subtle-foreground">
                {formatDateTime(model.created_at)}
              </div>
            </TableCell>
            <TableCell className="text-xs">{model.train_range ?? PLACEHOLDER}</TableCell>
            <TableCell className="text-xs">
              <div>{m.evalRange ?? model.valid_range ?? PLACEHOLDER}</div>
              {/* 出所が違えば同じ列でも別の量。学習時の値は実運用のルールではない */}
              <div className="text-subtle-foreground" title={sourceDescription(m.source)}>
                {sourceLabel(m.source)}
                {m.nRaces != null && ` · ${m.nRaces.toLocaleString()} レース`}
              </div>
              {/* 学習期間と重なった窓の値は out-of-sample ではない。並べると横比較が崩れる */}
              {m.inSample && (
                <div className="text-destructive" title={inSampleWarning(m) ?? undefined}>
                  学習期間を含む
                </div>
              )}
            </TableCell>
            <TableCell className="text-right tabular-nums">{ratio(m.paybackWin)}</TableCell>
            <TableCell className="text-right tabular-nums">{ratio(m.paybackPlace)}</TableCell>
            <TableCell className="text-right tabular-nums">
              <span
                className={
                  m.logLoss != null && m.marketLogLoss != null && m.logLoss < m.marketLogLoss
                    ? 'text-success'
                    : undefined
                }
              >
                {score(m.logLoss)}
              </span>
              {m.marketLogLoss != null && (
                <div className="text-xs text-subtle-foreground">市場 {score(m.marketLogLoss)}</div>
              )}
            </TableCell>
            <TableCell className="text-right tabular-nums text-muted-foreground">
              {score(m.ndcg3)}
            </TableCell>
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
              {/* **操作は三点リーダーに畳む。** ボタンを 5 つ並べると、行の主役
                  である数字より操作の方が目立ってしまう。 */}
              <RowActionsMenu
                label={`モデル ${model.id} の操作`}
                actions={[
                  {
                    label: activatingId === model.id ? '切り替え中…' : 'Activate',
                    title: 'このモデルで買い目を決めるようにする',
                    disabled: model.is_active || activatingId !== null,
                    onSelect: () => onActivate(model.id),
                  },
                  {
                    // 学習時の指標は実運用の賭けルールと別物なので、ここで測り直せる
                    // ようにする。log-loss は学習側に存在せず、これでしか埋まらない。
                    label: evaluatingId === model.id ? '計測中…' : '計測',
                    title:
                      m.source === 'backtest'
                        ? '実運用の賭けルールで測り直します (5,000 レースで 10 分前後)'
                        : '実運用の賭けルールで測ります。log-loss と評価窓もここで埋まります',
                    disabled: evaluatingId !== null,
                    onSelect: () => onEvaluate(model),
                  },
                  {
                    // 役割の割り当ては Settings ではなくこの画面で行う。
                    // モデルを見比べている場所で選べないと意味がないため。
                    label: model.is_probability_model ? '確率を解除' : '確率に設定',
                    title: model.is_probability_model
                      ? '確率モデルの割り当てを解除します'
                      : '複勝の確信度と連系の確率にこのモデルを使います',
                    disabled: settingProbability,
                    onSelect: () =>
                      onSetProbability(model.is_probability_model ? null : model),
                  },
                  { label: '名称を編集', onSelect: () => onEdit(model) },
                  {
                    label: '削除',
                    tone: 'destructive',
                    title: model.is_active ? 'Active モデルは削除できません' : undefined,
                    disabled: model.is_active,
                    onSelect: () => onDelete(model),
                  },
                ]}
              />
            </TableCell>
          </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
