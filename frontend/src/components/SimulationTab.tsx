import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Play, Loader2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table';
import { EmptyState } from '@/components/EmptyState';
import { MetricCard } from '@/components/MetricCard';
import { runSimulation, formatErrorMessageSync } from '@/lib/api';
import { formatPercent, formatRatio, formatYen } from '@/lib/formatters';
import { toast } from '@/components/ui/toast';
import type {
  SimulationGroupStats,
  SimulationResponse,
  SimulationStrategy,
} from '@/types/api';

// ── Strategy preset metadata ─────────────────────────────────────────────────

interface StrategyPreset {
  key: SimulationStrategy;
  emoji: string;
  label: string;
  description: string;
}

const STRATEGY_PRESETS: StrategyPreset[] = [
  {
    key: 'conservative',
    emoji: '🛡',
    label: '安定',
    description: '高 EV (期待値 1.30+) の案件のみ少額で。少ない bet 数。',
  },
  {
    key: 'balanced',
    emoji: '⚖',
    label: '標準',
    description: '中程度の EV (1.10+) を Kelly 1/4 で。バランス重視。',
  },
  {
    key: 'aggressive',
    emoji: '🔥',
    label: '積極的',
    description: 'positive edge ならどの combo にも賭ける。bet 数多い。',
  },
];

// ── Date helpers ──────────────────────────────────────────────────────────────

function _addMonths(d: Date, months: number): Date {
  const r = new Date(d);
  r.setMonth(r.getMonth() + months);
  return r;
}
function _isoDate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function _diffDays(start: string, end: string): number | null {
  const s = new Date(start);
  const e = new Date(end);
  if (Number.isNaN(s.getTime()) || Number.isNaN(e.getTime())) return null;
  return Math.round((e.getTime() - s.getTime()) / 86_400_000);
}

// バックエンドの MAX_WINDOW_DAYS と一致させる。
const MAX_WINDOW_DAYS = 186;

// ── Group breakdown table ─────────────────────────────────────────────────────

interface GroupTableProps {
  title: string;
  rows: SimulationGroupStats[];
}

function GroupTable({ title, rows }: GroupTableProps) {
  if (rows.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">該当するベットがありません。</p>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>ラベル</TableHead>
              <TableHead className="text-right">bet 数</TableHead>
              <TableHead className="text-right">投資</TableHead>
              <TableHead className="text-right">払戻</TableHead>
              <TableHead className="text-right">回収率</TableHead>
              <TableHead className="text-right">的中率</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={r.label}>
                <TableCell className="font-medium">{r.label}</TableCell>
                <TableCell className="text-right">{r.n_bets}</TableCell>
                <TableCell className="text-right">{formatYen(r.invested)}</TableCell>
                <TableCell className="text-right">{formatYen(r.payout)}</TableCell>
                <TableCell
                  className={`text-right ${
                    r.payback_rate >= 1
                      ? 'text-green-600 font-semibold'
                      : 'text-muted-foreground'
                  }`}
                >
                  {formatRatio(r.payback_rate)}
                </TableCell>
                <TableCell className="text-right">{formatPercent(r.hit_rate)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

// ── Main SimulationTab ────────────────────────────────────────────────────────

export function SimulationTab() {
  const today = new Date();
  const defaultEnd = _isoDate(today);
  const defaultStart = _isoDate(_addMonths(today, -3));

  const [start, setStart] = useState(defaultStart);
  const [end, setEnd] = useState(defaultEnd);
  const [budget, setBudget] = useState(100_000);
  const [strategy, setStrategy] = useState<SimulationStrategy>('balanced');
  const [result, setResult] = useState<SimulationResponse | null>(null);

  const mutation = useMutation({
    mutationFn: () =>
      runSimulation({
        start: start || undefined,
        end: end || undefined,
        budget,
        strategy,
      }),
    onSuccess: (data) => {
      setResult(data);
      toast.success(`シミュレーション完了 (${data.n_settled_races} race)`);
    },
    onError: (err) => {
      toast.error(`シミュレーション失敗: ${formatErrorMessageSync(err)}`);
    },
  });

  const windowDays = _diffDays(start, end);
  const windowTooLong = windowDays !== null && windowDays > MAX_WINDOW_DAYS;

  function handleRun() {
    if (windowTooLong) {
      toast.error(
        `期間が長すぎます (${windowDays} 日)。${MAX_WINDOW_DAYS} 日以内で指定してください。`,
      );
      return;
    }
    mutation.mutate();
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Form: window + budget + strategy */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">シミュレーション設定</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="sim-start">期間 開始日</Label>
              <Input
                id="sim-start"
                type="date"
                value={start}
                onChange={(e) => setStart(e.target.value)}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="sim-end">期間 終了日</Label>
              <Input
                id="sim-end"
                type="date"
                value={end}
                onChange={(e) => setEnd(e.target.value)}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="sim-budget">元手 / Bankroll (円)</Label>
              <Input
                id="sim-budget"
                type="number"
                min={1000}
                step={10_000}
                value={budget}
                onChange={(e) => setBudget(Math.max(1000, Number(e.target.value) || 0))}
              />
              <p className="text-xs text-muted-foreground">
                Kelly 計算の基準額。1 race ごとの stake 上限 = 元手 × 5% 。
                累計支出のキャップではなく、race 数が増えると累計 invested は増えます。
              </p>
            </div>
          </div>

          {windowTooLong && (
            <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              期間が長すぎます ({windowDays} 日)。{MAX_WINDOW_DAYS} 日以内
              (約 6 か月) で指定してください。1 年規模だと逐次予測が数分かかり
              HTTP timeout します。
            </div>
          )}

          <div className="flex flex-col gap-2">
            <Label>戦略</Label>
            <div className="flex flex-wrap gap-2">
              {STRATEGY_PRESETS.map((p) => (
                <button
                  key={p.key}
                  type="button"
                  onClick={() => setStrategy(p.key)}
                  className={`flex flex-col items-start gap-1 rounded-md border px-4 py-2 text-left transition ${
                    strategy === p.key
                      ? 'border-primary bg-primary/10 ring-1 ring-primary'
                      : 'border-border hover:bg-accent'
                  }`}
                >
                  <span className="text-sm font-medium">
                    {p.emoji} {p.label}
                  </span>
                  <span className="text-xs text-muted-foreground">{p.description}</span>
                </button>
              ))}
            </div>
          </div>

          <div>
            <Button
              onClick={handleRun}
              disabled={mutation.isPending || windowTooLong}
              className="gap-2"
            >
              {mutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  実行中...（30 〜 60 秒）
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  シミュレーション実行
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Result */}
      {mutation.isPending ? (
        <EmptyState
          message="シミュレーション実行中..."
          description="アクティブモデルで全レースを predict + recommend + settle しています。完了まで 30 〜 60 秒。"
        />
      ) : !result ? (
        <EmptyState
          message="シミュレーション未実行"
          description="期間・元手・戦略を選んで「実行」ボタンを押してください。"
        />
      ) : (
        <>
          {/* Summary KPI cards */}
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
            <MetricCard
              title="累計投資"
              value={result.summary.invested}
              format="yen"
              description={`${result.summary.n_bets} bets / ${result.n_settled_races} race`}
            />
            <MetricCard
              title="累計払戻"
              value={result.summary.payout}
              format="yen"
              description={`的中 ${formatPercent(result.summary.hit_rate)}`}
            />
            <MetricCard
              title="純利益"
              value={result.summary.payout - result.summary.invested}
              format="yen"
              description={
                result.summary.payout >= result.summary.invested
                  ? 'プラス収支'
                  : 'マイナス収支'
              }
            />
            <MetricCard
              title="回収率"
              value={result.summary.payback_rate}
              format="ratio"
              description="1.00 = 損益分岐"
            />
            <MetricCard
              title="的中率"
              value={result.summary.hit_rate}
              format="percent"
              description="bet 全体"
            />
          </div>

          {/* Group breakdown tables */}
          <GroupTable title="馬券種別" rows={result.by_bet_type} />
          <GroupTable title="レース格別" rows={result.by_race_class} />
          <GroupTable title="コース別" rows={result.by_course} />
        </>
      )}
    </div>
  );
}
