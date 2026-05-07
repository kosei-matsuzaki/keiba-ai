import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Play, Loader2, Archive, Trash2, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table';
import { BankrollChart } from '@/components/BankrollChart';
import { DateYMDPicker } from '@/components/DateYMDPicker';
import { EmptyState } from '@/components/EmptyState';
import { MetricCard } from '@/components/MetricCard';
import {
  deleteSimulationRun,
  fetchJob,
  formatErrorMessageSync,
  getSimulationRun,
  listSimulationRuns,
  startSimulationJob,
} from '@/lib/api';
import { formatPercent, formatRatio, formatYen } from '@/lib/formatters';
import { toast } from '@/components/ui/toast';
import type {
  SimulationGroupStats,
  SimulationResponse,
  SimulationRunSummary,
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

// バックエンドの MAX_BG_WINDOW_DAYS と一致させる (background job で 1 年まで OK)。
const MAX_WINDOW_DAYS = 366;

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

// ── SavedRunsPanel: 保存済みシミュレーション一覧 ─────────────────────────────

interface SavedRunsPanelProps {
  /** 表示中の result の run_id (highlight 用)。 */
  activeRunId: number | null;
  /** click した run の詳細を読み込む (親で setResult)。 */
  onLoad: (runId: number) => void;
  /** 削除後に list を refetch するキック。 */
  onDeleted: () => void;
}

function _formatRunTimestamp(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  // YYYY-MM-DD HH:mm
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const mi = String(d.getMinutes()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}`;
}

function SavedRunsPanel({ activeRunId, onLoad, onDeleted }: SavedRunsPanelProps) {
  const queryClient = useQueryClient();
  const listQuery = useQuery({
    queryKey: ['simulation-runs'],
    queryFn: listSimulationRuns,
    staleTime: 0,
  });

  const deleteMutation = useMutation({
    mutationFn: (runId: number) => deleteSimulationRun(runId),
    onSuccess: () => {
      toast.success('保存済み実行を削除しました');
      queryClient.invalidateQueries({ queryKey: ['simulation-runs'] });
      onDeleted();
    },
    onError: (err) => {
      toast.error(`削除失敗: ${formatErrorMessageSync(err)}`);
    },
  });

  function handleDelete(e: React.MouseEvent, runId: number) {
    e.stopPropagation();
    if (!window.confirm('この実行結果を削除しますか?')) return;
    deleteMutation.mutate(runId);
  }

  const runs = listQuery.data?.runs ?? [];

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
        <CardTitle className="flex items-center gap-2 text-base">
          <Archive className="h-4 w-4" />
          保存済みシミュレーション ({runs.length})
        </CardTitle>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => listQuery.refetch()}
          disabled={listQuery.isFetching}
          className="gap-1.5"
        >
          <RefreshCw
            className={`h-3.5 w-3.5 ${listQuery.isFetching ? 'animate-spin' : ''}`}
          />
          更新
        </Button>
      </CardHeader>
      <CardContent>
        {listQuery.isPending ? (
          <p className="text-sm text-muted-foreground">読込中…</p>
        ) : runs.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            保存済みの実行はありません。シミュレーションを実行すると自動的にここに保存されます (上限 50 件)。
          </p>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>実行日時</TableHead>
                  <TableHead>期間</TableHead>
                  <TableHead>戦略</TableHead>
                  <TableHead className="text-right">初期</TableHead>
                  <TableHead className="text-right">最終</TableHead>
                  <TableHead className="text-right">ピーク</TableHead>
                  <TableHead className="text-right">races</TableHead>
                  <TableHead></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {runs.map((r: SimulationRunSummary) => {
                  const isActive = r.id === activeRunId;
                  const profit = r.final_bankroll - r.budget;
                  return (
                    <TableRow
                      key={r.id}
                      className={`cursor-pointer ${
                        isActive ? 'bg-primary/10' : ''
                      }`}
                      onClick={() => onLoad(r.id)}
                    >
                      <TableCell className="font-medium">
                        {_formatRunTimestamp(r.created_at)}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {r.window_start ?? '-'} 〜 {r.window_end ?? '-'}
                      </TableCell>
                      <TableCell>{r.strategy}</TableCell>
                      <TableCell className="text-right">
                        {formatYen(r.budget)}
                      </TableCell>
                      <TableCell
                        className={`text-right ${
                          profit > 0
                            ? 'text-green-600 font-semibold'
                            : profit < 0
                            ? 'text-red-600'
                            : ''
                        }`}
                      >
                        {formatYen(r.final_bankroll)}
                      </TableCell>
                      <TableCell className="text-right">
                        {formatYen(r.peak_bankroll)}
                      </TableCell>
                      <TableCell className="text-right text-xs text-muted-foreground">
                        {r.n_settled_races} / {r.n_races}
                      </TableCell>
                      <TableCell>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={(e) => handleDelete(e, r.id)}
                          disabled={deleteMutation.isPending}
                          aria-label="削除"
                        >
                          <Trash2 className="h-3.5 w-3.5 text-muted-foreground" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}


// ── Main SimulationTab ────────────────────────────────────────────────────────

export function SimulationTab() {
  const today = new Date();
  const defaultEnd = _isoDate(today);
  const defaultStart = _isoDate(_addMonths(today, -3));
  const queryClient = useQueryClient();

  const [start, setStart] = useState(defaultStart);
  const [end, setEnd] = useState(defaultEnd);
  const [budget, setBudget] = useState(100_000);
  const [strategy, setStrategy] = useState<SimulationStrategy>('balanced');
  // 1 race 絶対上限 (円)。0 で無効。default 5000 円 (= 100k 元手の 5%)。
  const [maxStakePerRaceYen, setMaxStakePerRaceYen] = useState(5_000);
  const [result, setResult] = useState<SimulationResponse | null>(null);

  // ── Background job orchestration ────────────────────────────────────
  // 走行中の job_id と経過秒数。job_id がセットされている間 GET /jobs/{id}
  // をポーリングし、完了したら getSimulationRun(run_id) で結果を取得する。
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [elapsedSec, setElapsedSec] = useState(0);
  const startedAtRef = useRef<number | null>(null);

  // 経過秒タイマー (job 走行中のみ)
  useEffect(() => {
    if (activeJobId === null) {
      setElapsedSec(0);
      startedAtRef.current = null;
      return;
    }
    startedAtRef.current = Date.now();
    setElapsedSec(0);
    const id = window.setInterval(() => {
      if (startedAtRef.current !== null) {
        setElapsedSec(Math.floor((Date.now() - startedAtRef.current) / 1000));
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [activeJobId]);

  // job ポーリング (2 秒間隔)
  const jobQuery = useQuery({
    queryKey: ['simulation-job', activeJobId],
    queryFn: () => fetchJob(activeJobId!),
    enabled: activeJobId !== null,
    refetchInterval: (query) => {
      const data = query.state.data;
      // running の間だけ 2 秒間隔で polling、それ以外は止める
      if (!data) return 2000;
      const isDone = data.status !== 'running' && data.status !== 'pending';
      return isDone ? false : 2000;
    },
    staleTime: 0,
  });

  // job 完了の監視: status が completed/failed になったら処理
  useEffect(() => {
    if (!activeJobId || !jobQuery.data) return;
    const job = jobQuery.data;
    if (job.status === 'running' || job.status === 'pending') return;

    if (job.status === 'completed') {
      const runId = job.result?.run_id as number | undefined;
      if (typeof runId === 'number') {
        getSimulationRun(runId).then((data) => {
          setResult(data);
          toast.success(`シミュレーション完了 (${data.n_settled_races} race) — 保存しました`);
          queryClient.invalidateQueries({ queryKey: ['simulation-runs'] });
        }).catch((err) => {
          toast.error(`結果取得失敗: ${formatErrorMessageSync(err)}`);
        });
      } else {
        toast.error('完了したが run_id が取得できませんでした');
      }
    } else if (job.status === 'failed') {
      toast.error(`シミュレーション失敗: ${job.error ?? '不明なエラー'}`);
    }
    // どちらの場合も polling を止める
    setActiveJobId(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobQuery.data?.status, activeJobId]);

  // 起動 mutation: job_id を返したら state にセットしてポーリング開始
  const startMutation = useMutation({
    mutationFn: () =>
      startSimulationJob({
        start: start || undefined,
        end: end || undefined,
        budget,
        strategy,
        max_stake_per_race_yen: maxStakePerRaceYen,
      }),
    onSuccess: (data) => {
      setActiveJobId(data.job_id);
      toast.success('シミュレーションをバックグラウンドで開始しました');
    },
    onError: (err) => {
      toast.error(`起動失敗: ${formatErrorMessageSync(err)}`);
    },
  });

  const isRunning = activeJobId !== null || startMutation.isPending;

  const loadMutation = useMutation({
    mutationFn: (runId: number) => getSimulationRun(runId),
    onSuccess: (data) => {
      setResult(data);
      toast.success('保存済み実行をロードしました');
    },
    onError: (err) => {
      toast.error(`ロード失敗: ${formatErrorMessageSync(err)}`);
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
    startMutation.mutate();
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
              <Label>期間 開始日</Label>
              <DateYMDPicker
                value={start}
                onChange={setStart}
                ariaLabel="開始日"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>期間 終了日</Label>
              <DateYMDPicker
                value={end}
                onChange={setEnd}
                ariaLabel="終了日"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="sim-budget">予算 / Budget (円)</Label>
              <Input
                id="sim-budget"
                type="number"
                min={1000}
                step={10_000}
                value={budget}
                onChange={(e) => setBudget(Math.max(1000, Number(e.target.value) || 0))}
              />
              <p className="text-xs text-muted-foreground">
                初期資産 (Kelly 戦略の元手)。回収分は次レースの bet 余力に加算され、
                自信のあるレース (高 EV) ほど Kelly が大きく賭けます。
                資産が尽きたら以降は実質 bet しません (破産)。
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="sim-max-stake">1 race の投資額上限 (円)</Label>
              <Input
                id="sim-max-stake"
                type="number"
                min={0}
                step={500}
                value={maxStakePerRaceYen}
                onChange={(e) =>
                  setMaxStakePerRaceYen(Math.max(0, Number(e.target.value) || 0))
                }
              />
              <p className="text-xs text-muted-foreground">
                1 race の累計 stake の絶対上限。compounding wealth で資産が
                増えても、各 race の bet 額がインフレせずこの値で頭打ちになります。
                <strong>0 で無効</strong> (元手の 5% cap のみ)。
              </p>
            </div>
          </div>

          {windowTooLong && (
            <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              期間が長すぎます ({windowDays} 日)。{MAX_WINDOW_DAYS} 日以内
              (約 1 年) で指定してください。
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
                  className={`flex flex-col items-start gap-1 rounded-lg border px-4 py-3 text-left transition-all active:scale-[0.99] ${
                    strategy === p.key
                      ? 'border-primary bg-primary/15 text-foreground ring-1 ring-primary/40'
                      : 'border-border bg-card text-muted-foreground hover:border-border-strong hover:bg-card-elevated/40 hover:text-foreground'
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
              disabled={isRunning || windowTooLong}
              className="gap-2"
            >
              {isRunning ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  実行中... ({elapsedSec} 秒)
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

      {/* Saved runs */}
      <SavedRunsPanel
        activeRunId={result?.run_id ?? null}
        onLoad={(runId) => loadMutation.mutate(runId)}
        onDeleted={() => {
          // 表示中の run が消えたら result を空に
          if (result?.run_id) {
            // 簡易: 削除完了時点では active run id が残る可能性があるので
            // user 確認のため残す方針 (削除されても result はそのまま)
          }
        }}
      />

      {/* Result */}
      {isRunning || loadMutation.isPending ? (
        <EmptyState
          message={
            isRunning
              ? `シミュレーション実行中... (${elapsedSec} 秒経過)`
              : '保存済み実行をロード中...'
          }
          description={
            isRunning
              ? 'アクティブモデルで全レースを predict + recommend + settle しています。完了まで window のサイズ次第で数十秒〜数分。タブを閉じてもバックエンドで継続実行されます。'
              : ''
          }
        />
      ) : !result ? (
        <EmptyState
          message="シミュレーション未実行"
          description="期間・予算・戦略を選んで「実行」ボタンを押してください。または保存済みの実行をクリックしてロードできます。"
        />
      ) : (
        <>
          {/* Bankroll KPI cards: 資産の絶対値 */}
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <MetricCard
              title="初期資産"
              value={result.budget}
              format="yen"
              description="シミュレーション開始時"
            />
            <MetricCard
              title="最終資産"
              value={result.final_bankroll}
              format="yen"
              tone={
                result.final_bankroll === 0
                  ? 'negative'
                  : result.final_bankroll >= result.budget
                  ? 'positive'
                  : 'negative'
              }
              description={
                result.final_bankroll === 0
                  ? '破産'
                  : result.final_bankroll >= result.budget
                  ? `+${formatYen(result.final_bankroll - result.budget)}`
                  : `−${formatYen(result.budget - result.final_bankroll)}`
              }
            />
            <MetricCard
              title="ピーク資産"
              value={result.peak_bankroll}
              format="yen"
              description="期間中の最高値"
            />
            <MetricCard
              title="資産変化率"
              value={
                result.budget > 0 ? result.final_bankroll / result.budget : 0
              }
              format="ratio"
              description="1.00 = 損益なし"
            />
          </div>

          {/* Bankroll timeseries chart */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">資産推移</CardTitle>
            </CardHeader>
            <CardContent>
              <BankrollChart
                points={result.bankroll_timeseries}
                initialBudget={result.budget}
              />
            </CardContent>
          </Card>

          {/* Bet stats KPI cards: bet 単位の統計 */}
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
              tone={
                result.summary.payout >= result.summary.invested
                  ? 'positive'
                  : 'negative'
              }
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
