import { useEffect, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import { Link } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Play, Loader2, Archive, Trash2, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table';
import { ProfitChart } from '@/components/ProfitChart';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
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
import { formatPercent, formatRatio, formatSignedYen, formatYen } from '@/lib/formatters';
import { toast } from '@/lib/toast';
import { ALL_BET_TYPES } from '@/lib/betTypes';
import type {
  BetType,
  SimulationGroupStats,
  SimulationResponse,
  SimulationRunSummary,
} from '@/types/api';

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
  /** 1 列目の見出し。タブごとに何で切った表かが変わるので必ず渡す。 */
  label: string;
  rows: SimulationGroupStats[];
}

function GroupTable({ rows, label }: GroupTableProps) {
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">該当するベットがありません。</p>;
  }
  // **投資の多い順に並べる。** 回収率だけ見ると 10 点で 0.5 の行と 5,000 点で
  // 0.9 の行が同じ重さに見えるが、損益への効き方はまるで違う。
  const sorted = [...rows].sort((a, b) => b.invested - a.invested);
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>{label}</TableHead>
          <TableHead className="text-right">点数</TableHead>
          <TableHead className="text-right">投資</TableHead>
          <TableHead className="text-right">払戻</TableHead>
          <TableHead className="text-right">収支</TableHead>
          <TableHead className="text-right">回収率</TableHead>
          <TableHead className="text-right">的中率</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {sorted.map((r) => {
          const profit = r.payout - r.invested;
          return (
            <TableRow key={r.label}>
              <TableCell className="font-medium">{r.label}</TableCell>
              <TableCell className="text-right tabular-nums">
                {r.n_bets.toLocaleString()}
              </TableCell>
              <TableCell className="text-right tabular-nums">{formatYen(r.invested)}</TableCell>
              <TableCell className="text-right tabular-nums">{formatYen(r.payout)}</TableCell>
              <TableCell
                className={`text-right tabular-nums ${
                  profit >= 0 ? 'text-success' : 'text-destructive'
                }`}
              >
                {formatSignedYen(profit)}
              </TableCell>
              <TableCell
                className={`text-right tabular-nums ${
                  r.payback_rate >= 1 ? 'text-success' : 'text-muted-foreground'
                }`}
              >
                {formatRatio(r.payback_rate)}
              </TableCell>
              <TableCell className="text-right tabular-nums text-muted-foreground">
                {formatPercent(r.hit_rate)}
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

// ── SavedRunsPanel: 保存済みシミュレーション一覧 ─────────────────────────────

interface SavedRunsPanelProps {
  /** このモデル (model_runs.id) の実行のみ表示する。 */
  modelId: number;
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

/**
 * 保存済みの実行を並べる。設定を変えて回し直しても後から見分けられるよう、
 * 実行時の条件 (conditions_json) を結果の見出しに一緒に出す。
 */
function SavedRunsPanel({ modelId, activeRunId, onLoad, onDeleted }: SavedRunsPanelProps) {
  const queryClient = useQueryClient();
  const listQuery = useQuery({
    queryKey: ['simulation-runs', modelId],
    queryFn: () => listSimulationRuns(modelId),
    staleTime: 0,
  });

  const deleteMutation = useMutation({
    mutationFn: (runId: number) => deleteSimulationRun(runId),
    onSuccess: () => {
      toast.success('保存済み実行を削除しました');
      queryClient.invalidateQueries({ queryKey: ['simulation-runs', modelId] });
      onDeleted();
    },
    onError: (err) => {
      toast.error(`削除失敗: ${formatErrorMessageSync(err)}`);
    },
  });

  function handleDelete(e: MouseEvent, runId: number) {
    e.stopPropagation();
    if (!window.confirm('この実行結果を削除しますか?')) return;
    deleteMutation.mutate(runId);
  }

  const runs = listQuery.data?.runs ?? [];

  return (
    <Card className="border-t border-border pt-6">
      <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
        <CardTitle className="flex items-center gap-2 text-label-ja">
          <Archive className="h-4 w-4" />
          過去の実行 ({runs.length})
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
            まだありません。実行すると自動で保存されます（上限 50 件）。
          </p>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>実行日時</TableHead>
                  <TableHead>期間</TableHead>
                  <TableHead className="text-right">1 レースの上限</TableHead>
                  <TableHead className="text-right">損益</TableHead>
                  <TableHead className="text-right">最大益</TableHead>
                  <TableHead className="text-right">races</TableHead>
                  <TableHead></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {runs.map((r: SimulationRunSummary) => {
                  const isActive = r.id === activeRunId;
                  const profit = r.final_profit;
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
                      <TableCell className="text-right">
                        {formatYen(r.race_budget)}
                      </TableCell>
                      <TableCell
                        className={`text-right ${
                          profit > 0
                            ? 'font-medium text-success'
                            : profit < 0
                            ? 'text-destructive'
                            : ''
                        }`}
                      >
                        {formatSignedYen(r.final_profit)}
                      </TableCell>
                      <TableCell className="text-right">
                        {formatSignedYen(r.peak_profit)}
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


// ── ModelSimulationPanel: 特定モデルのバックテスト ──────────────────────────

interface ModelSimulationPanelProps {
  /** バックテスト対象モデル (model_runs.id)。 */
  modelId: number;
}

/**
 * モデルを過去のレースに当てて損益を見るパネル。
 *
 * RACE 画面の推奨買目とまったく同じ仕組みで回す。入力は 1 レースに使う上限だけで、
 * 初期資産も賭け金の決め方も持たない — 賭け金が残高に依存しないので破産が起きず、
 * 評価が途中で止まらない。結果は資産残高ではなく 0 から始まる累計損益で出す。
 */
export function ModelSimulationPanel({ modelId }: ModelSimulationPanelProps) {
  const today = new Date();
  const defaultEnd = _isoDate(today);
  const defaultStart = _isoDate(_addMonths(today, -3));
  const queryClient = useQueryClient();

  const [start, setStart] = useState(defaultStart);
  const [end, setEnd] = useState(defaultEnd);
  const [betTypes, setBetTypes] = useState<BetType[] | null>(null);
  // 1 レースに使ってよい上限 (円)。**使い切る目標ではない。**
  // RACE 画面と同じ意味で、実際に賭ける額は確信度の下限が決める。
  // 既定は Settings の値に合わせる (未取得のうちは 5,000 円)。
  const [raceBudget, setRaceBudget] = useState(5_000);
  // Settings は実行時にバックエンドが読む。ここでは「何が引き継がれるか」の表示用。
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
          queryClient.invalidateQueries({ queryKey: ['simulation-runs', modelId] });
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
        race_budget: raceBudget,
        ...(betTypes ? { bet_types: betTypes } : {}),
        model_id: modelId,
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
      {/* ── 1. 条件 ─────────────────────────────────────────────
          入力は 3 つだけ (期間 / 1 レースの上限 / 買う馬券)。縦積みだと
          「まだ何か入力欄があるのでは」と読ませてしまうので横に並べる。 */}
      <Card className="border-t border-border pt-6">
        <CardHeader>
          <CardTitle className="text-label-ja">条件</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <div className="flex max-w-sm flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <Label>期間 開始日</Label>
              <DateYMDPicker value={start} onChange={setStart} ariaLabel="開始日" />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>期間 終了日</Label>
              <DateYMDPicker value={end} onChange={setEnd} ariaLabel="終了日" />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="sim-race-budget">1 レースに使う上限 (円)</Label>
              <Input
                id="sim-race-budget"
                type="number"
                min={100}
                step={500}
                className="text-right font-mono tabular-nums"
                value={raceBudget}
                onChange={(e) => setRaceBudget(Math.max(100, Number(e.target.value) || 0))}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>買う馬券</Label>
              <div className="flex flex-wrap items-center gap-1.5">
                {ALL_BET_TYPES.map((betType) => {
                  const current = betTypes ?? (ALL_BET_TYPES as readonly BetType[] as BetType[]);
                  const on = current.includes(betType);
                  return (
                    <button
                      key={betType}
                      type="button"
                      aria-pressed={on}
                      onClick={() => {
                        const next = on
                          ? current.filter((b) => b !== betType)
                          : [...current, betType];
                        if (next.length === 0) return; // 全部外すと候補が空になる
                        setBetTypes(next);
                      }}
                      className={`rounded-sm border px-2 py-1 text-xs transition-colors ${
                        on
                          ? 'border-primary bg-primary/10 text-primary'
                          : 'border-border bg-transparent text-subtle-foreground hover:border-border-strong hover:text-foreground'
                      }`}
                    >
                      {betType}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-4">
            {/* このシミュレーションは **2 つのモデル**を使う。片方 (確率モデル) は
                Settings / Models 画面で決まり、実行時に読まれる。ここに出さないと
                「同じボタンを押したのに前回と結果が違う」理由が分からない。 */}
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
              <span className="text-muted-foreground">買い目を決めるモデル</span>
              <span className="text-foreground">この画面のモデル</span>
              {/* 買い方 (複勝の下限・連系の下限) は Settings の値を実行時に読む。
                  説明を並べる代わりに、変えに行ける導線だけ置く。 */}
              <Link
                to="/settings"
                className="ml-auto text-primary underline-offset-2 hover:underline"
              >
                買い方の設定 →
              </Link>
            </div>

          </div>

          {windowTooLong && (
            <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              期間が長すぎます ({windowDays} 日)。{MAX_WINDOW_DAYS} 日以内
              (約 1 年) で指定してください。
            </div>
          )}

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
              ? 'このモデルで全レースを predict + recommend + settle しています。完了まで window のサイズ次第で数十秒〜数分。画面を離れてもバックエンドで継続実行されます。'
              : ''
          }
        />
      ) : !result ? (
        <EmptyState
          message="シミュレーション未実行"
          description="期間と 1 レースに使う上限を決めて「実行」ボタンを押してください。または保存済みの実行をクリックしてロードできます。"
        />
      ) : (
        <>

          {/* ── 2. 結果のまとめ ────────────────────────────────
              収支は 0 スタート。元手が無いので「増えたか減ったか」だけを出す。 */}
          <Card className="flex flex-col gap-3 border-t border-border pt-6">
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <h3 className="text-label-ja">結果</h3>
              {/* **どの条件で走ったかを結果の見出しに畳む。** 設定を変えれば同じ
                  ボタンでも別条件で走るので、結果と離すと後から比べられない。 */}
              <p className="text-xs text-subtle-foreground">
                {result.window.start ?? '—'} 〜 {result.window.end ?? '—'} ・{' '}
                {result.n_settled_races.toLocaleString()} レース
                {result.conditions && (
                  <>
                    {' ・ '}1 レース {(result.conditions.race_budget ?? 0).toLocaleString()} 円まで
                    {' ・ '}
                    {result.conditions.enabled_bet_types.join(' / ') || '—'}
                    {result.conditions.probability_model
                      ? ` ・ 確率モデル ${result.conditions.probability_model}`
                      : ' ・ 確率モデル未使用'}
                  </>
                )}
              </p>
            </div>
            {/* **カードにするのは収支だけ。** これがこの画面の答えで、
                残りは読み解くための補助 (全部囲うと答えが埋もれる)。 */}
            <div className="flex flex-wrap items-start gap-4">
              <MetricCard
                label="累計損益"
                value={formatSignedYen(result.final_profit)}
                tone={result.final_profit >= 0 ? 'positive' : 'negative'}
                note="0 から始めた場合の収支"
                className="min-w-[11rem]"
              />
              <dl className="flex flex-wrap gap-x-8 gap-y-3 pt-1">
                <div>
                  <dt className="text-xs text-muted-foreground">最大益 / 最大損</dt>
                  <dd className="font-mono text-lg tabular-nums">
                    {formatSignedYen(result.peak_profit)}
                    <span className="mx-1 text-subtle-foreground">/</span>
                    {formatSignedYen(result.trough_profit)}
                  </dd>
                </div>
                <div title="途中で止まらずに回すのに要した額 (= 最大損の絶対値)">
                  <dt className="text-xs text-muted-foreground">必要だった資金</dt>
                  <dd className="font-mono text-lg tabular-nums">
                    {formatYen(result.required_capital)}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-muted-foreground">回収率</dt>
                  <dd className="font-mono text-lg tabular-nums">
                    {formatRatio(result.summary.payback_rate)}
                  </dd>
                </div>
              </dl>
            </div>
          </Card>

          {/* ── 3. 損益推移 ───────────────────────────────────── */}
          <Card className="border-t border-border pt-6">
            <CardHeader>
              <CardTitle className="text-label-ja">損益推移</CardTitle>
            </CardHeader>
            <CardContent>
              <ProfitChart points={result.profit_timeseries} />
            </CardContent>
          </Card>

          {/* ── 4. 内訳 ───────────────────────────────────────
              以前は「bet 単位の統計 5 枚」と「内訳の表」が別のパネルだった。
              5 枚のうち回収率と純利益は上の「結果」と同じ数字で、同じ値が画面に
              2 度出ていた。**残りの 4 つ (投資・払戻・的中率・点数) は表の合計**
              なので、表の見出し行として 1 つのカードに畳む。 */}
          <Card className="border-t border-border pt-6">
            <CardHeader className="flex flex-row flex-wrap items-baseline justify-between gap-x-6 gap-y-1 space-y-0">
              <CardTitle className="text-label-ja">内訳</CardTitle>
              <p className="font-mono text-xs tabular-nums text-subtle-foreground">
                {formatYen(result.summary.invested)}
                <span className="mx-1.5 font-sans">→</span>
                {formatYen(result.summary.payout)}
                <span className="mx-2 font-sans text-border">|</span>
                <span className="font-sans">的中 </span>
                {formatPercent(result.summary.hit_rate)}
                <span className="mx-2 font-sans text-border">|</span>
                {result.summary.n_bets.toLocaleString()}
                <span className="font-sans"> 点 / </span>
                {result.n_settled_races.toLocaleString()}
                <span className="font-sans"> レース</span>
              </p>
            </CardHeader>
            <CardContent>
              {/* 3 つの表を積むと縦に伸びるだけで見比べられない。同じ形の表なので
                  タブで切り替える。 */}
              <Tabs defaultValue="bet_type">
                <TabsList>
                  <TabsTrigger value="bet_type">馬券種別</TabsTrigger>
                  <TabsTrigger value="race_class">レース格別</TabsTrigger>
                  <TabsTrigger value="course">コース別</TabsTrigger>
                </TabsList>
                <TabsContent value="bet_type" className="pt-3">
                  <GroupTable rows={result.by_bet_type} label="券種" />
                </TabsContent>
                <TabsContent value="race_class" className="pt-3">
                  <GroupTable rows={result.by_race_class} label="レース格" />
                </TabsContent>
                <TabsContent value="course" className="pt-3">
                  <GroupTable rows={result.by_course} label="コース" />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

        </>
      )}

      {/* ── 6. 過去の実行 ─────────────────────────────────────
          入力と結果の間に挟むと流れが切れるので最後に置く。 */}
      <SavedRunsPanel
        modelId={modelId}
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
    </div>
  );
}
