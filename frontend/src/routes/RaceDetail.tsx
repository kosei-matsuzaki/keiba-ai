import { useEffect, useMemo, useState } from 'react';
import { useParams, useSearchParams, Link } from 'react-router-dom';
import {
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  ChevronDown,
  ChevronsUpDown,
  Sparkles,
  RefreshCw,
  AlertTriangle,
} from 'lucide-react';

import { useRaceDetail } from '@/hooks/useRaceDetail';
import { usePredictions } from '@/hooks/usePredictions';
import { useRecommendations } from '@/hooks/useRecommendations';
import { useRunShutuba } from '@/hooks/useRunShutuba';
import { useSettings } from '@/hooks/useSettings';
import { RecommendationsCard } from '@/components/RecommendationsCard';
import type { RecommendationOverrides } from '@/components/RecommendationParamsBar';
import { EmptyState } from '@/components/EmptyState';
import { Umaban } from '@/components/Umaban';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { isNotFoundError, isServiceUnavailableError, formatErrorMessage } from '@/lib/api';
import { formatOdds, formatPercent, formatRatio, formatScore, formatYen } from '@/lib/formatters';
import { labelClass } from '@/lib/labels';
import { toast } from '@/components/ui/toast';
import type { EntrySummary, HorsePrediction, RaceInfoCoverage } from '@/types/api';

/**
 * レース番号。章番号 (§NN) の代わりにこの題材固有の番号を見出しに置く。
 * 琥珀 = オッズ・金額・レース番号に使うアクセント。
 */
function RaceNumber({ raceId }: { raceId: string }) {
  const n = raceId.slice(-2).replace(/^0/, '');
  if (!n) return null;
  return (
    <span className="font-mono text-2xl font-bold tabular-nums leading-none text-primary">
      {n}
      <span className="text-base">R</span>
    </span>
  );
}

function RaceDetailSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-40 w-full rounded-sm" />
      <Skeleton className="h-64 w-full rounded-sm" />
      <Skeleton className="h-64 w-full rounded-sm" />
    </div>
  );
}

/**
 * 単勝期待値 (EV) = 単勝確率 × 単勝オッズ。
 * 現行モデルは decision-focused（ROI 直接最適化）なので、score より EV が
 * 「買うべきか」の主指標。オッズ未確定 (odds_win=null) のときは null。
 */
function winEv(pred: HorsePrediction | null, entry: EntrySummary | undefined): number | null {
  if (!pred || entry?.odds_win == null) return null;
  return pred.win_prob * entry.odds_win;
}

/** 単勝のオッズ下限 (Settings の win_min_odds)。取得前・欠損時だけ既定値。 */
const DEFAULT_WIN_MIN_ODDS = 1.1;

function useWinMinOdds(): number {
  const settings = useSettings();
  const v = settings.data?.win_min_odds;
  return typeof v === 'number' && Number.isFinite(v) ? v : DEFAULT_WIN_MIN_ODDS;
}

/**
 * 単勝の推奨は「**モデル 1 位の馬**をオッズ下限より上のときに買う」。
 *
 * EV > 閾値 ではない。較正済み確率で EV フィルタを掛けると、平坦な確率 × 大穴
 * オッズで偽の期待値が量産され回収率が 0.931 → 0.698 まで落ちる (test 19ヶ月実測)。
 * バックエンドの推奨ロジック (strategy.recommend_for_race の win_min_odds) と同じ規則。
 */
function isBuy(
  entry: EntrySummary | undefined,
  isTopPick: boolean,
  minOdds: number,
): boolean {
  if (!isTopPick) return false;
  const odds = entry?.odds_win;
  return odds != null && odds > minOdds;
}

/**
 * 単勝 EV の色分け: 1.0 (収支トントン) を超えた場合だけ強調する。
 * 緑は「プラス収支」専用の色なので、ここ以外では使わない。
 * **BUY = +EV ではない**点に注意 (本命買いも平均は 0.931 = マイナス)。
 */
interface EntryRow {
  entry: EntrySummary;
  pred: HorsePrediction | null;
}

type SortKey =
  | 'post_position'
  | 'horse_name'
  | 'odds_win'
  | 'popularity'
  | 'finish_position'
  | 'score'
  | 'win_prob'
  | 'place_prob'
  | 'win_ev';

type SortDir = 'asc' | 'desc';

interface SortState {
  key: SortKey;
  dir: SortDir;
}

// Columns that use asc as the initial direction when first clicked
const ASC_FIRST_KEYS: ReadonlySet<SortKey> = new Set(['post_position', 'popularity']);

/**
 * Compare two nullable numbers for sort purposes, returning the raw numeric
 * difference (positive = a after b, negative = a before b, 0 = equal).
 * null / NaN comparisons are NOT handled here — handled in sortRows to keep
 * null-last logic independent of sort direction.
 */
function compareNumbers(a: number, b: number): number {
  return a - b;
}

function compareStrings(a: string, b: string): number {
  return a.localeCompare(b, 'ja');
}

/**
 * Extract the raw numeric value for a given sort key from a row.
 * Returns undefined when the value is null / NaN / undefined.
 */
function numericValue(row: EntryRow, key: SortKey): number | undefined {
  let v: number | null | undefined;
  switch (key) {
    case 'post_position': v = row.entry.post_position; break;
    case 'odds_win':      v = row.entry.odds_win; break;
    case 'popularity':    v = row.entry.popularity; break;
    case 'finish_position': v = row.entry.finish_position; break;
    case 'score':         v = row.pred?.score; break;
    case 'win_prob':      v = row.pred?.win_prob; break;
    case 'place_prob':    v = row.pred?.place_prob; break;
    case 'win_ev':        v = winEv(row.pred, row.entry); break;
    default: return undefined;
  }
  if (v == null || isNaN(v as number)) return undefined;
  return v as number;
}

function sortRows(rows: EntryRow[], sort: SortState): EntryRow[] {
  const multiplier = sort.dir === 'asc' ? 1 : -1;

  return [...rows].sort((a, b) => {
    if (sort.key === 'horse_name') {
      const aNull = a.entry.horse_name == null;
      const bNull = b.entry.horse_name == null;
      if (aNull && bNull) return 0;
      // null is always last regardless of direction
      if (aNull) return 1;
      if (bNull) return -1;
      return compareStrings(a.entry.horse_name!, b.entry.horse_name!) * multiplier;
    }

    const av = numericValue(a, sort.key);
    const bv = numericValue(b, sort.key);

    if (av === undefined && bv === undefined) return 0;
    // null / NaN is always last regardless of direction
    if (av === undefined) return 1;
    if (bv === undefined) return -1;

    return compareNumbers(av, bv) * multiplier;
  });
}

interface SortableHeaderProps {
  label: string;
  sortKey: SortKey;
  sort: SortState;
  onSort: (key: SortKey) => void;
  className?: string;
  /** 列の意味を補う説明 (EV の式など)。 */
  title?: string;
}

function SortableHeader({ label, sortKey, sort, onSort, className, title }: SortableHeaderProps) {
  const isActive = sort.key === sortKey;
  const Icon = isActive
    ? sort.dir === 'asc'
      ? ChevronUp
      : ChevronDown
    : ChevronsUpDown;

  return (
    <TableHead
      className={`cursor-pointer select-none whitespace-nowrap ${
        isActive ? 'text-primary' : ''
      } ${className ?? ''}`}
      onClick={() => onSort(sortKey)}
      aria-sort={isActive ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'}
      title={title}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        <Icon
          className={`h-3 w-3 ${isActive ? 'text-primary' : 'text-muted-foreground/40'}`}
        />
      </span>
    </TableHead>
  );
}

const EV_FORMULA = 'EV = 単勝確率 × 単勝オッズ（1.0 = 収支トントン）';

function buyTooltip(minOdds: number): string {
  return (
    `${EV_FORMULA}。BUY は「モデル1位の馬（オッズ ${minOdds} 超）」に出しています。` +
    'EV が 1.0 を超えた馬ではありません — 較正済み確率で EV 条件にすると大穴を' +
    '買い込んで回収率が 0.931→0.698 に落ちるためです。本番モデルの OOS 単勝回収率は' +
    ' 0.931（人気1番 0.792 を上回るが依然 1.0 未満＝平均では負け越し）。参考値。'
  );
}

interface EntryPredictionTableProps {
  entries: EntrySummary[];
  predictions: HorsePrediction[] | null;
}

/**
 * AI 予想グループの列に敷く淡い面。「どこからが AI の出力か」を色ではなく
 * 面で示す (色は 1 語 1 義に保ちたいので、グループの識別に色相を消費しない)。
 */
const AI_SURFACE = 'bg-primary/[0.035]';

/** 未確定・未算出セル。「—」は作りかけに見えるので控えめな中黒にする。 */
function Pending() {
  return <span className="text-subtle-foreground/50">·</span>;
}

/**
 * 確率セル。数値の背後にバーを敷いて、18 頭の分布が一目で分かるようにする。
 * これで表が「データ表」から「可視化」になる。
 */
function ProbCell({ value }: { value: number | null | undefined }) {
  if (value == null || !Number.isFinite(value)) return <Pending />;
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="relative flex justify-end">
      <div
        aria-hidden="true"
        className="absolute inset-y-1 right-0 bg-primary/15"
        style={{ width: `${pct}%` }}
      />
      <span className="relative px-1">{formatPercent(value)}</span>
    </div>
  );
}

/**
 * Unified table merging entry data and prediction scores.
 *
 * 列は「結論 → 根拠 → 事実」の順に並べる:
 *   馬番 | 馬名 ‖ 1着確率 3着内率 スコア 参考EV ‖ オッズ 人気 騎手 年齢/性別 (着順) 馬体重
 *              └──── AI の根拠 (淡い primary 面) ────┘  └──── 実績 (無彩色) ────┘
 *
 * 買う馬は「推奨」カードに出す。表の行に BUY バッジを付けていたが、**買うのは
 * 常にモデル 1 位の 1 頭**で、バッジは 1 行にしか付かない。列を 1 つ使って
 * 「1 位かどうか」を二重に示していただけなので外した。
 *
 * 着順はレース後にしか確定しないので、結果が入るまで列ごと出さない
 * (「—」で埋まった列があると作りかけに見えるため)。
 *
 * Default sort: score desc when predictions are available, post_position asc otherwise.
 * Clicking a sortable column header toggles sort direction.
 * null / NaN values always sort to the bottom regardless of direction.
 */
function EntryPredictionTable({ entries, predictions }: EntryPredictionTableProps) {
  const defaultSort: SortState = predictions
    ? { key: 'score', dir: 'desc' }
    : { key: 'post_position', dir: 'asc' };

  const [sort, setSort] = useState<SortState>(defaultSort);

  function handleSort(key: SortKey) {
    setSort((prev) => {
      if (prev.key === key) {
        return { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' };
      }
      // Different column: start with asc for asc-first keys, desc for the rest
      const dir: SortDir = ASC_FIRST_KEYS.has(key) ? 'asc' : 'desc';
      return { key, dir };
    });
  }

  const rows = useMemo<EntryRow[]>(() => {
    const predMap = new Map(predictions?.map((p) => [p.horse_id, p]) ?? []);
    const merged: EntryRow[] = entries.map((entry) => ({
      entry,
      pred: predMap.get(entry.horse_id) ?? null,
    }));
    return merged;
  }, [entries, predictions]);

  const sortedRows = useMemo(() => sortRows(rows, sort), [rows, sort]);
  const minOdds = useWinMinOdds();
  const buyTip = buyTooltip(minOdds);
  // レース後 (= 着順が 1 頭でも確定している) かどうか。未確定なら着順列は出さない。
  const hasResults = useMemo(
    () => entries.some((e) => e.finish_position != null),
    [entries],
  );

  const headerProps = { sort, onSort: handleSort };
  const factsColSpan = hasResults ? 6 : 5;
  // 枠番は馬番と出走頭数から導出する (API に枠番が無いため暫定)
  const runners = entries.length;

  return (
    <Table aria-label="出走馬">
      <TableHeader>
        {/* AI の出力 (根拠) と実績データを面で分離するためのグループ行 */}
        <TableRow>
          <TableHead colSpan={2} />
          <TableHead
            colSpan={4}
            className={`text-center text-[11px] font-medium text-primary ${AI_SURFACE}`}
          >
            AI の根拠
          </TableHead>
          <TableHead
            colSpan={factsColSpan}
            className="text-center text-[11px] font-medium text-muted-foreground"
          >
            実績データ
          </TableHead>
        </TableRow>
        <TableRow>

          <SortableHeader
            label="馬番"
            sortKey="post_position"
            className="sticky left-0 z-20 w-12 bg-background text-center"
            {...headerProps}
          />
          <SortableHeader
            label="馬名"
            sortKey="horse_name"
            className="sticky left-12 z-20 bg-background"
            {...headerProps}
          />
          <SortableHeader
            label="1着確率"
            sortKey="win_prob"
            className={`text-right ${AI_SURFACE}`}
            title="この馬が1着になる確率 (較正済み)"
            {...headerProps}
          />
          <SortableHeader
            label="3着内率"
            sortKey="place_prob"
            className={`text-right ${AI_SURFACE}`}
            title="この馬が3着以内に入る確率"
            {...headerProps}
          />
          <SortableHeader
            label="スコア"
            sortKey="score"
            className={`text-right ${AI_SURFACE}`}
            title="モデルの生の出力。確率に変換する前の順位づけの元"
            {...headerProps}
          />
          <SortableHeader
            label="参考EV"
            sortKey="win_ev"
            className={`text-right ${AI_SURFACE} text-subtle-foreground`}
            title={EV_FORMULA}
            {...headerProps}
          />
          <SortableHeader label="単勝オッズ" sortKey="odds_win" className="text-right" {...headerProps} />
          <SortableHeader label="人気" sortKey="popularity" className="text-right" {...headerProps} />
          <TableHead>騎手</TableHead>
          <TableHead className="text-center">年齢/性別</TableHead>
          {hasResults && (
            <SortableHeader label="着順" sortKey="finish_position" className="text-center" {...headerProps} />
          )}
          <TableHead className="text-right">馬体重</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {sortedRows.map(({ entry, pred }) => {
          return (
            <TableRow
              key={entry.horse_id}
              // bg-background は sticky セルの bg-inherit の土台 (hover もここで切替わる)
              className="bg-background hover:bg-card-elevated"
            >
              {/* ── 識別 ── */}
              {/* 横スクロールしても «どの馬か» を見失わないよう馬番・馬名を固定 */}
              <TableCell className="sticky left-0 z-10 bg-inherit text-center">
                <Umaban n={entry.post_position} runners={runners} />
              </TableCell>
              <TableCell className="sticky left-12 z-10 bg-inherit">
                {entry.horse_name ?? (
                  <span className="font-mono text-xs text-muted-foreground">{entry.horse_id}</span>
                )}
              </TableCell>
              {/* ── AI の根拠 ── 買う順序を決めているのは確率。EV は参考値なので最後 */}
              <TableCell className={`cell-num ${AI_SURFACE}`}>
                <ProbCell value={pred?.win_prob} />
              </TableCell>
              <TableCell className={`cell-num ${AI_SURFACE}`}>
                <ProbCell value={pred?.place_prob} />
              </TableCell>
              <TableCell className={`cell-num ${AI_SURFACE}`}>
                {pred != null ? formatScore(pred.score) : <Pending />}
              </TableCell>
              <TableCell className={`cell-num ${AI_SURFACE} text-subtle-foreground`} title={buyTip}>
                {pred != null ? formatRatio(winEv(pred, entry)) : <Pending />}
              </TableCell>
              {/* ── 実績 (無彩色) ── */}
              <TableCell className="cell-num">
                {entry.odds_win != null ? formatOdds(entry.odds_win) : <Pending />}
              </TableCell>
              <TableCell className="cell-num">{entry.popularity ?? <Pending />}</TableCell>
              <TableCell>{entry.jockey_name ?? <Pending />}</TableCell>
              <TableCell className="text-center">
                {entry.age ?? '?'}{entry.sex ?? ''}
              </TableCell>
              {hasResults && (
                <TableCell className="text-center">
                  {/* 着順は「外から取ってきた事実」なので色を付けない。
                      掲示板 (3着以内) だけ面と太さで区別する。 */}
                  {entry.finish_position != null ? (
                    entry.finish_position <= 3 ? (
                      <Badge variant="outline" className="text-foreground">
                        {entry.finish_position}着
                      </Badge>
                    ) : (
                      <span className="text-muted-foreground">{entry.finish_position}着</span>
                    )
                  ) : (
                    <Pending />
                  )}
                </TableCell>
              )}
              <TableCell className="cell-num">
                {entry.horse_weight != null ? (
                  <>
                    {entry.horse_weight}
                    <span className="text-unit">kg</span>
                    {entry.horse_weight_diff != null && (
                      <span className="ml-1 text-xs text-muted-foreground">
                        ({entry.horse_weight_diff > 0 ? '+' : ''}{entry.horse_weight_diff})
                      </span>
                    )}
                  </>
                ) : (
                  <Pending />
                )}
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

/**
 * 出走馬に過去走がほとんど無いレース (新馬戦など) の注意書き。
 *
 * モデルは per-race 履歴 GRU と直近着順・上がり・脚質を主要な入力にしているので、
 * 全員が初出走だとそれが全滅し、枠順・馬体重・騎手・血統・オッズだけの予想になる。
 * 同じ画面・同じスコアでも入力の質が別物なので、黙って出さずに明示する。
 *
 * 実測 (test 19ヶ月・432 レース) では的中率はむしろ高い (単勝 29.2% / 複勝 63.7%) が、
 * 人気馬に寄るぶんオッズが低く、回収率は単勝 0.866 と全体 (0.933) を下回る。
 * 「当たりやすいが儲かりにくい」ので、的中率だけ見て信用しすぎないための注記でもある。
 */
function LowInformationNotice({ coverage }: { coverage: RaceInfoCoverage }) {
  return (
    <Card className="border-warning/50 bg-warning/5">
      <CardContent className="flex items-start gap-3 py-4">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" aria-hidden="true" />
        <div className="flex flex-col gap-1">
          <p className="text-sm font-medium text-foreground">
            このレースは判断材料が少なめです
          </p>
          <p className="text-xs leading-relaxed text-muted-foreground">
            出走 {coverage.n_runners} 頭のうち <strong>{coverage.n_debut} 頭に過去走がありません</strong>
            （1 頭あたり平均 {coverage.mean_starts} 走）。AI が重く使っている「前走までの
            着順・上がり・脚質」がほとんど無いため、枠順・馬体重・騎手・血統・オッズだけで
            予想しています。参考程度にしてください。
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * 買い方を 1 箇所で説明する。
 *
 * 以前は「BUY バッジの説明」「オッズの出所」「確信度」「買い目の並び順」が別々の
 * 注記として散らばり、EV / 期待値 / 確信度 / 的中確率 が混在していた。読む側は
 * **どの数字がどの判断に使われるのか**を知りたいので、券種ごとに条件と金額を並べ、
 * 使う数字を 2 つに絞って示す。
 */
function BettingRuleNote() {
  const minOdds = useWinMinOdds();
  return (
    <div className="mt-3 flex flex-col gap-2 border-t border-border pt-3 text-xs">
      <p className={labelClass('mb-0')}>買い方</p>
      <table className="w-full max-w-2xl text-left">
        <thead className="text-subtle-foreground">
          <tr>
            <th className="py-0.5 pr-4 font-normal">券種</th>
            <th className="py-0.5 pr-4 font-normal">買う条件</th>
            <th className="py-0.5 font-normal">点数</th>
          </tr>
        </thead>
        <tbody className="text-muted-foreground">
          <tr>
            <td className="py-0.5 pr-4 text-foreground">単勝</td>
            <td className="py-0.5 pr-4">モデル1位の馬。オッズ {minOdds} 倍超のときだけ</td>
            <td className="py-0.5">1 点</td>
          </tr>
          <tr>
            <td className="py-0.5 pr-4 text-foreground">複勝</td>
            <td className="py-0.5 pr-4">モデル1位の馬。確信度が下限以上のとき</td>
            <td className="py-0.5">確信度が高いほど厚く (1〜3 倍)</td>
          </tr>
          <tr>
            <td className="py-0.5 pr-4 text-foreground">連系</td>
            <td className="py-0.5 pr-4">上位数頭の組合せを、的中確率の高い順に</td>
            <td className="py-0.5">券種ごとに上位数点まで</td>
          </tr>
        </tbody>
      </table>
      <p className="text-subtle-foreground">
        使う数字は 2 つだけです。<strong className="font-medium">的中確率</strong>
        （その馬・その組合せが当たる確率。買う順序を決める）と
        <strong className="font-medium">確信度</strong>
        （確率専用モデルが「モデル1位の馬は3着以内」と見た確率。複勝の可否と厚みを決める）。
        <span className="ml-1">
          期待値（EV）は買う判断に使っていません — 較正済みの確率で EV を条件にすると
          大穴に寄り、実測で単勝回収率が 0.93 → 0.70 に落ちるためです。表の「参考EV」は
          値を見せているだけです。
        </span>
      </p>
      <p className="text-subtle-foreground">
        <strong className="font-medium">予算は上限であって、使い切る目標ではありません。</strong>
        連系は的中確率の高い順に並ぶので、深く買うほど当たりにくい買い目に金を足すことに
        なります（実測 5,404 レース: 1 点目の的中率 15.5% → 10 点目 3.7%）。上位数点で
        止めると、連系に使う金が半分以下になって回収率は同じかやや良くなります。
      </p>
      <p className="text-subtle-foreground">
        実測の回収率は単勝 0.93 / 複勝 0.89（確信度で絞ると 0.92）/ 連系 0.85〜0.88。
        <strong className="font-medium">いずれも 1.0 未満</strong>で、控除率の内側です。
      </p>
    </div>
  );
}

/**
 * 結論。表を読ませる前に AI の推奨を数行で出す (B-4 ①)。
 * 「18 頭 × 13 列のどこを見ればいいか」を最初に解決する。
 */
function ConclusionCard({
  entries,
  predictions,
}: {
  entries: EntrySummary[];
  predictions: HorsePrediction[] | null;
}) {
  const runners = entries.length;
  const minOdds = useWinMinOdds();
  const picks = useMemo(() => {
    if (!predictions || predictions.length === 0) return [];
    const byId = new Map(entries.map((e) => [e.horse_id, e]));
    const topId = predictions.reduce((a, b) => (b.score > a.score ? b : a)).horse_id;
    return predictions
      .map((pred) => ({ pred, entry: byId.get(pred.horse_id) }))
      .filter((r) => r.entry && isBuy(r.entry, r.pred.horse_id === topId, minOdds));
  }, [entries, predictions, minOdds]);

  if (!predictions) return null;

  if (picks.length === 0) {
    return (
      <div className="border-y border-border py-4">
        <p className="text-label-ja mb-1">推奨</p>
        <p className="text-sm text-muted-foreground">
          単勝 EV が 1.1 を超える馬はいません。このレースは見送りが妥当です。
        </p>
      </div>
    );
  }

  return (
    <div className="border-y border-border py-4">
      <p className="text-label-ja mb-2">推奨</p>
      <ul className="flex flex-col gap-2">
        {picks.slice(0, 3).map(({ pred, entry }) => (
          <li key={pred.horse_id} className="flex flex-wrap items-center gap-x-6 gap-y-1">
            <span className="flex items-center gap-2">
              <Umaban n={entry!.post_position} runners={runners} />
              <span className="font-medium">{entry!.horse_name ?? entry!.horse_id}</span>
            </span>
            <span className="text-sm" title={EV_FORMULA}>
              <span className="text-label">単勝EV</span>{' '}
              <span className="font-mono font-medium tabular-nums text-success">
                {formatRatio(winEv(pred, entry))}
              </span>
            </span>
            <span className="text-sm">
              <span className="text-label">単勝確率</span>{' '}
              <span className="font-mono tabular-nums">{formatPercent(pred.win_prob)}</span>
            </span>
            <span className="text-sm">
              <span className="text-label">オッズ</span>{' '}
              <span className="font-mono tabular-nums text-primary">
                {formatOdds(entry!.odds_win)}
              </span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * レース後の答え合わせ (B-4 ⑤)。推奨した馬がどうだったかを回収率で出す。
 * 着順精度ではなく回収率で評価する、というこのモデルの設計と一致させる。
 */
function ResultReviewCard({
  entries,
  predictions,
  payoutWin,
}: {
  entries: EntrySummary[];
  predictions: HorsePrediction[] | null;
  payoutWin: number | null;
}) {
  const minOdds = useWinMinOdds();
  const review = useMemo(() => {
    if (!predictions || predictions.length === 0) return null;
    const byId = new Map(entries.map((e) => [e.horse_id, e]));
    const topId = predictions.reduce((a, b) => (b.score > a.score ? b : a)).horse_id;
    const picks = predictions
      .map((pred) => ({ pred, entry: byId.get(pred.horse_id) }))
      .filter((r) => r.entry && isBuy(r.entry, r.pred.horse_id === topId, minOdds));
    if (picks.length === 0) return null;

    const finished = picks.filter((r) => r.entry?.finish_position != null);
    if (finished.length === 0) return null;

    const winners = finished.filter((r) => r.entry!.finish_position === 1);
    const placed = finished.filter((r) => (r.entry!.finish_position ?? 99) <= 3);
    // 単勝を各 100 円ずつ買った場合の回収率。payout_win は 100 円あたりの払戻。
    const invested = picks.length * 100;
    const returned = winners.length > 0 && payoutWin != null ? payoutWin * winners.length : 0;
    return {
      picks: picks.length,
      winners: winners.length,
      placed: placed.length,
      roi: invested > 0 ? returned / invested : null,
      returned,
      winnerNames: winners.map((r) => r.entry!.horse_name ?? r.entry!.horse_id),
    };
  }, [entries, predictions, payoutWin, minOdds]);

  if (!review) return null;

  const hit = review.winners > 0;
  return (
    <div className="border-y border-border py-4">
      <p className="text-label-ja mb-2">答え合わせ</p>
      <div className="flex flex-wrap items-baseline gap-x-8 gap-y-2 text-sm">
        <span>
          推奨 <span className="font-mono tabular-nums">{review.picks}</span> 頭中{' '}
          <span className={hit ? 'font-medium text-success' : 'text-muted-foreground'}>
            {review.winners}
          </span>{' '}
          頭が1着 / <span className="font-mono tabular-nums">{review.placed}</span> 頭が3着以内
        </span>
        {hit && review.returned > 0 && (
          <span>
            <span className="text-label">単勝払戻</span>{' '}
            <span className="font-mono tabular-nums text-primary">{formatYen(review.returned)}</span>
          </span>
        )}
        {review.roi != null && (
          <span>
            <span className="text-label">この予想の回収率</span>{' '}
            <span
              className={`font-mono text-lg font-medium tabular-nums ${
                review.roi >= 1 ? 'text-success' : 'text-destructive'
              }`}
            >
              {formatPercent(review.roi, 0)}
            </span>
          </span>
        )}
      </div>
      <p className="mt-2 text-xs leading-relaxed text-subtle-foreground">
        推奨した馬の単勝を各 100 円ずつ買った場合の回収率です（100% = 収支トントン）。
        {hit && review.winnerNames.length > 0 && ` 的中: ${review.winnerNames.join('・')}`}
      </p>
    </div>
  );
}

/** 「12:04」形式。オッズの鮮度と最終実行時刻に使う。 */
function formatClock(d: Date): string {
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

/**
 * 実行中の進捗。段階名を出す 1 本のバーにまとめる。
 * ジョブはインメモリ管理なので、バックエンド再起動で追跡できなくなることも伝える。
 */
function RunProgress({ stage }: { stage: 'entries' | 'predict' }) {
  const label = stage === 'entries' ? '出馬表を取得中…' : '予想を計算中…';
  return (
    <div className="border-y border-border py-3">
      <div className="flex items-center gap-3">
        <div className="h-1 flex-1 overflow-hidden rounded-full bg-card-elevated">
          <div
            className="h-full w-1/3 animate-skeleton-shimmer bg-primary"
            style={{ marginLeft: stage === 'entries' ? '0' : '33%' }}
          />
        </div>
        <span className="font-mono text-[11px] tabular-nums text-primary">{label}</span>
      </div>
      {stage === 'entries' && (
        <p className="mt-1.5 text-xs text-subtle-foreground">
          この実行はバックエンドを再起動すると追跡できなくなります。
        </p>
      )}
    </div>
  );
}

/** 表の形をしたスケルトン。塊で出すと表示された瞬間にレイアウトが跳ねる。 */
function TableSkeleton({ rows }: { rows: number }) {
  return (
    <div className="w-full">
      <div className="border-b-2 border-border-strong py-2">
        <Skeleton className="h-3 w-40" />
      </div>
      {Array.from({ length: Math.min(rows, 18) }).map((_, i) => (
        <div key={i} className="flex items-center gap-3 border-b border-border py-2">
          <Skeleton className="h-6 w-6 shrink-0" />
          <Skeleton className="h-3 flex-1" />
          <Skeleton className="h-3 w-16 shrink-0" />
          <Skeleton className="h-3 w-16 shrink-0" />
        </div>
      ))}
    </div>
  );
}

/**
 * 前後のレースへ移動 (B-4 ⑦)。race_id の末尾 2 桁がレース番号なので、
 * 1〜12 の範囲で前後に振る。1 開催を順に見る動線がこれで通る。
 */
function RaceStepper({ raceId, date }: { raceId: string; date: string | null }) {
  const prefix = raceId.slice(0, -2);
  const n = Number(raceId.slice(-2));
  if (!Number.isFinite(n) || n < 1) return null;
  const q = date ? `?date=${date}` : '';
  const to = (target: number) => `/races/${prefix}${String(target).padStart(2, '0')}${q}`;
  return (
    <div className="flex items-center gap-1">
      <Button asChild variant="ghost" size="sm" disabled={n <= 1}>
        <Link to={to(n - 1)} aria-label="前のレース" aria-disabled={n <= 1}>
          <ChevronLeft className="h-4 w-4" />
          {n - 1}R
        </Link>
      </Button>
      <Button asChild variant="ghost" size="sm" disabled={n >= 12}>
        <Link to={to(n + 1)} aria-label="次のレース" aria-disabled={n >= 12}>
          {n + 1}R
          <ChevronRight className="h-4 w-4" />
        </Link>
      </Button>
    </div>
  );
}

interface MetaItemProps {
  label: string;
  value: string;
  mono?: boolean;
}

function MetaItem({ label, value, mono }: MetaItemProps) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className={mono ? 'font-mono text-xs' : ''}>{value}</dd>
    </div>
  );
}

export function RaceDetail() {
  const { race_id = '' } = useParams<{ race_id: string }>();
  const [searchParams] = useSearchParams();
  const dateParam = searchParams.get('date');

  // AI 予想 (予想スコア + 推奨買い目) は画面を開いた瞬間ではなく、ボタンを
  // 押して初めて走らせる (重い推論を自動実行しない / ユーザー要望)。
  const [aiRequested, setAiRequested] = useState(false);
  // 出馬表の取得完了を待って予想まで自動で進めるためのフラグ
  const [autoPredict, setAutoPredict] = useState(false);
  const [lastRunAt, setLastRunAt] = useState<Date | null>(null);
  // 出馬表を再取得した時刻 = オッズの鮮度。API に odds_fetched_at が無いので
  // この画面で取得した時刻を持つ (再訪時は分からないため null になる)。
  const [oddsFetchedAt, setOddsFetchedAt] = useState<Date | null>(null);
  // このレースだけの条件 (券種・予算)。空 = Settings の既定値をそのまま使う。
  const [overrides, setOverrides] = useState<RecommendationOverrides>({});

  const raceQuery = useRaceDetail(race_id);
  const predQuery = usePredictions(race_id, aiRequested);
  // 画面で扱う値 (使う金額 / 1 点あたり / 券種 / 狙い方) が
  // そのまま API のパラメータになる (変換なし)。
  const recQuery = useRecommendations(
    race_id,
    aiRequested && Boolean(race_id) && !raceQuery.isPending && !raceQuery.isError,
    overrides,
  );
  // runShutuba scoped to this race so raceDetail is invalidated on completion
  const runShutubaMutation = useRunShutuba(race_id);

  const race = raceQuery.data;
  const entryCount = race?.entries.length ?? 0;

  // 出馬表が入った時点で、待っていた予想を続けて実行する (B-1 ①)
  useEffect(() => {
    if (!autoPredict || entryCount === 0) return;
    setAutoPredict(false);
    setAiRequested(true);
    setLastRunAt(new Date());
    setOddsFetchedAt(new Date());
  }, [autoPredict, entryCount]);

  // NOTE: 出馬表取込・AI 予想はいずれも画面表示時に自動実行しない。
  // すべて下部の各ボタン (出馬表を取得 / AI 予想を実行) で明示的に開始する。

  // Race ページの Past タブへ戻る (?tab=past)。date を引き継いで一覧の選択日を復元。
  // 旧 `/past` は /races へ redirect され query を落とすため直接 /races を指す。
  const backLink = dateParam
    ? `/races?tab=past&date=${dateParam}`
    : '/races?tab=past';

  if (raceQuery.isPending) {
    return (
      <div className="flex flex-col gap-8 p-6">
        <BackLink to={backLink} />
        <PageHeader
          marker={<RaceNumber raceId={race_id} />}
          eyebrow="Race Detail"
          title="レース詳細"
          description={race_id}
        />
        <RaceDetailSkeleton />
      </div>
    );
  }

  if (raceQuery.isError) {
    const is404 = isNotFoundError(raceQuery.error);
    return (
      <div className="flex flex-col gap-8 p-6">
        <BackLink to={backLink} />
        <PageHeader
          marker={<RaceNumber raceId={race_id} />}
          eyebrow="Race Detail"
          title="レース詳細"
          description={race_id}
        />
        <EmptyState
          message={is404 ? '指定レース ID は見つかりません' : 'レース詳細の取得に失敗しました'}
          description={is404 ? undefined : 'バックエンドが起動しているか確認してください。'}
        />
        {is404 && (
          <div className="flex justify-center">
            <Button asChild variant="outline">
              <Link to="/upcoming">Upcoming Races へ戻る</Link>
            </Button>
          </div>
        )}
      </div>
    );
  }

  // raceQuery が success であっても TanStack Query の型は data: RaceDetail | undefined。
  // ここで明示的に narrowing し、以降は race を非 null として扱えるようにする。
  if (!race) {
    return (
      <div className="flex flex-col gap-8 p-6">
        <BackLink to={backLink} />
        <PageHeader
          marker={<RaceNumber raceId={race_id} />}
          eyebrow="Race Detail"
          title="レース詳細"
          description={race_id}
        />
        <RaceDetailSkeleton />
      </div>
    );
  }

  const predictions = predQuery.data?.predictions ?? null;
  const infoCoverage = predQuery.data?.info_coverage ?? null;

  const hasEntries = race.entries.length > 0;

  const isScrapingShutuba = runShutubaMutation.isPending || runShutubaMutation.isPolling;
  const isPredicting = aiRequested && (predQuery.isFetching || recQuery.isFetching);
  // 出馬表の取得 → 予想 の一連を 1 ボタンで通す。いまどの段階かを 1 本で見せる。
  const stage: 'idle' | 'entries' | 'predict' = isScrapingShutuba
    ? 'entries'
    : isPredicting
      ? 'predict'
      : 'idle';
  const busy = stage !== 'idle';

  function handleRunShutuba(thenPredict: boolean) {
    if (thenPredict) setAutoPredict(true);
    runShutubaMutation.mutate(
      { race_ids: [race_id] },
      {
        onError: async (err) => {
          setAutoPredict(false);
          const msg = await formatErrorMessage(err);
          toast.error('出馬表の取得に失敗しました', {
            description: msg,
            action: { label: '再試行', onClick: () => handleRunShutuba(thenPredict) },
          });
        },
      }
    );
  }

  /**
   * 「予想を見る」= 出馬表が無ければ取得 → 完了を待って予想、まで連鎖させる。
   * 2 回目以降は「予想を更新」。同じ文言のまま挙動を変えない。
   */
  function handleShowPrediction() {
    if (!hasEntries) {
      handleRunShutuba(true);
      return;
    }
    if (!aiRequested) {
      setAiRequested(true);
      setLastRunAt(new Date());
      return;
    }
    predQuery.refetch();
    recQuery.refetch();
    setLastRunAt(new Date());
  }

  return (
    <div className="flex flex-col gap-8 p-6">
      <div className="flex items-center justify-between">
        <BackLink to={backLink} />
        <RaceStepper raceId={race.race_id} date={dateParam} />
      </div>

      <PageHeader
        marker={<RaceNumber raceId={race.race_id} />}
        eyebrow="Race Detail"
        title={race.name ?? `${race.course} ${race.race_class ?? ''}`.trim()}
        description={`${race.date}・${race.surface}${race.distance}m・${race.race_id}`}
      >
        {/* ボタンは 1 行に揃え、注記は下の固定高の行にまとめる。
            ボタンごとに下へ注記を付けると、注記が出た瞬間に高さが変わって
            ボタンの位置がずれてしまう。 */}
        <div className="flex flex-col items-end gap-1">
          <div className="flex items-center gap-2">
            {hasEntries && (
              <Button
                variant="outline"
                size="sm"
                disabled={busy}
                onClick={() => handleRunShutuba(false)}
                title="出馬表を再取得して単勝オッズ・人気・馬場状態を最新化します（発走が近いほど確定値に近づく）"
              >
                <RefreshCw className="mr-1.5 h-4 w-4" />
                オッズ更新
              </Button>
            )}
            <Button size="sm" disabled={busy} onClick={handleShowPrediction}>
              <Sparkles className="mr-1.5 h-4 w-4" />
              {aiRequested ? '予想を更新' : '予想を見る'}
            </Button>
          </div>
          {/* 高さは常に確保する (空でも詰めない) */}
          <div className="flex h-4 items-center gap-3 font-mono text-[10px] tabular-nums text-subtle-foreground">
            {oddsFetchedAt && <span>オッズ {formatClock(oddsFetchedAt)} 時点</span>}
            {lastRunAt && !busy && <span>予想 {formatClock(lastRunAt)}</span>}
          </div>
        </div>
      </PageHeader>

      {/* 進捗は 1 本にまとめ、いまどの段階かを言葉で出す (B-1 ①) */}
      {busy && <RunProgress stage={stage} />}

      {/* Race overview */}
      <Card className="border-t border-border pt-6">
        <CardHeader>
          <CardTitle className="text-label-ja">レース概要</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3 lg:grid-cols-4">
            <MetaItem label="レース ID" value={race.race_id} mono />
            <MetaItem label="レース名" value={race.name ?? '·'} />
            <MetaItem label="開催日" value={race.date} />
            <MetaItem label="競馬場" value={race.course} />
            <MetaItem label="馬場種別" value={race.surface} />
            <MetaItem label="距離" value={race.distance ? `${race.distance} m` : '·'} />
            <MetaItem label="天候" value={race.weather ?? '·'} />
            <MetaItem label="馬場状態" value={race.track_condition ?? '·'} />
            <MetaItem label="クラス" value={race.race_class ?? '·'} />
            <MetaItem label="出走頭数" value={race.n_runners?.toString() ?? '·'} />
            {/* 払戻はレース後にしか確定しない。未確定なら行ごと隠す
                (「単勝払戻 —」が並ぶと動いていないアプリに見えるため)。 */}
            {race.payout_win != null && (
              <MetaItem label="単勝払戻" value={formatYen(race.payout_win)} />
            )}
            {race.payout_place != null && (
              <MetaItem label="複勝払戻" value={race.payout_place} />
            )}
          </dl>
        </CardContent>
      </Card>

      {/* 出馬表が未取得のとき: ボタンで取り込む (自動取得しない) */}
      {!hasEntries && (
        <Card className="border-t border-border pt-6">
          <CardContent>
            <EmptyState
              message="出馬表がまだ取り込まれていません"
              description="下のボタンで出馬表の取得から予想までまとめて実行します。"
            >
              <Button onClick={handleShowPrediction} disabled={busy}>
                <Sparkles className="mr-1.5 h-4 w-4" />
                予想を見る
              </Button>
            </EmptyState>
          </CardContent>
        </Card>
      )}

      {/* 結論 → 答え合わせ → 根拠(表) の順。表を読ませる前に結論を出す */}
      {hasEntries && aiRequested && !predQuery.isPending && !predQuery.isError && (
        <>
          {infoCoverage?.is_low_information && (
            <LowInformationNotice coverage={infoCoverage} />
          )}
          <ConclusionCard entries={race.entries} predictions={predictions} />
          <ResultReviewCard
            entries={race.entries}
            predictions={predictions}
            payoutWin={race.payout_win}
          />
        </>
      )}

      {/* 推奨買目も表より上に置く */}
      {hasEntries && aiRequested && (
        <RecommendationsCard
          raceId={race_id}
          data={recQuery.data}
          isPending={recQuery.isPending}
          isError={recQuery.isError}
          error={recQuery.error}
          runners={race.entries.length}
          overrides={overrides}
          onOverridesChange={setOverrides}
        />
      )}

      {/* Unified entry + prediction table */}
      {hasEntries && (
        <Card className="border-t border-border pt-6">
          <CardHeader>
            <CardTitle className="text-label-ja">出走馬一覧</CardTitle>
          </CardHeader>
          <CardContent>
            {!aiRequested ? (
              // AI 予想 未実行: 実績データのみ表示 (予想列は空欄)。
              // 上部の「AI 予想を実行」ボタンでスコア + 推奨を取得する。
              <>
                <p className="mb-3 text-sm text-muted-foreground">
                  「予想を見る」で予想スコア（単勝/複勝確率）と推奨買い目を取得します。
                </p>
                <EntryPredictionTable entries={race.entries} predictions={null} />
                <BettingRuleNote />
              </>
            ) : predQuery.isPending ? (
              <TableSkeleton rows={race.entries.length || 8} />
            ) : predQuery.isError ? (
              <>
                <p className="mb-3 text-sm text-muted-foreground">
                  {isServiceUnavailableError(predQuery.error)
                    ? 'active モデルが見つかりません。予想スコア列は非表示です。'
                    : '予想データを取得できません。予想スコア列は非表示です。'}
                </p>
                <EntryPredictionTable entries={race.entries} predictions={null} />
                <BettingRuleNote />
              </>
            ) : (
              <>
                <EntryPredictionTable entries={race.entries} predictions={predictions} />
                <BettingRuleNote />
              </>
            )}
          </CardContent>
        </Card>
      )}

    </div>
  );
}

interface BackLinkProps {
  to: string;
}

function BackLink({ to }: BackLinkProps) {
  return (
    <Link
      to={to}
      className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      aria-label="Past Races へ戻る"
    >
      <ChevronLeft className="h-4 w-4" />
      戻る
    </Link>
  );
}
