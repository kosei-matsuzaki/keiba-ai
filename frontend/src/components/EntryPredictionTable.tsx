import { useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, ChevronUp, ChevronsUpDown } from 'lucide-react';

import { HorsePastRuns } from '@/components/HorsePastRuns';
import { Umaban } from '@/components/Umaban';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useSettings } from '@/hooks/useSettings';
import {
  ASC_FIRST_KEYS,
  sortRows,
  winEv,
  type EntryRow,
  type SortKey,
  type SortState,
  type SortDir,
} from '@/lib/entrySort';
import { formatOdds, formatPercent, formatRatio, formatScore } from '@/lib/formatters';
import type { EntrySummary, HorsePrediction } from '@/types/api';

/** 単勝のオッズ下限 (Settings の win_min_odds)。取得前・欠損時だけ既定値。 */
const DEFAULT_WIN_MIN_ODDS = 1.1;

function useWinMinOdds(): number {
  const settings = useSettings();
  const v = settings.data?.win_min_odds;
  return typeof v === 'number' && Number.isFinite(v) ? v : DEFAULT_WIN_MIN_ODDS;
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

/** 参考 EV 列のツールチップ。**買う判断には使っていない**ことだけ伝える。 */
function buyTooltip(minOdds: number): string {
  return (
    `${EV_FORMULA}。参考値で、買う判断には使っていません` +
    `（単勝は「モデル1位の馬・オッズ ${minOdds} 超」で買います）。`
  );
}

interface EntryPredictionTableProps {
  entries: EntrySummary[];
  predictions: HorsePrediction[] | null;
  /** このレースの開催日。過去走はこれより**厳密に前**だけを出す。 */
  raceDate: string;
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

/** 確率セル。**数値だけ**を出す (背後のバーは見比べの役に立たず、目が散る)。 */
function ProbCell({ value }: { value: number | null | undefined }) {
  if (value == null || !Number.isFinite(value)) return <Pending />;
  return <span>{formatPercent(value)}</span>;
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
export function EntryPredictionTable({
  entries,
  predictions,
  raceDate,
}: EntryPredictionTableProps) {
  // 開いている馬。**複数開ける** — 何頭かの前走を見比べるのが実際の使い方なので。
  const [openHorses, setOpenHorses] = useState<Set<string>>(new Set());

  function toggleHorse(horseId: string) {
    setOpenHorses((prev) => {
      const next = new Set(prev);
      if (next.has(horseId)) next.delete(horseId);
      else next.add(horseId);
      return next;
    });
  }

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
            title="この馬が1着になる確率。推奨買目の「単勝の確信度」と同じ数字"
            {...headerProps}
          />
          <SortableHeader
            label="3着内率"
            sortKey="place_prob"
            className={`text-right ${AI_SURFACE}`}
            title="この馬が3着以内に入る確率。推奨買目の「複勝の確信度」と同じ数字"
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
        {sortedRows.flatMap(({ entry, pred }) => {
          const open = openHorses.has(entry.horse_id);
          return [
            <TableRow
              key={entry.horse_id}
              // bg-background は sticky セルの bg-inherit の土台 (hover もここで切替わる)
              className="cursor-pointer bg-background hover:bg-card-elevated"
              onClick={() => toggleHorse(entry.horse_id)}
              title="クリックでこの馬の過去走を開く"
            >
              {/* ── 識別 ── */}
              {/* 横スクロールしても «どの馬か» を見失わないよう馬番・馬名を固定 */}
              <TableCell className="sticky left-0 z-10 bg-inherit text-center">
                <Umaban n={entry.post_position} runners={runners} />
              </TableCell>
              <TableCell className="sticky left-12 z-10 bg-inherit">
                <span className="inline-flex items-center gap-1.5">
                  <ChevronRight
                    className={`h-3 w-3 shrink-0 text-subtle-foreground transition-transform ${
                      open ? 'rotate-90' : ''
                    }`}
                    aria-hidden="true"
                  />
                  {entry.horse_name ?? (
                    <span className="font-mono text-xs text-muted-foreground">
                      {entry.horse_id}
                    </span>
                  )}
                </span>
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
            </TableRow>,
            // 開いた行の下に、このレース日より前の走りを出す
            ...(open
              ? [
                  <TableRow key={`${entry.horse_id}-past`} className="bg-card-elevated/40">
                    <TableCell colSpan={20} className="px-4 py-2">
                      <HorsePastRuns horseId={entry.horse_id} before={raceDate} />
                    </TableCell>
                  </TableRow>,
                ]
              : []),
          ];
        })}
      </TableBody>
    </Table>
  );
}
