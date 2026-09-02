import { useQuery } from '@tanstack/react-query';

import { fetchHorseHistory } from '@/lib/api';
import { formatOdds } from '@/lib/formatters';
import { cn } from '@/lib/cn';

/**
 * その馬の**このレースより前**の成績。出走馬一覧の行を開くと出る。
 *
 * AI が履歴 GRU で食べているのと同じ範囲 (前走まで) を人も見られるようにする。
 * 当日以降は返らない (API 側で `before` より厳密に過去だけに絞っている)。
 *
 * 列は「いつ・どこで・どう走ったか」に絞る。馬体重や斤量まで並べると、
 * 出走馬一覧の中に 2 つ目の表ができてしまい、開いた意味が薄れる。
 */
interface HorsePastRunsProps {
  horseId: string;
  /** このレースの日。これより前の走りだけを出す。 */
  before: string;
  /** 何走まで出すか。 */
  limit?: number;
}

/** 1〜3 着は色を付ける。着順は「勝ったか」が最初に読みたい情報。 */
function finishClass(pos: number | null): string {
  if (pos === 1) return 'text-primary font-medium';
  if (pos != null && pos <= 3) return 'text-foreground';
  return 'text-muted-foreground';
}

/** 秒 → "1:34.2"。競馬の走破時計は分秒表記でないと読めない。 */
function formatRaceTime(sec: number | null): string {
  if (sec == null || !Number.isFinite(sec)) return '·';
  const m = Math.floor(sec / 60);
  const rest = sec - m * 60;
  return m > 0 ? `${m}:${rest.toFixed(1).padStart(4, '0')}` : rest.toFixed(1);
}

export function HorsePastRuns({ horseId, before, limit = 5 }: HorsePastRunsProps) {
  const query = useQuery({
    queryKey: ['horse-history', horseId, before, limit],
    queryFn: () => fetchHorseHistory(horseId, { before, limit }),
    staleTime: 5 * 60 * 1000,
  });

  if (query.isPending) {
    return <p className="py-2 text-xs text-muted-foreground">過去走を読み込み中…</p>;
  }
  if (query.isError) {
    return <p className="py-2 text-xs text-destructive">過去走を取得できませんでした</p>;
  }
  if (query.data.runs.length === 0) {
    return (
      <p className="py-2 text-xs text-muted-foreground">
        この日より前の出走記録がありません（初出走、または未取込）。
      </p>
    );
  }

  return (
    <table className="w-full text-xs">
      <thead className="text-subtle-foreground">
        <tr className="text-left">
          <th className="py-1 pr-3 font-normal">日付</th>
          <th className="py-1 pr-3 font-normal">レース</th>
          <th className="py-1 pr-3 font-normal">コース</th>
          <th className="py-1 pr-3 text-right font-normal">着順</th>
          <th className="py-1 pr-3 text-right font-normal">人気</th>
          <th className="py-1 pr-3 text-right font-normal">オッズ</th>
          <th className="py-1 pr-3 text-right font-normal">タイム</th>
          <th className="py-1 pr-3 text-right font-normal">上がり</th>
          <th className="py-1 font-normal">通過</th>
        </tr>
      </thead>
      <tbody>
        {query.data.runs.map((r) => (
          <tr key={r.race_id} className="border-t border-border/60">
            <td className="py-1 pr-3 font-mono tabular-nums text-muted-foreground">{r.date}</td>
            <td className="max-w-[14rem] truncate py-1 pr-3" title={r.race_name ?? ''}>
              {r.race_name ?? '·'}
              {r.race_class && (
                <span className="ml-1 text-subtle-foreground">{r.race_class}</span>
              )}
            </td>
            <td className="whitespace-nowrap py-1 pr-3 text-muted-foreground">
              {r.course}
              {r.surface}
              {r.distance ? `${r.distance}m` : ''}
              {r.track_condition && (
                <span className="ml-1 text-subtle-foreground">{r.track_condition}</span>
              )}
            </td>
            <td className={cn('py-1 pr-3 text-right font-mono tabular-nums', finishClass(r.finish_position))}>
              {r.finish_position ?? '·'}
              {r.n_runners ? (
                <span className="text-subtle-foreground">/{r.n_runners}</span>
              ) : null}
            </td>
            <td className="py-1 pr-3 text-right font-mono tabular-nums text-muted-foreground">
              {r.popularity ?? '·'}
            </td>
            <td className="py-1 pr-3 text-right font-mono tabular-nums text-muted-foreground">
              {r.odds_win != null ? formatOdds(r.odds_win) : '·'}
            </td>
            <td className="py-1 pr-3 text-right font-mono tabular-nums text-muted-foreground">
              {formatRaceTime(r.finish_time)}
            </td>
            <td className="py-1 pr-3 text-right font-mono tabular-nums text-muted-foreground">
              {r.agari_3f != null ? r.agari_3f.toFixed(1) : '·'}
            </td>
            <td className="py-1 font-mono tabular-nums text-subtle-foreground">
              {r.passing ?? '·'}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
