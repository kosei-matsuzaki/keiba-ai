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

/**
 * 順位 1〜3 に色を付ける。**netkeiba / JRA の馬柱と同じ 1 位=赤・2 位=青・3 位=緑**。
 *
 * 見慣れた配色に合わせるほうが速く読めるので、このアプリの意味体系より
 * 競馬側の語彙を優先している（枠色と同じ扱い＝「データの色」）。
 * `--destructive` / `--info` / `--success` を使い回すので色数は増えない。
 * 過去走の表には損益の値が無いため、赤＝マイナス収支との取り違えは起きない。
 *
 * 4 位以下と、順位が出せないとき (null) は無彩色。
 */
function rankMarker(rank: number | null | undefined): string {
  // **余白は順位に関わらず常に同じ。**色が付いた行にだけ px を足すと、
  // 右揃えの数字が色の有無で左右にずれる (馬柱として読めなくなる)。
  const base = 'rounded-sm px-1 font-medium';
  if (rank === 1) return cn(base, 'bg-destructive/20 text-destructive');
  if (rank === 2) return cn(base, 'bg-info/20 text-info');
  if (rank === 3) return cn(base, 'bg-success/20 text-success');
  return cn(base, 'text-muted-foreground');
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
    // table-fixed + 列幅の明示が要る。既定の auto レイアウトだと **馬ごとに
    // 別の table** が描かれ、その馬の中身だけで列幅が決まるため、行を 2 頭ぶん
    // 開くと着順・人気・オッズ・タイムの位置が縦に揃わない。
    // 幅を書かない「レース」列が余りを吸う。
    <table className="w-full table-fixed text-xs">
      <thead className="text-subtle-foreground">
        <tr className="text-left">
          <th className="w-24 py-1 pr-3 font-normal">日付</th>
          <th className="py-1 pr-3 font-normal">レース</th>
          <th className="w-40 py-1 pr-3 font-normal">コース</th>
          <th className="w-16 py-1 pr-3 text-right font-normal">着順</th>
          <th className="w-12 py-1 pr-3 text-right font-normal">人気</th>
          <th className="w-20 py-1 pr-3 text-right font-normal">オッズ</th>
          <th className="w-20 py-1 pr-3 text-right font-normal">タイム</th>
          <th className="w-16 py-1 pr-3 text-right font-normal">上がり</th>
          <th className="w-24 py-1 font-normal">通過</th>
        </tr>
      </thead>
      <tbody>
        {query.data.runs.map((r) => (
          <tr key={r.race_id} className="border-t border-border/60">
            <td className="py-1 pr-3 font-mono tabular-nums text-muted-foreground">{r.date}</td>
            <td className="truncate py-1 pr-3" title={r.race_name ?? ''}>
              {r.race_name ?? '·'}
              {r.race_class && (
                <span className="ml-1 text-subtle-foreground">{r.race_class}</span>
              )}
            </td>
            <td className="truncate whitespace-nowrap py-1 pr-3 text-muted-foreground">
              {r.course}
              {r.surface}
              {r.distance ? `${r.distance}m` : ''}
              {r.track_condition && (
                <span className="ml-1 text-subtle-foreground">{r.track_condition}</span>
              )}
            </td>
            <td className="py-1 pr-3 text-right font-mono tabular-nums">
              <span className={rankMarker(r.finish_position)}>{r.finish_position ?? '·'}</span>
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
            {/* 色は値ではなく **そのレースでの順位** で決める。同じ 33.8 でも
                高速馬場なら平凡、時計のかかる馬場なら最速なので、値の大小に
                色を付けても意味を持たない。 */}
            <td
              className="py-1 pr-3 text-right font-mono tabular-nums"
              title={r.agari_rank != null ? `このレースで上がり ${r.agari_rank} 位` : undefined}
            >
              <span className={rankMarker(r.agari_rank)}>
                {r.agari_3f != null ? r.agari_3f.toFixed(1) : '·'}
              </span>
            </td>
            <td
              className="truncate py-1 font-mono tabular-nums text-subtle-foreground"
              title={r.passing ?? undefined}
            >
              {r.passing ?? '·'}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
