import { useMemo } from 'react';
import { useParams, useSearchParams, Link } from 'react-router-dom';
import { Trophy, ChevronLeft } from 'lucide-react';

import { useRaceDetail } from '@/hooks/useRaceDetail';
import { usePredictions } from '@/hooks/usePredictions';
import { useRecommendations } from '@/hooks/useRecommendations';
import { RecommendationsCard } from '@/components/RecommendationsCard';
import { EmptyState } from '@/components/EmptyState';
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
import { isNotFoundError, isServiceUnavailableError } from '@/lib/api';
import { formatOdds, formatPercent, formatScore, formatYen } from '@/lib/formatters';
import type { EntrySummary, HorsePrediction } from '@/types/api';

function RaceDetailSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-40 w-full rounded-lg" />
      <Skeleton className="h-64 w-full rounded-lg" />
      <Skeleton className="h-64 w-full rounded-lg" />
    </div>
  );
}

/** Indicates BUY when single-win expected value > 1.1 */
function isBuy(pred: HorsePrediction, entry: EntrySummary | undefined): boolean {
  if (!entry?.odds_win) return false;
  return pred.win_prob * entry.odds_win > 1.1;
}

interface EntryRow {
  entry: EntrySummary;
  pred: HorsePrediction | null;
}

interface EntryPredictionTableProps {
  entries: EntrySummary[];
  predictions: HorsePrediction[] | null;
}

/**
 * Unified table merging entry data and prediction scores.
 * Sorted by prediction score desc when predictions are available,
 * otherwise by post_position asc.
 */
function EntryPredictionTable({ entries, predictions }: EntryPredictionTableProps) {
  const rows = useMemo<EntryRow[]>(() => {
    const predMap = new Map(predictions?.map((p) => [p.horse_id, p]) ?? []);

    const merged: EntryRow[] = entries.map((entry) => ({
      entry,
      pred: predMap.get(entry.horse_id) ?? null,
    }));

    if (predictions) {
      merged.sort((a, b) => (b.pred?.score ?? -Infinity) - (a.pred?.score ?? -Infinity));
    } else {
      merged.sort((a, b) => (a.entry.post_position ?? 99) - (b.entry.post_position ?? 99));
    }

    return merged;
  }, [entries, predictions]);

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-12">馬番</TableHead>
          <TableHead>馬名</TableHead>
          <TableHead className="text-center">年齢/性別</TableHead>
          <TableHead className="text-right">単勝オッズ</TableHead>
          <TableHead className="text-right">人気</TableHead>
          <TableHead className="text-center">着順</TableHead>
          <TableHead className="text-right">スコア</TableHead>
          <TableHead className="text-right">単勝確率</TableHead>
          <TableHead className="text-right">複勝確率</TableHead>
          <TableHead className="text-center">推奨</TableHead>
          <TableHead>SHAP</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map(({ entry, pred }) => (
          <TableRow key={entry.horse_id}>
            <TableCell className="font-medium">{entry.post_position ?? '—'}</TableCell>
            <TableCell>
              {entry.horse_name ?? (
                <span className="font-mono text-xs text-muted-foreground">{entry.horse_id}</span>
              )}
            </TableCell>
            <TableCell className="text-center">
              {entry.age ?? '?'}{entry.sex ?? ''}
            </TableCell>
            <TableCell className="text-right">{formatOdds(entry.odds_win)}</TableCell>
            <TableCell className="text-right">{entry.popularity ?? '—'}</TableCell>
            <TableCell className="text-center">
              {entry.finish_position != null ? (
                <Badge variant={entry.finish_position <= 3 ? 'default' : 'secondary'}>
                  {entry.finish_position}着
                </Badge>
              ) : (
                '—'
              )}
            </TableCell>
            {/* Prediction columns */}
            <TableCell className="text-right">
              {pred != null ? formatScore(pred.score) : '—'}
            </TableCell>
            <TableCell className="text-right">
              {pred != null ? formatPercent(pred.win_prob) : '—'}
            </TableCell>
            <TableCell className="text-right">
              {pred != null ? formatPercent(pred.place_prob) : '—'}
            </TableCell>
            <TableCell className="text-center">
              {pred != null && isBuy(pred, entry) && (
                <Badge variant="success">BUY</Badge>
              )}
            </TableCell>
            <TableCell className="text-xs text-muted-foreground italic">
              {pred != null ? 'SHAP 寄与は M9 以降' : '—'}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
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
      <dt className="text-muted-foreground">{label}</dt>
      <dd className={mono ? 'font-mono text-xs' : ''}>{value}</dd>
    </div>
  );
}

export function RaceDetail() {
  const { race_id = '' } = useParams<{ race_id: string }>();
  const [searchParams] = useSearchParams();
  const dateParam = searchParams.get('date');

  const raceQuery = useRaceDetail(race_id);
  const predQuery = usePredictions(race_id);
  const recQuery = useRecommendations(
    race_id,
    Boolean(race_id) && !raceQuery.isPending && !raceQuery.isError,
  );

  const backLink = dateParam ? `/past?date=${dateParam}` : '/past';

  if (raceQuery.isPending) {
    return (
      <div className="flex flex-col gap-6 p-6">
        <BackLink to={backLink} />
        <PageHeader icon={Trophy} title="Race Detail" description={race_id} />
        <RaceDetailSkeleton />
      </div>
    );
  }

  if (raceQuery.isError) {
    const is404 = isNotFoundError(raceQuery.error);
    return (
      <div className="flex flex-col gap-6 p-6">
        <BackLink to={backLink} />
        <PageHeader icon={Trophy} title="Race Detail" description={race_id} />
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

  const race = raceQuery.data;
  const predictions = predQuery.data?.predictions ?? null;

  return (
    <div className="flex flex-col gap-6 p-6">
      <BackLink to={backLink} />

      <PageHeader
        icon={Trophy}
        title={`${race.course} ${race.race_class ?? ''}`.trim()}
        description={`${race.date}・${race.surface}${race.distance}m・${race.race_id}`}
      />

      {/* Race overview */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">レース概要</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3 lg:grid-cols-4">
            <MetaItem label="レース ID" value={race.race_id} mono />
            <MetaItem label="開催日" value={race.date} />
            <MetaItem label="競馬場" value={race.course} />
            <MetaItem label="馬場種別" value={race.surface} />
            <MetaItem label="距離" value={race.distance ? `${race.distance} m` : '—'} />
            <MetaItem label="天候" value={race.weather ?? '—'} />
            <MetaItem label="馬場状態" value={race.track_condition ?? '—'} />
            <MetaItem label="クラス" value={race.race_class ?? '—'} />
            <MetaItem label="出走頭数" value={race.n_runners?.toString() ?? '—'} />
            <MetaItem label="単勝払戻" value={race.payout_win != null ? formatYen(race.payout_win) : '—'} />
            <MetaItem label="複勝払戻" value={race.payout_place ?? '—'} />
          </dl>
        </CardContent>
      </Card>

      {/* Unified entry + prediction table */}
      {race.entries.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">出走馬一覧</CardTitle>
          </CardHeader>
          <CardContent>
            {predQuery.isPending ? (
              <Skeleton className="h-40 w-full" />
            ) : predQuery.isError ? (
              <>
                <p className="mb-3 text-sm text-muted-foreground">
                  {isServiceUnavailableError(predQuery.error)
                    ? 'active モデルが見つかりません。予想スコア列は非表示です。'
                    : '予想データを取得できません。予想スコア列は非表示です。'}
                </p>
                <EntryPredictionTable entries={race.entries} predictions={null} />
              </>
            ) : (
              <EntryPredictionTable entries={race.entries} predictions={predictions} />
            )}
          </CardContent>
        </Card>
      )}

      {/* Recommendations card */}
      <RecommendationsCard
        raceId={race_id}
        data={recQuery.data}
        isPending={recQuery.isPending}
        isError={recQuery.isError}
        error={recQuery.error}
      />
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
