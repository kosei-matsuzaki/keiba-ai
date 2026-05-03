import { useParams, Link } from 'react-router-dom';
import { Trophy } from 'lucide-react';

import { useRaceDetail } from '@/hooks/useRaceDetail';
import { usePredictions } from '@/hooks/usePredictions';
import { PredictionTable } from '@/components/PredictionTable';
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
import { formatOdds, formatYen } from '@/lib/formatters';

function RaceDetailSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-40 w-full rounded-lg" />
      <Skeleton className="h-64 w-full rounded-lg" />
      <Skeleton className="h-64 w-full rounded-lg" />
    </div>
  );
}

export function RaceDetail() {
  const { race_id = '' } = useParams<{ race_id: string }>();
  const raceQuery = useRaceDetail(race_id);
  const predQuery = usePredictions(race_id);

  if (raceQuery.isPending) {
    return (
      <div className="flex flex-col gap-6 p-6">
        <PageHeader icon={Trophy} title="Race Detail" description={race_id} />
        <RaceDetailSkeleton />
      </div>
    );
  }

  if (raceQuery.isError) {
    const is404 = isNotFoundError(raceQuery.error);
    return (
      <div className="flex flex-col gap-6 p-6">
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

  return (
    <div className="flex flex-col gap-6 p-6">
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

      {/* Entry table */}
      {race.entries.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">出走馬一覧</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>馬番</TableHead>
                  <TableHead>馬 ID</TableHead>
                  <TableHead className="text-center">年齢/性別</TableHead>
                  <TableHead className="text-right">単勝オッズ</TableHead>
                  <TableHead className="text-right">人気</TableHead>
                  <TableHead className="text-center">着順</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {race.entries.map((entry) => (
                  <TableRow key={entry.horse_id}>
                    <TableCell>{entry.post_position ?? '—'}</TableCell>
                    <TableCell className="font-mono text-xs">{entry.horse_id}</TableCell>
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
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Prediction table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">予想スコア</CardTitle>
        </CardHeader>
        <CardContent>
          {predQuery.isPending ? (
            <Skeleton className="h-40 w-full" />
          ) : predQuery.isError ? (
            <EmptyState
              message="予想データを取得できません"
              description={
                isServiceUnavailableError(predQuery.error)
                  ? 'active モデルが見つかりません。Models 画面から train を実行してください。'
                  : 'バックエンドが起動しているか確認してください。'
              }
            />
          ) : (
            <PredictionTable predictions={predQuery.data.predictions} entries={race.entries} />
          )}
        </CardContent>
      </Card>
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
      <dt className="text-muted-foreground">{label}</dt>
      <dd className={mono ? 'font-mono text-xs' : ''}>{value}</dd>
    </div>
  );
}
