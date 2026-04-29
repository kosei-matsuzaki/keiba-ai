import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import type { RaceSummary } from '@/types/api';

interface RaceCardProps {
  race: RaceSummary;
}

const surfaceLabel: Record<string, string> = {
  芝: '芝',
  ダ: 'ダート',
};

export function RaceCard({ race }: RaceCardProps) {
  const navigate = useNavigate();

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-base leading-snug">
            {race.course} {race.race_class ?? ''}
          </CardTitle>
          {race.race_class && (
            <Badge variant="secondary" className="shrink-0">
              {race.race_class}
            </Badge>
          )}
        </div>
        <p className="text-sm text-muted-foreground">{race.date}</p>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-3 gap-x-4 gap-y-1 text-sm">
          <dt className="text-muted-foreground">馬場</dt>
          <dd className="col-span-2">{surfaceLabel[race.surface] ?? race.surface}</dd>
          <dt className="text-muted-foreground">距離</dt>
          <dd className="col-span-2">{race.distance}m</dd>
          <dt className="text-muted-foreground">頭数</dt>
          <dd className="col-span-2">{race.n_runners ?? '—'}頭</dd>
        </dl>
        <Button
          size="sm"
          variant="outline"
          className="mt-4 w-full"
          onClick={() => navigate(`/races/${race.race_id}`)}
        >
          予想を見る
        </Button>
      </CardContent>
    </Card>
  );
}
