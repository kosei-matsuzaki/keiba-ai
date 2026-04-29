import { Card, CardContent } from '@/components/ui/card';
import { Construction } from 'lucide-react';

interface PlaceholderScreenProps {
  title: string;
  note?: string;
}

export function PlaceholderScreen({ title, note = 'M7 で実装予定' }: PlaceholderScreenProps) {
  return (
    <div className="flex flex-1 flex-col gap-6 p-6">
      <h1 className="text-2xl font-bold">{title}</h1>
      <Card className="flex flex-1 flex-col items-center justify-center gap-4 py-24 text-center">
        <CardContent className="flex flex-col items-center gap-4">
          <Construction className="h-16 w-16 text-muted-foreground/40" />
          <p className="text-lg font-medium text-muted-foreground">{note}</p>
          <p className="text-sm text-muted-foreground/70">
            この画面の本実装は後続マイルストーンで対応します。
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
