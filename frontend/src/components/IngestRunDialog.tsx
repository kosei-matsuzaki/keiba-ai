import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import type { ScraperRunRequest } from '@/types/api';

interface IngestRunDialogProps {
  onSubmit: (req: ScraperRunRequest) => void;
  isPending: boolean;
}

function todayString(): string {
  return new Date().toISOString().slice(0, 10);
}

export function IngestRunDialog({ onSubmit, isPending }: IngestRunDialogProps) {
  const [open, setOpen] = useState(false);
  const [date, setDate] = useState(todayString());
  const [limit, setLimit] = useState('');

  function handleSubmit() {
    const req: ScraperRunRequest = { date: date || undefined };
    if (limit) req.limit = Number(limit);
    onSubmit(req);
    setOpen(false);
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button disabled={isPending}>
          {isPending ? '実行中…' : '取り込みを実行'}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>スクレイピング実行</DialogTitle>
          <DialogDescription>取り込み対象日と件数上限を指定してください。</DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div className="space-y-1">
            <Label htmlFor="ingest-date">対象日 (YYYY-MM-DD)</Label>
            <Input
              id="ingest-date"
              placeholder="例: 2024-06-01"
              value={date}
              onChange={(e) => setDate(e.target.value)}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="ingest-limit">件数上限（任意）</Label>
            <Input
              id="ingest-limit"
              type="number"
              min={1}
              placeholder="例: 10"
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            キャンセル
          </Button>
          <Button onClick={handleSubmit}>実行</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
