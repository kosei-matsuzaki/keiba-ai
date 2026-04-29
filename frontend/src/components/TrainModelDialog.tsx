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
import type { TrainRequest } from '@/types/api';

interface TrainModelDialogProps {
  onSubmit: (req: TrainRequest) => void;
  isPending: boolean;
}

export function TrainModelDialog({ onSubmit, isPending }: TrainModelDialogProps) {
  const [open, setOpen] = useState(false);
  const [trainEnd, setTrainEnd] = useState('');
  const [validMonths, setValidMonths] = useState('');
  const [testMonths, setTestMonths] = useState('');

  function handleSubmit() {
    const req: TrainRequest = {};
    if (trainEnd) req.train_end = trainEnd;
    if (validMonths) req.valid_months = Number(validMonths);
    if (testMonths) req.test_months = Number(testMonths);
    onSubmit(req);
    setOpen(false);
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button disabled={isPending}>
          {isPending ? '再学習中…' : '再学習を実行'}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>モデル再学習</DialogTitle>
          <DialogDescription>
            学習パラメータを入力してください（すべてオプション）。
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div className="space-y-1">
            <Label htmlFor="train-end">学習終了日 (YYYY-MM-DD)</Label>
            <Input
              id="train-end"
              placeholder="例: 2025-12-31"
              value={trainEnd}
              onChange={(e) => setTrainEnd(e.target.value)}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="valid-months">検証期間 (月数)</Label>
            <Input
              id="valid-months"
              type="number"
              min={1}
              placeholder="例: 3"
              value={validMonths}
              onChange={(e) => setValidMonths(e.target.value)}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="test-months">テスト期間 (月数)</Label>
            <Input
              id="test-months"
              type="number"
              min={1}
              placeholder="例: 3"
              value={testMonths}
              onChange={(e) => setTestMonths(e.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            キャンセル
          </Button>
          <Button onClick={handleSubmit}>学習開始</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
