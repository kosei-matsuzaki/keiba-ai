import { useState } from 'react';
import { ClipboardCheck } from 'lucide-react';

import { useRunResults } from '@/hooks/useRunResults';
import { DateYMDPicker } from '@/components/DateYMDPicker';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';

/** N 日前の日付 YYYY-MM-DD（ローカル）。 */
function isoDaysAgo(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${d.getFullYear()}-${m}-${day}`;
}

/**
 * 確定したレース（結果＋確定オッズ）を期間指定で取り込むダイアログ。
 * - 範囲内でも取込済みのレース/オッズはスキップ。
 * - 今日は未確定のため対象外（昨日まで）。
 * - 結果が db.netkeiba に未掲載の直近分は後日反映（確定オッズは先に取り込み）。
 */
export function RunResultsDialog() {
  const [open, setOpen] = useState(false);
  const [from, setFrom] = useState(() => isoDaysAgo(14));
  const [to, setTo] = useState(() => isoDaysAgo(1)); // 昨日まで

  const runResults = useRunResults();
  const isFetching = runResults.isPending || runResults.isPolling;
  const canRun = from !== '' && to !== '' && from <= to && !isFetching;

  function handleRun() {
    if (!canRun) return;
    runResults.mutate(
      { from, to },
      { onSuccess: () => setOpen(false) },
    );
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" disabled={isFetching} aria-label="結果取込">
          <ClipboardCheck className="mr-1.5 h-4 w-4" />
          {isFetching ? '結果取込中...' : '結果取込'}
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>確定レースの取込</DialogTitle>
          <DialogDescription>
            指定期間の確定したレース（出走馬・着順・払戻＋確定オッズ）を取り込みます。
            取込済みはスキップ。今日は未確定のため対象外です。
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3">
          <div className="grid gap-1.5">
            <Label>開始日</Label>
            <DateYMDPicker value={from} onChange={setFrom} ariaLabel="開始日" />
          </div>
          <div className="grid gap-1.5">
            <Label>終了日</Label>
            <DateYMDPicker value={to} onChange={setTo} ariaLabel="終了日" />
          </div>
          {from > to && (
            <p className="text-xs text-destructive">開始日は終了日以前にしてください。</p>
          )}
          <p className="text-xs text-muted-foreground">
            ※ 直近のレースは結果アーカイブ反映に数日かかる場合があります（その間は確定オッズのみ
            先に取り込み、結果は後日再実行で取得されます）。期間は最大 90 日です。
          </p>
        </div>

        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline">閉じる</Button>
          </DialogClose>
          <Button onClick={handleRun} disabled={!canRun}>
            {isFetching ? '取込中...' : '取り込む'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
