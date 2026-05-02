import { type ReactNode, useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

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

const optionalDate = z
  .union([
    z.literal(''),
    z.string().regex(/^\d{4}-\d{2}-\d{2}$/, 'YYYY-MM-DD 形式で入力してください'),
  ])
  .optional();

const optionalNonNegInt = z
  .union([
    z.literal(''),
    z.coerce.number().int('整数を入力してください').min(0, '0 以上を指定してください'),
  ])
  .optional();

const schema = z.object({
  train_end: optionalDate,
  valid_months: optionalNonNegInt,
  test_months: optionalNonNegInt,
});

type FormValues = z.infer<typeof schema>;

export function TrainModelDialog({ onSubmit, isPending }: TrainModelDialogProps) {
  const [open, setOpen] = useState(false);
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isValid },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { train_end: '', valid_months: '', test_months: '' },
    mode: 'onChange',
  });

  useEffect(() => {
    if (open) reset({ train_end: '', valid_months: '', test_months: '' });
  }, [open, reset]);

  function submit(values: FormValues) {
    const req: TrainRequest = {};
    if (typeof values.train_end === 'string' && values.train_end) {
      req.train_end = values.train_end;
    }
    if (values.valid_months !== '' && values.valid_months != null) {
      req.valid_months = Number(values.valid_months);
    }
    if (values.test_months !== '' && values.test_months != null) {
      req.test_months = Number(values.test_months);
    }
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
        <form onSubmit={handleSubmit(submit)} className="space-y-4 py-2" noValidate>
          <FieldRow
            label="学習終了日 (YYYY-MM-DD)"
            id="train-end"
            error={errors.train_end?.message as string | undefined}
          >
            <Input id="train-end" placeholder="例: 2025-12-31" {...register('train_end')} />
          </FieldRow>
          <FieldRow
            label="検証期間 (月数)"
            id="valid-months"
            error={errors.valid_months?.message as string | undefined}
          >
            <Input
              id="valid-months"
              type="number"
              min={0}
              placeholder="例: 3"
              {...register('valid_months')}
            />
          </FieldRow>
          <FieldRow
            label="テスト期間 (月数)"
            id="test-months"
            error={errors.test_months?.message as string | undefined}
          >
            <Input
              id="test-months"
              type="number"
              min={0}
              placeholder="例: 3"
              {...register('test_months')}
            />
          </FieldRow>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => setOpen(false)}>
              キャンセル
            </Button>
            <Button type="submit" disabled={!isValid}>
              学習開始
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

interface FieldRowProps {
  label: string;
  id: string;
  error?: string;
  children: ReactNode;
}

function FieldRow({ label, id, error, children }: FieldRowProps) {
  return (
    <div className="space-y-1">
      <Label htmlFor={id}>{label}</Label>
      {children}
      {error && <p className="text-sm text-destructive">{error}</p>}
    </div>
  );
}
