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
import type { ScraperRunRequest } from '@/types/api';

interface IngestRunDialogProps {
  onSubmit: (req: ScraperRunRequest) => void;
  isPending: boolean;
}

const schema = z.object({
  date: z
    .string()
    .regex(/^\d{4}-\d{2}-\d{2}$/, 'YYYY-MM-DD 形式で入力してください'),
  // input は string で来るので coerce.number で変換し、optional 扱い
  limit: z
    .union([
      z.literal(''),
      z.coerce.number().int('整数を入力してください').min(1, '1 以上を指定してください'),
    ])
    .optional(),
});

type FormValues = z.infer<typeof schema>;

function todayString(): string {
  return new Date().toISOString().slice(0, 10);
}

export function IngestRunDialog({ onSubmit, isPending }: IngestRunDialogProps) {
  const [open, setOpen] = useState(false);
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isValid },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { date: todayString(), limit: '' },
    mode: 'onChange',
  });

  // ダイアログを開き直したら初期値に戻す
  useEffect(() => {
    if (open) reset({ date: todayString(), limit: '' });
  }, [open, reset]);

  function submit(values: FormValues) {
    const req: ScraperRunRequest = { date: values.date };
    if (values.limit !== '' && values.limit != null) req.limit = Number(values.limit);
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
        <form onSubmit={handleSubmit(submit)} className="space-y-4 py-2" noValidate>
          <FieldRow label="対象日 (YYYY-MM-DD)" id="ingest-date" error={errors.date?.message}>
            <Input id="ingest-date" placeholder="例: 2024-06-01" {...register('date')} />
          </FieldRow>
          <FieldRow
            label="件数上限（任意）"
            id="ingest-limit"
            error={errors.limit?.message as string | undefined}
          >
            <Input
              id="ingest-limit"
              type="number"
              min={1}
              placeholder="例: 10"
              {...register('limit')}
            />
          </FieldRow>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => setOpen(false)}>
              キャンセル
            </Button>
            <Button type="submit" disabled={!isValid}>
              実行
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
