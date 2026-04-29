import { useEffect, type ReactNode } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import type { SettingsResponse, SettingsUpdate } from '@/types/api';

const schema = z
  .object({
    user_agent: z.string().min(1, 'User-Agent を入力してください'),
    rate_min_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    rate_max_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    night_min_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    win_ev_threshold: z.coerce.number().min(1.0, '1.0 以上の値を入力してください'),
    place_ev_threshold: z.coerce.number().min(1.0, '1.0 以上の値を入力してください'),
    scraper_stopped: z.boolean(),
  })
  .refine((d) => d.rate_max_seconds >= d.rate_min_seconds, {
    message: 'rate_max は rate_min 以上にしてください',
    path: ['rate_max_seconds'],
  });

type FormValues = z.infer<typeof schema>;

interface SettingsFormProps {
  defaults: SettingsResponse;
  onSubmit: (values: SettingsUpdate) => void;
  isPending: boolean;
}

export function SettingsForm({ defaults, onSubmit, isPending }: SettingsFormProps) {
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isDirty },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: defaults,
  });

  // Sync when remote data changes (e.g., after successful save)
  useEffect(() => {
    reset(defaults);
  }, [defaults, reset]);

  function submit(values: FormValues) {
    onSubmit(values);
  }

  return (
    <form onSubmit={handleSubmit(submit)} className="space-y-5" noValidate>
      <FieldRow
        label="User-Agent"
        id="user_agent"
        error={errors.user_agent?.message}
      >
        <Input id="user_agent" {...register('user_agent')} />
      </FieldRow>

      <FieldRow
        label="rate_min_seconds（最小待機秒）"
        id="rate_min_seconds"
        error={errors.rate_min_seconds?.message}
      >
        <Input id="rate_min_seconds" type="number" step="0.1" {...register('rate_min_seconds')} />
      </FieldRow>

      <FieldRow
        label="rate_max_seconds（最大待機秒）"
        id="rate_max_seconds"
        error={errors.rate_max_seconds?.message}
      >
        <Input id="rate_max_seconds" type="number" step="0.1" {...register('rate_max_seconds')} />
      </FieldRow>

      <FieldRow
        label="night_min_seconds（夜間最小待機秒）"
        id="night_min_seconds"
        error={errors.night_min_seconds?.message}
      >
        <Input id="night_min_seconds" type="number" step="0.1" {...register('night_min_seconds')} />
      </FieldRow>

      <FieldRow
        label="win_ev_threshold（単勝 EV 閾値 ≥ 1.0）"
        id="win_ev_threshold"
        error={errors.win_ev_threshold?.message}
      >
        <Input id="win_ev_threshold" type="number" step="0.01" {...register('win_ev_threshold')} />
      </FieldRow>

      <FieldRow
        label="place_ev_threshold（複勝 EV 閾値 ≥ 1.0）"
        id="place_ev_threshold"
        error={errors.place_ev_threshold?.message}
      >
        <Input
          id="place_ev_threshold"
          type="number"
          step="0.01"
          {...register('place_ev_threshold')}
        />
      </FieldRow>

      <div className="flex items-center gap-3">
        <input
          id="scraper_stopped"
          type="checkbox"
          className="h-4 w-4 rounded border-gray-300"
          {...register('scraper_stopped')}
        />
        <Label htmlFor="scraper_stopped">スクレイパーを停止する（scraper_stopped）</Label>
      </div>

      <Button type="submit" disabled={!isDirty || isPending}>
        {isPending ? '保存中…' : '変更を保存'}
      </Button>
    </form>
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
