import { useEffect, type ReactNode } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { CircleDollarSign, Power, Timer } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
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

  useEffect(() => {
    reset(defaults);
  }, [defaults, reset]);

  function submit(values: FormValues) {
    onSubmit(values);
  }

  return (
    <form onSubmit={handleSubmit(submit)} className="flex max-w-3xl flex-col gap-5" noValidate>
      {/* Section 1: スクレイパー */}
      <Section
        icon={Timer}
        title="スクレイパー"
        description="netkeiba へのアクセス頻度と User-Agent。レート制御を緩めると検出リスクが上がります。"
      >
        <FieldRow
          label="User-Agent"
          id="user_agent"
          help="netkeiba へ送信するブラウザ identification 文字列"
          error={errors.user_agent?.message}
        >
          <Input id="user_agent" {...register('user_agent')} />
        </FieldRow>

        <div className="grid gap-4 sm:grid-cols-3">
          <FieldRow
            label="rate_min (秒)"
            id="rate_min_seconds"
            help="リクエスト間隔の下限"
            error={errors.rate_min_seconds?.message}
          >
            <Input id="rate_min_seconds" type="number" step="0.1" {...register('rate_min_seconds')} />
          </FieldRow>
          <FieldRow
            label="rate_max (秒)"
            id="rate_max_seconds"
            help="上限 (この間でランダム jitter)"
            error={errors.rate_max_seconds?.message}
          >
            <Input id="rate_max_seconds" type="number" step="0.1" {...register('rate_max_seconds')} />
          </FieldRow>
          <FieldRow
            label="night_min (秒)"
            id="night_min_seconds"
            help="22:00–05:00 JST の最小待機"
            error={errors.night_min_seconds?.message}
          >
            <Input id="night_min_seconds" type="number" step="0.1" {...register('night_min_seconds')} />
          </FieldRow>
        </div>
      </Section>

      {/* Section 2: ベッティング EV 閾値 */}
      <Section
        icon={CircleDollarSign}
        title="ベッティング期待値"
        description="evaluate.py で「賭ける / 賭けない」を判定する閾値。1.0 が損益分岐、上げると厳選、下げると幅広く賭ける。"
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <FieldRow
            label="単勝 EV 閾値"
            id="win_ev_threshold"
            help="win_prob × odds_win がこの値超で賭け"
            error={errors.win_ev_threshold?.message}
          >
            <Input id="win_ev_threshold" type="number" step="0.01" {...register('win_ev_threshold')} />
          </FieldRow>
          <FieldRow
            label="複勝 EV 閾値"
            id="place_ev_threshold"
            help="place_prob × min_payout/100 がこの値超で賭け"
            error={errors.place_ev_threshold?.message}
          >
            <Input
              id="place_ev_threshold"
              type="number"
              step="0.01"
              {...register('place_ev_threshold')}
            />
          </FieldRow>
        </div>
      </Section>

      {/* Section 3: 運用 */}
      <Section
        icon={Power}
        title="運用"
        description="緊急停止フラグ。ON にすると進行中ジョブが ScraperStopped 例外で中断される。"
      >
        <label
          htmlFor="scraper_stopped"
          className="flex cursor-pointer items-center justify-between rounded-md border bg-card px-4 py-3 text-sm transition-colors hover:bg-accent/50"
        >
          <div>
            <div className="font-medium">スクレイパーを停止する</div>
            <div className="mt-0.5 text-xs text-muted-foreground">
              KEIBA_SCRAPER_STOP=1 と同等。CLI 経由で実行中のジョブにも反映される
            </div>
          </div>
          <input
            id="scraper_stopped"
            type="checkbox"
            className="h-4 w-4 shrink-0 rounded border-gray-300"
            {...register('scraper_stopped')}
          />
        </label>
      </Section>

      <div className="flex justify-end">
        <Button type="submit" disabled={!isDirty || isPending}>
          {isPending ? '保存中…' : '変更を保存'}
        </Button>
      </div>
    </form>
  );
}

// ── Section helper ──────────────────────────────────────────────────────────

interface SectionProps {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description?: string;
  children: ReactNode;
}

function Section({ icon: Icon, title, description, children }: SectionProps) {
  return (
    <Card>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-primary" />
          {title}
        </CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent className="space-y-4">{children}</CardContent>
    </Card>
  );
}

// ── Field helper ────────────────────────────────────────────────────────────

interface FieldRowProps {
  label: string;
  id: string;
  /** Optional small help text shown below the label */
  help?: string;
  error?: string;
  children: ReactNode;
}

function FieldRow({ label, id, help, error, children }: FieldRowProps) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id} className="text-sm font-medium">
        {label}
      </Label>
      {children}
      {help && !error && <p className="text-xs text-muted-foreground">{help}</p>}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
