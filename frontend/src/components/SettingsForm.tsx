import { useEffect, useState, type ReactNode } from 'react';
import { useForm, useController } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  CircleDollarSign,
  Layers,
  Power,
  Timer,
  type LucideIcon,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { cn } from '@/lib/cn';
import type { BetType, SettingsResponse, SettingsUpdate } from '@/types/api';

const ALL_BET_TYPES: BetType[] = [
  '単勝',
  '複勝',
  '枠連',
  '馬連',
  'ワイド',
  '馬単',
  '三連複',
  '三連単',
];

const betTypeEnum = z.enum(['単勝', '複勝', '枠連', '馬連', 'ワイド', '馬単', '三連複', '三連単']);

const schema = z
  .object({
    user_agent: z.string().min(1, 'User-Agent を入力してください'),
    rate_min_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    rate_max_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    night_min_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    win_ev_threshold: z.coerce.number().min(1.0, '1.0 以上の値を入力してください'),
    place_ev_threshold: z.coerce.number().min(1.0, '1.0 以上の値を入力してください'),
    scraper_stopped: z.boolean(),
    bankroll: z.coerce
      .number()
      .int('整数で入力してください')
      .min(100, '100 以上の値を入力してください'),
    kelly_fraction: z.coerce
      .number()
      .gt(0, '0 より大きい値を入力してください')
      .max(1, '1 以下の値を入力してください'),
    max_stake_per_race_pct: z.coerce
      .number()
      .gt(0, '0 より大きい値を入力してください')
      .max(1, '1 以下の値を入力してください'),
    enabled_bet_types: z
      .array(betTypeEnum)
      .min(1, '1 つ以上の馬券種を選択してください'),
  })
  .refine((d) => d.rate_max_seconds >= d.rate_min_seconds, {
    message: 'rate_max は rate_min 以上にしてください',
    path: ['rate_max_seconds'],
  });

type FormValues = z.infer<typeof schema>;

type SectionKey = 'scraper' | 'betting' | 'bet_types' | 'ops';

interface SectionMeta {
  key: SectionKey;
  icon: LucideIcon;
  label: string;
  description: string;
}

const SECTIONS: SectionMeta[] = [
  {
    key: 'scraper',
    icon: Timer,
    label: 'スクレイパー',
    description: 'netkeiba へのアクセス頻度と User-Agent',
  },
  {
    key: 'betting',
    icon: CircleDollarSign,
    label: 'ベッティング',
    description: 'EV 閾値と Kelly 資金配分',
  },
  {
    key: 'bet_types',
    icon: Layers,
    label: '買い方ターゲット',
    description: '対象とする馬券種の選択',
  },
  {
    key: 'ops',
    icon: Power,
    label: '運用',
    description: 'スクレイパー停止フラグ',
  },
];

interface SettingsFormProps {
  defaults: SettingsResponse;
  onSubmit: (values: SettingsUpdate) => void;
  isPending: boolean;
}

export function SettingsForm({ defaults, onSubmit, isPending }: SettingsFormProps) {
  const [activeSection, setActiveSection] = useState<SectionKey>('scraper');
  const {
    register,
    handleSubmit,
    reset,
    control,
    formState: { errors, isDirty, dirtyFields },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      ...defaults,
      enabled_bet_types: [...defaults.enabled_bet_types],
    },
  });

  useEffect(() => {
    reset({
      ...defaults,
      enabled_bet_types: [...defaults.enabled_bet_types],
    });
  }, [defaults, reset]);

  const { field: enabledBetTypesField } = useController({
    name: 'enabled_bet_types',
    control,
  });

  const { field: scraperStoppedField } = useController({
    name: 'scraper_stopped',
    control,
  });

  function toggleBetType(betType: BetType) {
    const current = enabledBetTypesField.value;
    const next = current.includes(betType)
      ? current.filter((t) => t !== betType)
      : [...current, betType];
    enabledBetTypesField.onChange(next);
  }

  function submit(values: FormValues) {
    onSubmit(values);
  }

  // dirty 件数をフッターに表示
  const dirtyCount = countDirtyFields(dirtyFields);

  return (
    <form onSubmit={handleSubmit(submit)} className="flex flex-col gap-6" noValidate>
      <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
        {/* Left vertical nav */}
        <nav
          aria-label="設定セクション"
          className="flex shrink-0 flex-row gap-1 overflow-x-auto rounded-lg border bg-card p-2 lg:w-64 lg:flex-col lg:gap-1"
        >
          {SECTIONS.map((s) => {
            const Icon = s.icon;
            const isActive = activeSection === s.key;
            return (
              <button
                key={s.key}
                type="button"
                onClick={() => setActiveSection(s.key)}
                className={cn(
                  'flex shrink-0 items-start gap-3 rounded-md px-3 py-2.5 text-left text-sm transition-colors',
                  isActive
                    ? 'bg-primary/10 text-primary'
                    : 'text-foreground hover:bg-accent',
                )}
              >
                <Icon
                  className={cn(
                    'mt-0.5 h-4 w-4 shrink-0',
                    isActive ? 'text-primary' : 'text-muted-foreground',
                  )}
                />
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span
                    className={cn(
                      'font-medium leading-tight',
                      isActive ? 'text-primary' : '',
                    )}
                  >
                    {s.label}
                  </span>
                  <span className="hidden text-xs text-muted-foreground lg:block">
                    {s.description}
                  </span>
                </div>
              </button>
            );
          })}
        </nav>

        {/* Right content panel */}
        <div className="min-w-0 flex-1 rounded-lg border bg-card p-6">
          {activeSection === 'scraper' && (
            <SectionPanel
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
                  <Input
                    id="rate_min_seconds"
                    type="number"
                    step="0.1"
                    {...register('rate_min_seconds')}
                  />
                </FieldRow>
                <FieldRow
                  label="rate_max (秒)"
                  id="rate_max_seconds"
                  help="上限 (この間でランダム jitter)"
                  error={errors.rate_max_seconds?.message}
                >
                  <Input
                    id="rate_max_seconds"
                    type="number"
                    step="0.1"
                    {...register('rate_max_seconds')}
                  />
                </FieldRow>
                <FieldRow
                  label="night_min (秒)"
                  id="night_min_seconds"
                  help="22:00–05:00 JST の最小待機"
                  error={errors.night_min_seconds?.message}
                >
                  <Input
                    id="night_min_seconds"
                    type="number"
                    step="0.1"
                    {...register('night_min_seconds')}
                  />
                </FieldRow>
              </div>
            </SectionPanel>
          )}

          {activeSection === 'betting' && (
            <SectionPanel
              title="ベッティング期待値"
              description="evaluate.py で「賭ける / 賭けない」を判定する閾値と Kelly 資金配分。1.0 が損益分岐、上げると厳選、下げると幅広く賭ける。"
            >
              <div className="grid gap-4 sm:grid-cols-2">
                <FieldRow
                  label="単勝 EV 閾値"
                  id="win_ev_threshold"
                  help="win_prob × odds_win がこの値超で賭け"
                  error={errors.win_ev_threshold?.message}
                >
                  <Input
                    id="win_ev_threshold"
                    type="number"
                    step="0.01"
                    {...register('win_ev_threshold')}
                  />
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

              <div className="grid gap-4 sm:grid-cols-3">
                <FieldRow
                  label="バンクロール (円)"
                  id="bankroll"
                  help="運用資金の総額。Kelly 計算の基準となる"
                  error={errors.bankroll?.message}
                >
                  <Input
                    id="bankroll"
                    type="number"
                    step="100"
                    {...register('bankroll')}
                  />
                </FieldRow>
                <FieldRow
                  label="Kelly 分率"
                  id="kelly_fraction"
                  help="Kelly 配分の割合 (0.25 = 1/4 Kelly)"
                  error={errors.kelly_fraction?.message}
                >
                  <Input
                    id="kelly_fraction"
                    type="number"
                    step="0.05"
                    min="0.01"
                    max="1"
                    {...register('kelly_fraction')}
                  />
                </FieldRow>
                <FieldRow
                  label="1 レース最大賭け率"
                  id="max_stake_per_race_pct"
                  help="バンクロールに対する 1 レースあたりの上限比率 (0.05 = 5%)"
                  error={errors.max_stake_per_race_pct?.message}
                >
                  <Input
                    id="max_stake_per_race_pct"
                    type="number"
                    step="0.01"
                    min="0.01"
                    max="1"
                    {...register('max_stake_per_race_pct')}
                  />
                </FieldRow>
              </div>
            </SectionPanel>
          )}

          {activeSection === 'bet_types' && (
            <SectionPanel
              title="買い方ターゲット"
              description="推奨買目と evaluate.py の賭け判定で対象とする馬券種。チェックを外した券種は賭け対象から除外される。"
            >
              <div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-8">
                  {ALL_BET_TYPES.map((betType) => {
                    const isSelected = enabledBetTypesField.value.includes(betType);
                    return (
                      <button
                        key={betType}
                        type="button"
                        onClick={() => toggleBetType(betType)}
                        aria-pressed={isSelected}
                        className={cn(
                          'flex h-9 items-center justify-center rounded-full border text-sm font-medium transition-all active:scale-[0.97]',
                          isSelected
                            ? 'border-primary bg-primary/15 text-primary hover:bg-primary/25'
                            : 'border-border bg-card text-muted-foreground hover:border-border-strong hover:text-foreground',
                        )}
                      >
                        {betType}
                      </button>
                    );
                  })}
                </div>
                {errors.enabled_bet_types?.message && (
                  <p className="mt-2 text-xs text-destructive">
                    {errors.enabled_bet_types.message}
                  </p>
                )}
              </div>
            </SectionPanel>
          )}

          {activeSection === 'ops' && (
            <SectionPanel
              title="運用"
              description="緊急停止フラグ。ON にすると進行中ジョブが ScraperStopped 例外で中断される。"
            >
              <label
                htmlFor="scraper_stopped"
                className="flex cursor-pointer items-center justify-between gap-4 rounded-lg border border-border/60 bg-card px-4 py-3 text-sm transition-colors hover:bg-card-elevated/40"
              >
                <div className="min-w-0">
                  <div className="font-medium">スクレイパーを停止する</div>
                  <div className="mt-0.5 text-xs text-muted-foreground">
                    KEIBA_SCRAPER_STOP=1 と同等。CLI 経由で実行中のジョブにも反映される
                  </div>
                </div>
                <Switch
                  id="scraper_stopped"
                  checked={scraperStoppedField.value}
                  onCheckedChange={scraperStoppedField.onChange}
                />
              </label>
            </SectionPanel>
          )}
        </div>
      </div>

      {/* Sticky footer */}
      <div className="sticky bottom-0 z-10 -mx-6 flex items-center justify-end gap-3 border-t bg-background/95 px-6 py-3 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <span className="text-sm text-muted-foreground">
          {dirtyCount > 0 ? `${dirtyCount} 件の変更があります` : '変更なし'}
        </span>
        <Button type="submit" disabled={!isDirty || isPending}>
          {isPending ? '保存中…' : '変更を保存'}
        </Button>
      </div>
    </form>
  );
}

function countDirtyFields(dirty: object): number {
  let count = 0;
  for (const v of Object.values(dirty)) {
    if (typeof v === 'boolean' && v) count += 1;
    else if (Array.isArray(v) && v.some(Boolean)) count += 1;
    else if (typeof v === 'object' && v !== null) count += countDirtyFields(v);
  }
  return count;
}

// ── SectionPanel: 中央 panel の見出し + body ───────────────────────────────

interface SectionPanelProps {
  title: string;
  description?: string;
  children: ReactNode;
}

function SectionPanel({ title, description, children }: SectionPanelProps) {
  return (
    <div className="flex flex-col gap-5">
      <div>
        <h2 className="text-lg font-semibold">{title}</h2>
        {description && (
          <p className="mt-1 text-sm text-muted-foreground">{description}</p>
        )}
      </div>
      <div className="space-y-5">{children}</div>
    </div>
  );
}

// ── FieldRow ────────────────────────────────────────────────────────────────

interface FieldRowProps {
  label: string;
  id: string;
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

