import { useEffect, type ReactNode } from 'react';
import { useForm, useController } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/cn';
import { ALL_BET_TYPES } from '@/lib/betTypes';
import type { BetType, SettingsResponse, SettingsUpdate } from '@/types/api';

const betTypeEnum = z.enum(['単勝', '複勝', '枠連', '馬連', 'ワイド', '馬単', '三連複', '三連単']);

const schema = z
  .object({
    user_agent: z.string().min(1, 'User-Agent を入力してください'),
    rate_min_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    rate_max_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    night_min_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    win_min_odds: z.coerce.number().min(1.0, '1.0 以上の値を入力してください'),
    win_ev_threshold: z.coerce.number().min(1.0, '1.0 以上の値を入力してください'),
    scraper_stopped: z.boolean(),
    // 賭け金は「1 レースにいくらまで」と「1 点いくら」の 2 つだけ
    race_budget: z.coerce
      .number()
      .int('整数で入力してください')
      .min(100, '100 以上の値を入力してください'),
    stake_unit: z.coerce
      .number()
      .int('整数で入力してください')
      .min(100, '100 以上の値を入力してください')
      .refine((v) => v % 100 === 0, '100 円単位で入力してください'),
    enabled_bet_types: z
      .array(betTypeEnum)
      .min(1, '1 つ以上の馬券種を選択してください'),
  })
  .refine((d) => d.rate_max_seconds >= d.rate_min_seconds, {
    message: 'rate_max は rate_min 以上にしてください',
    path: ['rate_max_seconds'],
  });

type FormValues = z.infer<typeof schema>;

export type SettingsSection = 'scraper' | 'betting' | 'bet_types';

interface SettingsFormProps {
  defaults: SettingsResponse;
  onSubmit: (values: SettingsUpdate) => void;
  isPending: boolean;
  /** 表示するセクション。指定されなければ全セクションを縦並びで表示する。 */
  activeSection?: SettingsSection;
}

export function SettingsForm({ defaults, onSubmit, isPending, activeSection }: SettingsFormProps) {
  const {
    register,
    handleSubmit,
    reset,
    control,
    watch,
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

  // 予算と 1 点あたりの額から「最大何点買えるか」を出す
  const watchedBudget = watch('race_budget');
  const watchedUnit = watch('stake_unit');
  const maxPoints =
    Number.isFinite(Number(watchedBudget)) && Number(watchedUnit) > 0
      ? Math.floor(Number(watchedBudget) / Number(watchedUnit))
      : null;

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

  // activeSection 指定時はそれ以外を hidden に。指定なし (undefined) なら全表示。
  const visible = (key: SettingsSection): boolean =>
    activeSection === undefined || activeSection === key;

  return (
    <form onSubmit={handleSubmit(submit)} className="flex flex-col gap-6" noValidate>
      <div className="flex flex-col gap-6">
        <Section
          description="netkeiba へのアクセス頻度と User-Agent。レート制御を緩めると検出リスクが上がります。"
          hidden={!visible('scraper')}
        >
              <FieldRow
                label="User-Agent"
                id="user_agent"
                help="netkeiba へ送信するブラウザ identification 文字列"
                error={errors.user_agent?.message}
              >
                <Input id="user_agent" {...register('user_agent')} />
              </FieldRow>

              <div className="flex flex-col gap-4">
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
        </Section>

        <Section
          description="連系（馬連・三連複など）を「賭ける / 賭けない」判定する期待値の閾値。1.0 が損益分岐、上げると厳選、下げると幅広く賭ける。単勝・複勝は期待値ではなく AI の本命を買うルールなので、ここでは設定しません。"
          hidden={!visible('betting')}
        >
              <div className="flex flex-col gap-4">
                <FieldRow
                  label="連系を買う基準"
                  id="win_ev_threshold"
                  help="「賭けた額の何倍が期待できるか」の下限。1.00 が損益トントンで、1.10 なら 1 割の取り分が見込めるときだけ買います（的中確率 × オッズ で計算）。上げるほど買う回数は減ります。単勝・複勝には適用されません。"
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
                  label="単勝のオッズ下限"
                  id="win_min_odds"
                  help="単勝は AI の本命（予想 1 位）を買うルールです。ここを下回るオッズのときだけ見送ります。期待値で絞ると大穴に寄って回収率が落ちるため、期待値条件は使いません。"
                  error={errors.win_min_odds?.message}
                >
                  <Input
                    id="win_min_odds"
                    type="number"
                    step="0.01"
                    {...register('win_min_odds')}
                  />
                </FieldRow>
              </div>

              <div className="flex flex-col gap-4">
                {/* 賭け金は「1 レースにいくらまで」と「1 点いくら」の 2 つだけ。
                    資金比率 (Kelly) は廃止した。 */}
                <FieldRow
                  label="1 レースに使う上限"
                  id="race_budget"
                  help="1 レースに使ってよい金額の上限です。買う買い目が少なければ、ここまで使わずに終わります（1 点も買わないこともあります）。"
                  error={errors.race_budget?.message}
                >
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm text-subtle-foreground">¥</span>
                    <Input
                      id="race_budget"
                      type="number"
                      step="500"
                      min="100"
                      className="text-right font-mono tabular-nums"
                      {...register('race_budget')}
                    />
                  </div>
                </FieldRow>

                <FieldRow
                  label="1 点あたりの賭け金"
                  id="stake_unit"
                  help={`買い目 1 点あたりの金額です。馬券は 100 円単位でしか買えません。${
                    maxPoints != null ? `いまの設定なら 1 レース最大 ${maxPoints} 点まで。` : ''
                  }`}
                  error={errors.stake_unit?.message}
                >
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm text-subtle-foreground">¥</span>
                    <Input
                      id="stake_unit"
                      type="number"
                      step="100"
                      min="100"
                      className="text-right font-mono tabular-nums"
                      {...register('stake_unit')}
                    />
                  </div>
                </FieldRow>
              </div>
        </Section>

        <Section
          description="ふだん買う馬券の種類。ここで外した券種は推奨買目に出ません（レースごとに変えたいときは、レース詳細の「この予想の条件」で切り替えられます）。"
          hidden={!visible('bet_types')}
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
                          'flex h-9 items-center justify-center rounded-sm border text-sm transition-colors',
                          isSelected
                            ? 'border-primary bg-primary/10 text-primary'
                            : 'border-border bg-transparent text-subtle-foreground hover:border-border-strong hover:text-foreground',
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
        </Section>

      </div>

      {/* Sticky footer — 変更があるときだけ出す。
          常時出ていると「未保存かどうか」という情報そのものが失われる。 */}
      {(isDirty || isPending) && (
        <div className="sticky bottom-0 z-10 -mx-6 flex items-center justify-end gap-3 border-t border-border bg-background px-6 py-3">
          <span className="text-sm text-muted-foreground">
            {dirtyCount} 件の変更があります
          </span>
          <Button type="submit" disabled={!isDirty || isPending}>
            {isPending ? '保存中…' : '変更を保存'}
          </Button>
        </div>
      )}
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

// ── Section: タブ配下のフラット section (Card なし、タイトルなし) ──────────
// タブ名がそのままセクションのタイトルとして機能するので、カード/タイトルの
// 重複を避けてフラットに並べる。description のみ muted で 1 行表示。

interface SectionProps {
  /** 補足説明 (1 行 muted)。省略可能。 */
  description?: string;
  children: ReactNode;
  /** true のとき表示せず DOM には残す (form state を維持するため) */
  hidden?: boolean;
}

function Section({ description, children, hidden = false }: SectionProps) {
  return (
    <div className={cn('flex flex-col gap-4', hidden && 'hidden')}>
      {description && (
        <p className="text-sm leading-relaxed text-muted-foreground">{description}</p>
      )}
      {/* 行の区切りは FieldRow 側の border-b が持つので、ここでは間隔を空けない */}
      <div>{children}</div>
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

/**
 * 設定 1 項目。縦積みではなくラベルを左段に置いた 2 カラムにして、
 * 「フォーム」ではなく「仕様書」に見せる。行の区切りは罫線。
 */
function FieldRow({ label, id, help, error, children }: FieldRowProps) {
  return (
    <div className="grid gap-1.5 border-b border-border py-4 sm:grid-cols-[14rem_minmax(0,1fr)] sm:gap-6">
      <div>
        <Label htmlFor={id} className="text-[13px] font-medium">
          {label}
        </Label>
        {help && !error && (
          <p className="mt-1 text-xs leading-relaxed text-subtle-foreground">{help}</p>
        )}
      </div>
      <div className="max-w-md">
        {children}
        {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
      </div>
    </div>
  );
}

