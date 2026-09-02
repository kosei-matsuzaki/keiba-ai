import { useEffect, type ReactNode } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/cn';
import { COMBO_BET_TYPES } from '@/lib/betTypes';
import type { SettingsResponse, SettingsUpdate } from '@/types/api';

const schema = z
  .object({
    user_agent: z.string().min(1, 'User-Agent を入力してください'),
    rate_min_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    rate_max_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    night_min_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    win_min_odds: z.coerce.number().min(1.0, '1.0 以上の値を入力してください'),
    // 確率モデルの割り当ては Models 画面で行うので、ここでは扱わない。
    place_min_hit_prob: z.coerce
      .number()
      .min(0, '0 以上の値を入力してください')
      .max(1, '1 以下の値を入力してください'),
    scraper_stopped: z.boolean(),
    // 賭け金の設定は「1 レースにいくらまで」だけ。1 点 = 100 円は固定で、
    // 何点買うかは確信度が決めるので、券種ごとの金額も券種の選択も設定に無い。
    race_budget: z.coerce
      .number()
      .int('整数で入力してください')
      .min(100, '100 以上の値を入力してください'),
    // **画面は % で扱う。** 0.075 と書かせるより 7.5% の方が読み書きしやすい。
    // API は 0〜1 なので toForm / submit で 100 倍・1/100 する。
    combo_min_hit_prob: z.record(
      z.string(),
      z.coerce
        .number()
        .min(0, '0 以上の値を入力してください')
        .max(100, '100 以下の値を入力してください')
    ),
  })
  .refine((d) => d.rate_max_seconds >= d.rate_min_seconds, {
    message: 'rate_max は rate_min 以上にしてください',
    path: ['rate_max_seconds'],
  });

type FormValues = z.infer<typeof schema>;

export type SettingsSection = 'scraper' | 'betting';

interface SettingsFormProps {
  defaults: SettingsResponse;
  onSubmit: (values: SettingsUpdate) => void;
  isPending: boolean;
  /** 表示するセクション。指定されなければ全セクションを縦並びで表示する。 */
  activeSection?: SettingsSection;
}

export function SettingsForm({ defaults, onSubmit, isPending, activeSection }: SettingsFormProps) {
  const toForm = (d: SettingsResponse) => ({
    ...d,
    // 連系の下限だけ % 表示にする (0.075 → 7.5)
    combo_min_hit_prob: Object.fromEntries(
      Object.entries(d.combo_min_hit_prob ?? {}).map(([k, v]) => [k, +(v * 100).toFixed(2)])
    ),
  });

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isDirty, dirtyFields },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: toForm(defaults),
  });

  useEffect(() => {
    reset(toForm(defaults));
  }, [defaults, reset]);


  function submit(values: FormValues) {
    onSubmit({
      ...values,
      // % で受け取った値を API の 0〜1 に戻す
      combo_min_hit_prob: Object.fromEntries(
        Object.entries(values.combo_min_hit_prob ?? {}).map(([k, v]) => [k, Number(v) / 100])
      ),
    });
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
          hidden={!visible('scraper')}
        >
              <FieldRow
                label="User-Agent"
                id="user_agent"
                help="netkeiba に名乗る文字列。連絡先を入れておく"
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
          hidden={!visible('betting')}
        >
              <div className="flex flex-col gap-4">
                <FieldRow
                  label="複勝を買う確信度の下限"
                  id="place_min_hit_prob"
                  help="本命の 3 着内率がこれ未満なら複勝を見送る。0 で全レース購入"
                  error={errors.place_min_hit_prob?.message}
                >
                  <Input
                    id="place_min_hit_prob"
                    type="number"
                    step="0.05"
                    {...register('place_min_hit_prob')}
                  />
                </FieldRow>
                <FieldRow
                  label="単勝のオッズ下限"
                  id="win_min_odds"
                  help="単勝はこれ以下のオッズのときだけ見送る"
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
                  help="使ってよい上限。使い切る目標ではない"
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
                  label="連系を買う的中確率の下限"
                  id="combo_min_hit_prob"
                  help="これを超えた買い目だけ買う。超えた数だけ買うので点数はレースごとに変わる"
                  error={errors.combo_min_hit_prob?.root?.message}
                >
                  <div className="flex flex-col gap-2">
                    {COMBO_BET_TYPES.map((betType) => (
                      <div key={betType} className="flex items-center gap-2">
                        <span className="w-14 shrink-0 text-xs text-subtle-foreground">
                          {betType}
                        </span>
                        <Input
                          type="number"
                          step="0.5"
                          min="0"
                          max="100"
                          aria-label={`${betType} を買う的中確率の下限`}
                          className="text-right font-mono tabular-nums"
                          {...register(`combo_min_hit_prob.${betType}` as const)}
                        />
                        <span className="w-4 shrink-0 text-xs text-subtle-foreground">%</span>
                      </div>
                    ))}
                  </div>
                </FieldRow>
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
        <p className="text-xs text-subtle-foreground">{description}</p>
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
    <div className="grid gap-1.5 border-b border-border py-3 sm:grid-cols-[minmax(0,1fr)_20rem] sm:gap-8">
      <div>
        {/* **説明は畳まず、短く書く。** 設定値は「何を意味する数字か」が
            分からないと入力できないので、ホバーに隠すと使えない。
            長い理由や実測値は docs に置き、ここは 1 行に収める。 */}
        <Label htmlFor={id} className="text-[13px] font-medium">
          {label}
        </Label>
        {help && <p className="mt-0.5 text-xs text-subtle-foreground">{help}</p>}
      </div>
      {/* 入力は右端に揃える。説明文は長さがまちまちなので、左を伸ばして
          入力の左端が揃わないと「どこに書くか」を毎回探すことになる。 */}
      <div className="w-full sm:justify-self-end">
        {children}
        {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
      </div>
    </div>
  );
}

