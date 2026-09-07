import { useEffect, useMemo, type ReactNode } from 'react';
import { useForm, type UseFormRegisterReturn } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useBetBreakdown } from '@/hooks/useBetBreakdown';
import { cn } from '@/lib/cn';
import { COMBO_BET_TYPES } from '@/lib/betTypes';
import { formatCount, formatPercent, formatRatio } from '@/lib/formatters';
import type { SettingsResponse, SettingsUpdate } from '@/types/api';

// **画面はしきい値を % で扱う。** 0.075 と書かせるより 7.5% の方が読み書きしやすく、
// 何より 1 つの列に 0.60 と 7.5 が混ざると、100 倍違う数が同じ大きさに見える。
// API は 0〜1 なので toForm / submit で 100 倍・1/100 する。
//
// scraper_stopped は持たない。画面に入力欄が無いのにスキーマにあると、保存のたびに
// **読み込み時の値を送り返す** ので、その間に Race 画面から止めた停止フラグが戻る。
const schema = z
  .object({
    user_agent: z.string().min(1, 'User-Agent を入力してください'),
    rate_min_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    rate_max_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    night_min_seconds: z.coerce.number().min(0, '0 以上の値を入力してください'),
    win_min_odds: z.coerce.number().min(1.0, '1.0 以上の値を入力してください'),
    place_min_hit_prob: z.coerce
      .number()
      .min(0, '0 以上の値を入力してください')
      .max(100, '100 以下の値を入力してください'),
    // 賭け金の設定は「1 レースにいくらまで」だけ。1 点 = 100 円は固定で、
    // 何点買うかは確信度が決めるので、券種ごとの金額も券種の選択も設定に無い。
    race_budget: z.coerce
      .number()
      .int('整数で入力してください')
      .min(100, '100 以上の値を入力してください'),
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

interface SettingsFormProps {
  defaults: SettingsResponse;
  onSubmit: (values: SettingsUpdate) => void;
  isPending: boolean;
}

/**
 * 3 つの節で共有する列幅。**入力の位置を揃えるための唯一の出どころ**なので
 * 片方だけ変えない。
 *
 * 固定幅の合計は 752px で、器 (max-w-3xl = 768px) との差 16px は幅を持たない
 * spacer 列が飲む。spacer が無いと `table-fixed` が余りを固定幅の列へ比例配分し、
 * 実測の列がある節とない節で入力の位置がずれる。
 *
 * 実測は `w-16` (使える幅 40px) では「100.0%」が入らないので 5rem 取る。
 */
const COL = {
  subject: 'w-44',
  meaning: 'w-32',
  input: 'w-52',
  measure: 'w-20',
};

/** 単位は数の後ろ。幅を固定して常に出すので、入力の右端が全行で揃う。 */
const UNIT = 'w-8 shrink-0 text-left text-unit';

/** 設定 1 行分。予算・しきい値・取り込み方が同じ形を共有する。 */
interface SettingRow {
  /** 左端の名前。 */
  subject: string;
  /** 何の値か。ここが説明を兼ねるので、行に説明文は付けない。 */
  meaning: string;
  /** 入力の名前。表のヘッダだけでは行を特定できない。 */
  ariaLabel: string;
  /** 数値入力の刻みと範囲。文字列入力は持たない。 */
  step?: string;
  min?: string;
  max?: string;
  /** 数の後ろに出す単位。無い行も枠だけ取る。 */
  unit?: string;
  /** 設定の列を spacer まで広げる。値を見比べない文字列入力だけ。 */
  wide?: boolean;
}

interface ThresholdRow extends SettingRow {
  field: 'win_min_odds' | 'place_min_hit_prob' | `combo_min_hit_prob.${string}`;
}

/**
 * 並びは **推奨が予算を使う順** (単勝 → 複勝 → 連系)。
 * `docs/ai-model.md`「推奨ベットルール」がその順で買うと決めているので、
 * 設定の並びを合わせると画面がルールそのものになる。
 */
const THRESHOLD_ROWS: ThresholdRow[] = [
  {
    subject: '単勝',
    meaning: 'オッズ下限',
    field: 'win_min_odds',
    ariaLabel: '単勝のオッズ下限',
    step: '0.01',
    min: '1',
  },
  {
    subject: '複勝',
    meaning: '3 着内率',
    field: 'place_min_hit_prob',
    ariaLabel: '複勝を買う確信度の下限',
    step: '5',
    min: '0',
    max: '100',
    unit: '%',
  },
  ...COMBO_BET_TYPES.map((betType) => ({
    subject: betType,
    meaning: '的中率',
    field: `combo_min_hit_prob.${betType}` as const,
    ariaLabel: `${betType} を買う確信度の下限`,
    step: '0.5',
    min: '0',
    max: '100',
    unit: '%',
  })),
];

/** netkeiba との作法。**畳まない** — 隠すと「いま何秒で叩いているか」が読めなくなる。 */
const INGEST_ROWS: (SettingRow & {
  field: 'user_agent' | 'rate_min_seconds' | 'rate_max_seconds' | 'night_min_seconds';
})[] = [
  {
    subject: 'User-Agent',
    meaning: '名乗る文字列',
    field: 'user_agent',
    ariaLabel: 'User-Agent',
    // 他の行と縦に見比べる値ではないので、右端まで使う。
    // 幅を揃えると 20 文字ほどしか見えず、書き換えるのに端から端まで送ることになる。
    wide: true,
  },
  {
    subject: 'rate_min',
    meaning: '間隔の下限',
    field: 'rate_min_seconds',
    ariaLabel: 'リクエスト間隔の下限',
    step: '0.1',
    min: '0',
    unit: '秒',
  },
  {
    subject: 'rate_max',
    meaning: '間隔の上限',
    field: 'rate_max_seconds',
    ariaLabel: 'リクエスト間隔の上限',
    step: '0.1',
    min: '0',
    unit: '秒',
  },
  {
    subject: 'night_min',
    meaning: '夜間の最小',
    field: 'night_min_seconds',
    ariaLabel: '夜間の最小待機',
    step: '0.1',
    min: '0',
    unit: '秒',
  },
];

/**
 * 設定フォーム。**保存した値がそのまま返ってくるかは画面では分からない** —
 * バックエンドの _dict_to_response でキー名を間違えると pydantic が黙って捨て、
 * 既定値が返る (「保存したのに戻る」に見える)。項目を足すときは応答側も確かめる。
 */
export function SettingsForm({ defaults, onSubmit, isPending }: SettingsFormProps) {
  // しきい値の隣に置く実測。**数字はベタ書きしない** — 測り直すたびに動くので、
  // 写すと画面と docs のどちらかが必ず古くなる (docs/design.md「設定値と実測」)。
  const breakdown = useBetBreakdown({ group_by: 'bet_type' });
  const measured = useMemo(
    () => new Map((breakdown.data?.rows ?? []).map((r) => [r.group_key, r])),
    [breakdown.data]
  );
  // 記録が 1 件も無ければ列ごと出さない。初回起動は必ずこの状態で、
  // 「未計測」が 7 行並んでも実測が無いことを 7 回言うだけになる。
  const hasRecord = measured.size > 0;

  const toForm = (d: SettingsResponse): FormValues => ({
    user_agent: d.user_agent,
    rate_min_seconds: d.rate_min_seconds,
    rate_max_seconds: d.rate_max_seconds,
    night_min_seconds: d.night_min_seconds,
    win_min_odds: d.win_min_odds,
    race_budget: d.race_budget,
    place_min_hit_prob: toPercent(d.place_min_hit_prob),
    combo_min_hit_prob: Object.fromEntries(
      Object.entries(d.combo_min_hit_prob ?? {}).map(([k, v]) => [k, toPercent(v)])
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
      place_min_hit_prob: values.place_min_hit_prob / 100,
      combo_min_hit_prob: Object.fromEntries(
        Object.entries(values.combo_min_hit_prob ?? {}).map(([k, v]) => [k, Number(v) / 100])
      ),
    });
  }

  /** 行のフィールド名からエラーを引く。連系は record なのでキーごとに持つ。 */
  function errorFor(field: string): string | undefined {
    if (field.startsWith('combo_min_hit_prob.')) {
      const key = field.slice('combo_min_hit_prob.'.length);
      return errors.combo_min_hit_prob?.[key]?.message;
    }
    return (errors as Record<string, { message?: string } | undefined>)[field]?.message;
  }

  const dirtyCount = countDirtyFields(dirtyFields);

  return (
    <form onSubmit={handleSubmit(submit)} className="flex flex-col gap-8" noValidate>
      {/* 予算は券種ではないので表に混ぜない。面にも載せない — 枠であって答えでは
          ないので、面に載せると 1 行しかない予算が 7 行あるしきい値より重く見える。
          見出しを持った節にすれば、どちらも起きない。 */}
      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-medium">予算</h2>
        <Table aria-label="予算" className="w-full table-fixed">
          <TableBody>
            <SettingTableRow
              row={{
                subject: '1 レースに使う上限',
                meaning: '使ってよい額',
                ariaLabel: '1 レースに使う上限',
                step: '500',
                min: '100',
                unit: '円',
              }}
              registration={register('race_budget')}
              error={errorFor('race_budget')}
            />
          </TableBody>
        </Table>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-medium">券種ごとのしきい値</h2>
        <Table aria-label="券種ごとのしきい値" className="w-full table-fixed">
          <TableHeader>
            <TableRow>
              <TableHead className={COL.subject}>券種</TableHead>
              <TableHead className={COL.meaning}>しきい値</TableHead>
              {/* 入力はセルの左端から始まり、右に単位の枠が空く。ヘッダを
                  右揃えにすると入力の右端とも合わず、宙に浮いて見える。 */}
              <TableHead className={COL.input}>設定</TableHead>
              <TableHead />
              {hasRecord && (
                <>
                  <TableHead className={cn(COL.measure, 'text-right')}>記録</TableHead>
                  <TableHead className={cn(COL.measure, 'text-right')}>回収率</TableHead>
                  <TableHead className={cn(COL.measure, 'text-right')}>的中率</TableHead>
                </>
              )}
            </TableRow>
          </TableHeader>
          <TableBody>
            {THRESHOLD_ROWS.map((row) => {
              const m = measured.get(row.subject);
              return (
                <SettingTableRow
                  key={row.subject}
                  row={row}
                  registration={register(row.field)}
                  error={errorFor(row.field)}
                >
                  {hasRecord && (
                    <>
                      <TableCell className="cell-num">{m ? formatCount(m.bets) : '·'}</TableCell>
                      <TableCell
                        className={cn('cell-num', m && m.payback_rate >= 1 && 'text-success')}
                      >
                        {m ? formatRatio(m.payback_rate) : '·'}
                      </TableCell>
                      <TableCell className="cell-num">
                        {m ? formatPercent(m.hit_rate) : '·'}
                      </TableCell>
                    </>
                  )}
                </SettingTableRow>
              );
            })}
          </TableBody>
        </Table>
        {hasRecord && (
          <p className="text-2xs text-subtle-foreground">
            右 3 列は台帳に確定済みの買い目から。設定を変えた前後が混ざっている
          </p>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-medium">取り込み方</h2>
        <Table aria-label="取り込み方" className="w-full table-fixed">
          <TableBody>
            {INGEST_ROWS.map((row) => (
              <SettingTableRow
                key={row.field}
                row={row}
                registration={register(row.field)}
                error={errorFor(row.field)}
              />
            ))}
          </TableBody>
        </Table>
      </section>

      {/* Sticky footer — 変更があるときだけ出す。
          常時出ていると「未保存かどうか」という情報そのものが失われる。 */}
      {(isDirty || isPending) && (
        <div className="sticky bottom-0 z-10 flex items-center justify-end gap-3 border-t border-border bg-background py-3">
          <span className="text-sm text-muted-foreground">{dirtyCount} 件の変更があります</span>
          <Button type="submit" disabled={!isDirty || isPending}>
            {isPending ? '保存中…' : '変更を保存'}
          </Button>
        </div>
      )}
    </form>
  );
}

interface SettingTableRowProps {
  row: SettingRow;
  registration: UseFormRegisterReturn;
  error?: string;
  /** 右に続くセル (実測)。 */
  children?: ReactNode;
}

/**
 * 3 つの節で共有する 1 行。**行の区切り線は引かない** — 設定は 12 行あり、
 * 1 行ずつ罫線で挟むと引きで見たときに線が縞になって内容より先に目に付く
 * (docs/design.md「領域の作り方」と同じ理由)。
 *
 * 数値入力は幅を固定し、文字列入力は残りを埋める。**単位の枠は値が無くても
 * 取る**ので、どちらも右端が同じ位置で終わる。
 */
function SettingTableRow({ row, registration, error, children }: SettingTableRowProps) {
  const isNumber = row.step !== undefined;
  return (
    <TableRow className="hover:bg-transparent">
      <TableCell className={cn(COL.subject, 'whitespace-nowrap font-medium')}>
        {row.subject}
      </TableCell>
      <TableCell className={cn(COL.meaning, 'whitespace-nowrap text-xs text-subtle-foreground')}>
        {row.meaning}
      </TableCell>
      <TableCell className={row.wide ? undefined : COL.input} colSpan={row.wide ? 2 : undefined}>
        <div className="flex items-baseline gap-1">
          <Input
            type={isNumber ? 'number' : 'text'}
            step={row.step}
            min={row.min}
            max={row.max}
            aria-label={row.ariaLabel}
            className={cn(
              'font-mono tabular-nums',
              isNumber ? 'w-32 text-right' : 'min-w-0 flex-1'
            )}
            {...registration}
          />
          <span className={UNIT}>{row.unit ?? ''}</span>
        </div>
        {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
      </TableCell>
      {!row.wide && <TableCell />}
      {children}
    </TableRow>
  );
}

/** 0〜1 の確率を画面の % へ。0.075 → 7.5 (浮動小数の桁あふれを落とす)。 */
function toPercent(value: number | null | undefined): number {
  return +((value ?? 0) * 100).toFixed(2);
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
