import { useSettings } from '@/hooks/useSettings';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ALL_BET_TYPES } from '@/lib/betTypes';
import { cn } from '@/lib/cn';
import type { BetType } from '@/types/api';

/**
 * このレースだけの上書き。未設定の項目は Settings の既定値が使われる。
 *
 * 賭け金は「このレースに使う上限」と「1 点あたりの額」の 2 つだけで決まる
 * （資金比率の Kelly は廃止した）。利用者が考えるのは
 * 「このレースに何円使うか」「どの馬券を買うか」「穴を狙うか」なので、
 * その 3 つをそのまま操作対象にする。
 */
export interface RecommendationOverrides {
  /** このレースに使う上限 (円)。 */
  race_budget?: number;
  bet_types?: string[];
  /** 買い目に含める頭数 = 狙い方。 */
  top_n_horses?: number;
}

interface RecommendationParamsBarProps {
  value: RecommendationOverrides;
  onChange: (next: RecommendationOverrides) => void;
}

/**
 * 狙い方 = 買い目に何頭目まで含めるか。
 *
 * 上位だけで組めば本命寄り、頭数を増やすほど人気薄が買い目に入る。
 * 「穴狙い」という言い方は利用者の考え方に合わせたもので、内部では
 * top_n_horses（組み合わせを作る対象の頭数）を動かしているだけ。
 */
const AIM_PRESETS = [
  { value: 3, label: '本命中心', hint: '予想上位 3 頭だけで買い目を組む' },
  { value: 5, label: '標準', hint: '予想上位 5 頭で買い目を組む' },
  { value: 8, label: '穴も拾う', hint: '予想上位 8 頭まで広げ、人気薄も買い目に入れる' },
] as const;

const DEFAULT_AIM = 3;

export function RecommendationParamsBar({ value, onChange }: RecommendationParamsBarProps) {
  const { data: settings } = useSettings();

  const defaultBetTypes = (settings?.enabled_bet_types ?? []) as BetType[];
  const defaultBudget = settings?.race_budget ?? null;

  const selected = (value.bet_types ?? defaultBetTypes) as BetType[];
  const aim = value.top_n_horses ?? DEFAULT_AIM;
  const overridden =
    value.bet_types != null || value.race_budget != null || value.top_n_horses != null;

  function toggle(betType: BetType) {
    const next = selected.includes(betType)
      ? selected.filter((t) => t !== betType)
      : [...selected, betType];
    // 全部外すと候補が空になるので、最後の 1 つは外させない
    if (next.length === 0) return;
    onChange({ ...value, bet_types: next });
  }


  return (
    <div className="flex flex-col gap-3 border-b border-border pb-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-label-ja">この予想の条件</span>
        {overridden ? (
          <>
            <span className="font-mono text-[10px] tracking-[0.04em] text-primary">
              このレースだけ変更中
            </span>
            <Button variant="ghost" size="sm" className="h-6 px-2" onClick={() => onChange({})}>
              いつもの設定に戻す
            </Button>
          </>
        ) : (
          <span className="font-mono text-[10px] tracking-[0.04em] text-subtle-foreground">
            いつもの設定のまま
          </span>
        )}
      </div>

      <div className="grid gap-3 sm:grid-cols-[7rem_minmax(0,1fr)] sm:gap-x-6">
        {/* 使う金額 */}
        <span className="text-label-ja self-center">このレースに使う</span>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1">
            <span className="font-mono text-sm text-subtle-foreground">¥</span>
            <Input
              type="number"
              min={100}
              step={500}
              className="h-8 w-32 text-right font-mono tabular-nums"
              placeholder={defaultBudget != null ? String(defaultBudget) : ''}
              value={value.race_budget ?? ''}
              onChange={(e) => {
                const raw = e.target.value;
                const next = { ...value };
                if (raw === '') {
                  delete next.race_budget;
                } else {
                  const n = Number(raw);
                  if (Number.isFinite(n)) next.race_budget = n;
                }
                onChange(next);
              }}
              aria-label="このレースに使う金額"
            />
          </div>
          <span className="text-xs text-subtle-foreground">
            まで
            {value.race_budget == null && defaultBudget != null && '（いつもの設定）'}
          </span>
        </div>

        {/* 買う馬券 */}
        <span className="text-label-ja self-center">買う馬券</span>
        <div className="flex flex-wrap items-center gap-1.5">
          {ALL_BET_TYPES.map((betType) => {
            const on = selected.includes(betType);
            return (
              <button
                key={betType}
                type="button"
                onClick={() => toggle(betType)}
                aria-pressed={on}
                className={cn(
                  'rounded-sm border px-2 py-1 text-xs transition-colors',
                  on
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border bg-transparent text-subtle-foreground hover:border-border-strong hover:text-foreground'
                )}
              >
                {betType}
              </button>
            );
          })}
        </div>

        {/* 狙い方 */}
        <span className="text-label-ja self-center">狙い方</span>
        <div className="flex flex-wrap items-center gap-1.5">
          {AIM_PRESETS.map((preset) => {
            const on = aim === preset.value;
            return (
              <button
                key={preset.value}
                type="button"
                onClick={() => onChange({ ...value, top_n_horses: preset.value })}
                aria-pressed={on}
                title={preset.hint}
                className={cn(
                  'rounded-sm border px-2.5 py-1 text-xs transition-colors',
                  on
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border bg-transparent text-subtle-foreground hover:border-border-strong hover:text-foreground'
                )}
              >
                {preset.label}
              </button>
            );
          })}
          <span className="ml-1 text-xs text-subtle-foreground">
            {AIM_PRESETS.find((p) => p.value === aim)?.hint}
          </span>
        </div>
      </div>

      <p className="text-xs leading-relaxed text-subtle-foreground">
        <strong>1 点 = 100 円</strong>。単勝・複勝は AI の本命に、連系は上位 {aim} 頭で組んだ
        買い目に、<strong>的中確率の高い順</strong>で予算まで賭けます。券種ごとの点数は設定で
        決まり、複勝だけ確信度に応じて増えます。
        <span className="ml-1">条件の詳細は表の下の「買い方」。</span>
      </p>
    </div>
  );
}
