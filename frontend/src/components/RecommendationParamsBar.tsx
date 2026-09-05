import { useSettings } from '@/hooks/useSettings';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ALL_BET_TYPES } from '@/lib/betTypes';
import { cn } from '@/lib/cn';
import type { BetType } from '@/types/api';

/**
 * このレースだけの上書き。未設定の項目は Settings の既定値が使われる。
 *
 * 操作できるのは「このレースに使う上限」と「買う馬券」の 2 つだけ。
 *
 * **「狙い方」(上位何頭で買い目を組むか) は廃止した。** 買うかどうかは確信度の
 * 下限が決めるので、頭数を広げても線を超えない買い目が候補に増えるだけで、
 * 買い目そのものは変わらない。選べるのに何も起きない選択肢になっていた。
 */
export interface RecommendationOverrides {
  /** このレースに使う上限 (円)。 */
  race_budget?: number;
  bet_types?: string[];
}

interface RecommendationParamsBarProps {
  value: RecommendationOverrides;
  onChange: (next: RecommendationOverrides) => void;
}

export function RecommendationParamsBar({ value, onChange }: RecommendationParamsBarProps) {
  const { data: settings } = useSettings();

  // 券種は設定で絞らない。既定は全券種で、ここでは「このレースだけ」外せる。
  const defaultBetTypes = ALL_BET_TYPES as readonly BetType[];
  const defaultBudget = settings?.race_budget ?? null;

  const selected = (value.bet_types ?? defaultBetTypes) as BetType[];
  const overridden = value.bet_types != null || value.race_budget != null;

  function toggle(betType: BetType) {
    const next = selected.includes(betType)
      ? selected.filter((t) => t !== betType)
      : [...selected, betType];
    // 全部外すと候補が空になるので、最後の 1 つは外させない
    if (next.length === 0) return;
    onChange({ ...value, bet_types: next });
  }

  return (
    <div className="flex flex-col gap-3 pb-4">
      <div className="grid gap-3 sm:grid-cols-[7rem_minmax(0,1fr)] sm:gap-x-6">
        {overridden && (
          <>
            <span />
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-fit px-2 text-xs"
              onClick={() => onChange({})}
            >
              いつもの設定に戻す
            </Button>
          </>
        )}
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
          <span className="text-xs text-subtle-foreground">まで</span>
        </div>

        {/* 買う馬券 */}
        <span className="text-label-ja self-center">買う馬券</span>
        <div className="flex flex-wrap items-center gap-2">
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
      </div>
    </div>
  );
}
