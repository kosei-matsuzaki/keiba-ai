import { useEffect, useMemo, useState, type ChangeEvent } from 'react';
import { Plus } from 'lucide-react';

import { useRacesByDate } from '@/hooks/useRacesByDate';
import { useRaceDetail } from '@/hooks/useRaceDetail';
import { useCreateBetsBulk } from '@/hooks/useCreateBetsBulk';
import { betK, isOrdered, expandCombos, type BetMethod } from '@/lib/betCombos';
import { cn } from '@/lib/cn';
import { formatYen } from '@/lib/formatters';
import { DateYMDPicker } from '@/components/DateYMDPicker';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { BetType } from '@/types/api';

// 枠連は枠番ベースで entries に枠番を持たないため、馬番選択 UI からは除外する。
const BET_TYPES: BetType[] = ['単勝', '複勝', '馬連', 'ワイド', '馬単', '三連複', '三連単'];

const METHOD_LABEL: Record<BetMethod, string> = {
  single: '通常',
  box: 'ボックス',
  nagashi: 'ながし',
  formation: 'フォーメーション',
};

interface Horse {
  num: number;
  name: string;
}

/** 今日の日付 YYYY-MM-DD (ローカル)。 */
function todayIso(): string {
  const d = new Date();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${d.getFullYear()}-${m}-${day}`;
}

// ── 馬選択プリミティブ ────────────────────────────────────────────────────────

function HorseSelect({
  value,
  onChange,
  horses,
  exclude = [],
  placeholder,
}: {
  value: number | null;
  onChange: (n: number | null) => void;
  horses: Horse[];
  exclude?: number[];
  placeholder?: string;
}) {
  return (
    <Select
      value={value != null ? String(value) : ''}
      onValueChange={(v) => onChange(v === '' ? null : Number(v))}
    >
      <SelectTrigger>
        <SelectValue placeholder={placeholder ?? '選択'} />
      </SelectTrigger>
      <SelectContent>
        {horses
          .filter((h) => !exclude.includes(h.num))
          .map((h) => (
            <SelectItem key={h.num} value={String(h.num)}>
              {h.num} {h.name}
            </SelectItem>
          ))}
      </SelectContent>
    </Select>
  );
}

function HorseChips({
  horses,
  selected,
  onToggle,
  exclude = [],
}: {
  horses: Horse[];
  selected: number[];
  onToggle: (n: number) => void;
  exclude?: number[];
}) {
  const sel = new Set(selected);
  return (
    <div className="flex flex-wrap gap-1.5">
      {horses
        .filter((h) => !exclude.includes(h.num))
        .map((h) => {
          const on = sel.has(h.num);
          return (
            <button
              key={h.num}
              type="button"
              onClick={() => onToggle(h.num)}
              aria-pressed={on}
              className={cn(
                'rounded-md border px-2 py-1 text-xs transition-colors',
                on
                  ? 'border-primary bg-primary text-primary-foreground'
                  : 'border-border bg-background hover:bg-muted',
              )}
            >
              <span className="font-medium">{h.num}</span> {h.name}
            </button>
          );
        })}
    </div>
  );
}

// ── メインダイアログ ──────────────────────────────────────────────────────────

/**
 * 実購入した馬券を「買い方（通常 / ボックス / ながし / フォーメーション）」で記録する。
 * レースをシステムから選び、出走馬を馬番+馬名で選択 → 組合せを展開して一括登録する。
 * 結果が取込済みなら登録と同時にサーバ側で自動確定する（source='manual'）。
 */
export function AddBetDialog() {
  const [open, setOpen] = useState(false);
  const [date, setDate] = useState(todayIso());
  const [raceId, setRaceId] = useState('');
  const [betType, setBetType] = useState<BetType>('馬連');
  const [method, setMethod] = useState<BetMethod>('box');
  const [perStake, setPerStake] = useState(100);
  const [notes, setNotes] = useState('');

  // 買い方ごとの選択状態
  const [single, setSingle] = useState<(number | null)[]>([]);
  const [boxSet, setBoxSet] = useState<number[]>([]);
  const [axisCount, setAxisCount] = useState(1);
  const [axes, setAxes] = useState<(number | null)[]>([]);
  const [axisPositions, setAxisPositions] = useState<number[]>([1]);
  const [opp, setOpp] = useState<number[]>([]);
  const [cols, setCols] = useState<number[][]>([]);

  const racesQuery = useRacesByDate(date);
  const races = racesQuery.data?.races ?? [];
  const raceDetail = useRaceDetail(raceId);
  const bulk = useCreateBetsBulk();

  const k = betK(betType);
  const ordered = isOrdered(betType);
  const allowedMethods: BetMethod[] =
    k === 1 ? ['single', 'box'] : ['single', 'box', 'nagashi', 'formation'];

  const horses: Horse[] = useMemo(
    () =>
      (raceDetail.data?.entries ?? [])
        .filter((e) => e.post_position != null)
        .map((e) => ({ num: e.post_position as number, name: e.horse_name ?? '' }))
        .sort((a, b) => a.num - b.num),
    [raceDetail.data],
  );

  function clearSelections() {
    setSingle([]);
    setBoxSet([]);
    setAxisCount(1);
    setAxes([]);
    setAxisPositions([1]);
    setOpp([]);
    setCols([]);
  }

  // 券種 / レースが変わったら買い方を初期化（k が変わるため）。
  useEffect(() => {
    setMethod('box');
    clearSelections();
  }, [betType, raceId]);

  // 買い方が変わったら選択をクリア。
  useEffect(() => {
    clearSelections();
  }, [method]);

  // 軸の頭数が変わったら軸スロット/着順を同期（既定は 1着・2着…）。
  useEffect(() => {
    setAxisPositions((prev) => Array.from({ length: axisCount }, (_, i) => prev[i] ?? i + 1));
    setAxes((prev) => prev.slice(0, axisCount));
  }, [axisCount]);

  const combos = useMemo(
    () =>
      expandCombos({
        betType,
        method,
        single: single.filter((x): x is number => x != null),
        box: boxSet,
        axes: axes.filter((x): x is number => x != null),
        axisPositions,
        opponents: opp,
        columns: cols.slice(0, k),
      }),
    [betType, method, single, boxSet, axes, axisPositions, opp, cols, k],
  );

  const totalStake = combos.length * perStake;
  const canSubmit =
    raceId !== '' &&
    combos.length > 0 &&
    combos.length <= 1000 &&
    perStake >= 100 &&
    !bulk.isPending;

  function handleStakeChange(e: ChangeEvent<HTMLInputElement>) {
    const n = Number(e.target.value);
    setPerStake(Number.isNaN(n) ? 0 : Math.max(0, Math.floor(n / 100) * 100));
  }

  function toggleIn(setter: (fn: (prev: number[]) => number[]) => void, n: number) {
    setter((prev) => (prev.includes(n) ? prev.filter((x) => x !== n) : [...prev, n]));
  }

  function setSlot(i: number, v: number | null) {
    setSingle((prev) => {
      const a = [...prev];
      a[i] = v;
      return a;
    });
  }

  function toggleCol(i: number, n: number) {
    setCols((prev) => {
      const a = Array.from({ length: k }, (_, j) => [...(prev[j] ?? [])]);
      a[i] = a[i].includes(n) ? a[i].filter((x) => x !== n) : [...a[i], n];
      return a;
    });
  }

  function resetForm() {
    setRaceId('');
    setBetType('馬連');
    setMethod('box');
    setPerStake(100);
    setNotes('');
    clearSelections();
  }

  function handleSubmit() {
    if (!canSubmit) return;
    bulk.mutate(
      {
        race_id: raceId,
        bet_type: betType,
        source: 'manual',
        notes: notes.trim() || undefined,
        combos: combos.map((c) => ({ combo: c, stake: perStake })),
      },
      {
        onSuccess: () => {
          setOpen(false);
          resetForm();
        },
      },
    );
  }

  const noHorses = raceId !== '' && !raceDetail.isPending && horses.length === 0;

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        setOpen(o);
        if (!o) resetForm();
      }}
    >
      <DialogTrigger asChild>
        <Button size="sm">
          <Plus className="mr-1 h-4 w-4" />
          購入を記録
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>購入を記録</DialogTitle>
          <DialogDescription>
            買い方を選んで、出走馬を馬番・馬名から選択します。結果が取込済みなら自動で確定します。
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3">
          {/* 日付 + レース */}
          <div className="grid gap-1.5">
            <Label>日付</Label>
            <DateYMDPicker value={date} onChange={setDate} ariaLabel="開催日" />
          </div>
          <div className="grid gap-1.5">
            <Label>レース</Label>
            <Select value={raceId} onValueChange={setRaceId} disabled={races.length === 0}>
              <SelectTrigger>
                <SelectValue
                  placeholder={
                    racesQuery.isPending
                      ? '読込中...'
                      : races.length === 0
                        ? '該当日にレースがありません'
                        : 'レースを選択'
                  }
                />
              </SelectTrigger>
              <SelectContent>
                {races.map((r) => (
                  <SelectItem key={r.race_id} value={r.race_id}>
                    {r.race_id.slice(-2)}R {r.course} {r.name ?? ''}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* 券種 + 買い方 */}
          <div className="grid grid-cols-2 gap-3">
            <div className="grid gap-1.5">
              <Label>券種</Label>
              <Select value={betType} onValueChange={(v) => setBetType(v as BetType)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BET_TYPES.map((bt) => (
                    <SelectItem key={bt} value={bt}>
                      {bt}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-1.5">
              <Label>買い方</Label>
              <div className="flex flex-wrap gap-1">
                {allowedMethods.map((m) => (
                  <Button
                    key={m}
                    type="button"
                    size="sm"
                    variant={method === m ? 'default' : 'outline'}
                    className="h-9 px-2 text-xs"
                    onClick={() => setMethod(m)}
                  >
                    {METHOD_LABEL[m]}
                  </Button>
                ))}
              </div>
            </div>
          </div>

          {/* 馬選択パネル */}
          {raceId === '' ? (
            <p className="text-sm text-muted-foreground">レースを選択してください。</p>
          ) : raceDetail.isPending ? (
            <p className="text-sm text-muted-foreground">出走馬を読込中...</p>
          ) : noHorses ? (
            <p className="text-sm text-muted-foreground">
              出走馬が未取得です。Race 画面で出馬表を取り込んでください。
            </p>
          ) : (
            <div className="rounded-md border border-border/60 p-3">
              {method === 'single' && (
                <div className="grid gap-2">
                  {Array.from({ length: k }).map((_, i) => (
                    <div key={i} className="grid gap-1.5">
                      <Label className="text-xs">{ordered ? `${i + 1}着` : `${i + 1}頭目`}</Label>
                      <HorseSelect
                        value={single[i] ?? null}
                        onChange={(v) => setSlot(i, v)}
                        horses={horses}
                        exclude={single
                          .filter((_, j) => j !== i)
                          .filter((x): x is number => x != null)}
                      />
                    </div>
                  ))}
                </div>
              )}

              {method === 'box' && (
                <div className="grid gap-1.5">
                  <Label className="text-xs">馬を選択（{boxSet.length} 頭）</Label>
                  <HorseChips
                    horses={horses}
                    selected={boxSet}
                    onToggle={(n) => toggleIn(setBoxSet, n)}
                  />
                </div>
              )}

              {method === 'nagashi' && (
                <div className="grid gap-3">
                  {/* 三連系は 1頭軸 / 2頭軸 を選べる */}
                  {k === 3 && (
                    <div className="grid gap-1.5">
                      <Label className="text-xs">軸の頭数</Label>
                      <div className="flex gap-1">
                        {[1, 2].map((n) => (
                          <Button
                            key={n}
                            type="button"
                            size="sm"
                            variant={axisCount === n ? 'default' : 'outline'}
                            className="h-8 px-3 text-xs"
                            onClick={() => setAxisCount(n)}
                          >
                            {n}頭軸
                          </Button>
                        ))}
                      </div>
                    </div>
                  )}

                  {Array.from({ length: axisCount }).map((_, i) => (
                    <div key={i} className="grid gap-1.5">
                      <Label className="text-xs">{axisCount > 1 ? `軸${i + 1}` : '軸'}</Label>
                      <div className="flex gap-2">
                        <HorseSelect
                          value={axes[i] ?? null}
                          onChange={(v) =>
                            setAxes((prev) => {
                              const a = [...prev];
                              a[i] = v;
                              return a;
                            })
                          }
                          horses={horses}
                          exclude={axes
                            .filter((_, j) => j !== i)
                            .filter((x): x is number => x != null)}
                        />
                        {ordered && (
                          <Select
                            value={String(axisPositions[i] ?? i + 1)}
                            onValueChange={(v) =>
                              setAxisPositions((prev) => {
                                const a = [...prev];
                                a[i] = Number(v);
                                return a;
                              })
                            }
                          >
                            <SelectTrigger className="w-24 shrink-0">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {Array.from({ length: k }).map((_, p) => (
                                <SelectItem key={p} value={String(p + 1)}>
                                  {p + 1}着
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        )}
                      </div>
                    </div>
                  ))}

                  <div className="grid gap-1.5">
                    <Label className="text-xs">相手（{opp.length} 頭）</Label>
                    <HorseChips
                      horses={horses}
                      selected={opp}
                      onToggle={(n) => toggleIn(setOpp, n)}
                      exclude={axes.filter((x): x is number => x != null)}
                    />
                  </div>
                  {ordered && axisCount > 1 && new Set(axisPositions.slice(0, axisCount)).size < axisCount && (
                    <p className="text-xs text-destructive">軸の着順が重複しています。</p>
                  )}
                </div>
              )}

              {method === 'formation' && (
                <div className="grid gap-3">
                  {Array.from({ length: k }).map((_, i) => (
                    <div key={i} className="grid gap-1.5">
                      <Label className="text-xs">
                        {ordered ? `${i + 1}着` : `${i + 1}列目`}（{(cols[i] ?? []).length} 頭）
                      </Label>
                      <HorseChips
                        horses={horses}
                        selected={cols[i] ?? []}
                        onToggle={(n) => toggleCol(i, n)}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* 金額 + メモ */}
          <div className="grid grid-cols-2 gap-3">
            <div className="grid gap-1.5">
              <Label>1点あたり (円)</Label>
              <Input
                type="number"
                min={100}
                step={100}
                value={perStake}
                onChange={handleStakeChange}
              />
            </div>
            <div className="grid gap-1.5">
              <Label>メモ (任意)</Label>
              <Input value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="任意" />
            </div>
          </div>

          {/* 点数サマリ */}
          <div className="rounded-md bg-muted/50 px-3 py-2 text-sm">
            {combos.length > 0 ? (
              <>
                <span className="font-medium">{combos.length} 点</span>
                <span className="text-muted-foreground"> × {formatYen(perStake)} = </span>
                <span className="font-semibold">{formatYen(totalStake)}</span>
                {combos.length > 1000 && (
                  <span className="ml-2 text-destructive">点数が多すぎます（最大 1000）</span>
                )}
              </>
            ) : (
              <span className="text-muted-foreground">買い目を選択してください。</span>
            )}
          </div>
        </div>

        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline">キャンセル</Button>
          </DialogClose>
          <Button onClick={handleSubmit} disabled={!canSubmit}>
            {bulk.isPending ? '記録中...' : `記録する${combos.length > 0 ? `（${combos.length}点）` : ''}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
