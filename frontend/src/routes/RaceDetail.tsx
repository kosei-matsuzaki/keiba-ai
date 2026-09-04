import { useEffect, useState } from 'react';
import { useParams, useSearchParams, Link } from 'react-router-dom';
import {
  ChevronLeft,
  ChevronRight,
  Sparkles,
  RefreshCw,
  AlertTriangle,
} from 'lucide-react';

import { useRaceDetail } from '@/hooks/useRaceDetail';
import { usePredictions } from '@/hooks/usePredictions';
import { useRecommendations } from '@/hooks/useRecommendations';
import { useRunShutuba } from '@/hooks/useRunShutuba';
import { EntryPredictionTable } from '@/components/EntryPredictionTable';
import { RecommendationsCard } from '@/components/RecommendationsCard';
import type { RecommendationOverrides } from '@/components/RecommendationParamsBar';
import { EmptyState } from '@/components/EmptyState';
import { PageHeader } from '@/components/PageHeader';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { isNotFoundError, isServiceUnavailableError, formatErrorMessage } from '@/lib/api';
import { formatYen } from '@/lib/formatters';
import { toast } from '@/lib/toast';
import type { RaceInfoCoverage } from '@/types/api';

/**
 * レース番号。章番号 (§NN) の代わりにこの題材固有の番号を見出しに置く。
 * 琥珀 = オッズ・金額・レース番号に使うアクセント。
 */
function RaceNumber({ raceId }: { raceId: string }) {
  const n = raceId.slice(-2).replace(/^0/, '');
  if (!n) return null;
  return (
    <span className="font-mono text-2xl font-bold tabular-nums leading-none text-primary">
      {n}
      <span className="text-base">R</span>
    </span>
  );
}

function RaceDetailSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-40 w-full rounded-sm" />
      <Skeleton className="h-64 w-full rounded-sm" />
      <Skeleton className="h-64 w-full rounded-sm" />
    </div>
  );
}

/**
 * 出走馬に過去走がほとんど無いレース (新馬戦など) の注意書き。
 *
 * モデルは per-race 履歴 GRU と直近着順・上がり・脚質を主要な入力にしているので、
 * 全員が初出走だとそれが全滅し、枠順・馬体重・騎手・血統・オッズだけの予想になる。
 * 同じ画面・同じスコアでも入力の質が別物なので、黙って出さずに明示する。
 *
 * 実測 (test 19ヶ月・432 レース) では的中率はむしろ高い (単勝 29.2% / 複勝 63.7%) が、
 * 人気馬に寄るぶんオッズが低く、回収率は単勝 0.866 と全体 (0.933) を下回る。
 * 「当たりやすいが儲かりにくい」ので、的中率だけ見て信用しすぎないための注記でもある。
 */
function LowInformationNotice({ coverage }: { coverage: RaceInfoCoverage }) {
  return (
    <Card boxed className="border-warning/50 bg-warning/5">
      <CardContent className="flex items-start gap-3 p-4">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" aria-hidden="true" />
        <div className="flex flex-col gap-1">
          <p className="text-sm font-medium text-foreground">
            このレースは判断材料が少なめです
          </p>
          <p
            className="text-xs text-muted-foreground"
            title="AI が重く使う「前走までの着順・上がり・脚質」がほとんど無く、枠順・馬体重・騎手・血統・オッズだけで予想しています。"
          >
            出走 {coverage.n_runners} 頭のうち{' '}
            <strong>{coverage.n_debut} 頭に過去走がありません</strong>
            （平均 {coverage.mean_starts} 走）。参考程度に。
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

/** 「12:04」形式。オッズの鮮度と最終実行時刻に使う。 */
function formatClock(d: Date): string {
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

/**
 * 実行中の進捗。段階名を出す 1 本のバーにまとめる。
 * ジョブはインメモリ管理なので、バックエンド再起動で追跡できなくなることも伝える。
 */
function RunProgress({ stage }: { stage: 'entries' | 'predict' }) {
  const label = stage === 'entries' ? '出馬表を取得中…' : '予想を計算中…';
  return (
    <div className="py-3">
      <div className="flex items-center gap-3">
        <div className="h-1 flex-1 overflow-hidden rounded-full bg-card-elevated">
          <div
            className="h-full w-1/3 animate-skeleton-shimmer bg-primary"
            style={{ marginLeft: stage === 'entries' ? '0' : '33%' }}
          />
        </div>
        <span className="font-mono text-[11px] tabular-nums text-primary">{label}</span>
      </div>
    </div>
  );
}

/** 表の形をしたスケルトン。塊で出すと表示された瞬間にレイアウトが跳ねる。 */
function TableSkeleton({ rows }: { rows: number }) {
  return (
    <div className="w-full">
      <div className="border-b-2 border-border-strong py-2">
        <Skeleton className="h-3 w-40" />
      </div>
      {Array.from({ length: Math.min(rows, 18) }).map((_, i) => (
        <div key={i} className="flex items-center gap-3 border-b border-border py-2">
          <Skeleton className="h-6 w-6 shrink-0" />
          <Skeleton className="h-3 flex-1" />
          <Skeleton className="h-3 w-16 shrink-0" />
          <Skeleton className="h-3 w-16 shrink-0" />
        </div>
      ))}
    </div>
  );
}

/**
 * 前後のレースへ移動 (B-4 ⑦)。race_id の末尾 2 桁がレース番号なので、
 * 1〜12 の範囲で前後に振る。1 開催を順に見る動線がこれで通る。
 */
function RaceStepper({ raceId, date }: { raceId: string; date: string | null }) {
  const prefix = raceId.slice(0, -2);
  const n = Number(raceId.slice(-2));
  if (!Number.isFinite(n) || n < 1) return null;
  const q = date ? `?date=${date}` : '';
  const to = (target: number) => `/races/${prefix}${String(target).padStart(2, '0')}${q}`;
  return (
    <div className="flex items-center gap-1">
      <Button asChild variant="ghost" size="sm" disabled={n <= 1}>
        <Link to={to(n - 1)} aria-label="前のレース" aria-disabled={n <= 1}>
          <ChevronLeft className="h-4 w-4" />
          {n - 1}R
        </Link>
      </Button>
      <Button asChild variant="ghost" size="sm" disabled={n >= 12}>
        <Link to={to(n + 1)} aria-label="次のレース" aria-disabled={n >= 12}>
          {n + 1}R
          <ChevronRight className="h-4 w-4" />
        </Link>
      </Button>
    </div>
  );
}

interface MetaItemProps {
  label: string;
  value: string;
  mono?: boolean;
}

function MetaItem({ label, value, mono }: MetaItemProps) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className={mono ? 'font-mono text-xs' : ''}>{value}</dd>
    </div>
  );
}

/**
 * 1 レースの画面。出走馬・AI の根拠・推奨買目・答え合わせをこの 1 枚に集める。
 *
 * 買うかどうかを決めているのは推奨買目のカードで、表は根拠を見るためのもの。
 * 表の行に BUY バッジは出さない (買うのは常にモデル 1 位の 1 頭なので、
 * 列を 1 つ使って「1 位かどうか」を二重に示すだけになる)。
 */
export function RaceDetail() {
  const { race_id = '' } = useParams<{ race_id: string }>();
  const [searchParams] = useSearchParams();
  const dateParam = searchParams.get('date');

  // AI 予想 (予想スコア + 推奨買い目) は画面を開いた瞬間ではなく、ボタンを
  // 押して初めて走らせる (重い推論を自動実行しない / ユーザー要望)。
  const [aiRequested, setAiRequested] = useState(false);
  // 出馬表の取得完了を待って予想まで自動で進めるためのフラグ
  const [autoPredict, setAutoPredict] = useState(false);
  const [lastRunAt, setLastRunAt] = useState<Date | null>(null);
  // 出馬表を再取得した時刻 = オッズの鮮度。API に odds_fetched_at が無いので
  // この画面で取得した時刻を持つ (再訪時は分からないため null になる)。
  const [oddsFetchedAt, setOddsFetchedAt] = useState<Date | null>(null);
  // このレースだけの条件 (券種・予算)。空 = Settings の既定値をそのまま使う。
  const [overrides, setOverrides] = useState<RecommendationOverrides>({});

  const raceQuery = useRaceDetail(race_id);
  const predQuery = usePredictions(race_id, aiRequested);
  // 画面で扱う値 (使う金額 / 1 点あたり / 券種 / 狙い方) が
  // そのまま API のパラメータになる (変換なし)。
  const recQuery = useRecommendations(
    race_id,
    aiRequested && Boolean(race_id) && !raceQuery.isPending && !raceQuery.isError,
    overrides,
  );
  // runShutuba scoped to this race so raceDetail is invalidated on completion
  const runShutubaMutation = useRunShutuba(race_id);

  const race = raceQuery.data;
  const entryCount = race?.entries.length ?? 0;

  // 出馬表が入った時点で、待っていた予想を続けて実行する (B-1 ①)
  useEffect(() => {
    if (!autoPredict || entryCount === 0) return;
    setAutoPredict(false);
    setAiRequested(true);
    setLastRunAt(new Date());
    setOddsFetchedAt(new Date());
  }, [autoPredict, entryCount]);

  // NOTE: 出馬表取込・AI 予想はいずれも画面表示時に自動実行しない。
  // すべて下部の各ボタン (出馬表を取得 / AI 予想を実行) で明示的に開始する。

  // Race ページの Past タブへ戻る (?tab=past)。date を引き継いで一覧の選択日を復元。
  // 旧 `/past` は /races へ redirect され query を落とすため直接 /races を指す。
  const backLink = dateParam
    ? `/races?tab=past&date=${dateParam}`
    : '/races?tab=past';

  if (raceQuery.isPending) {
    return (
      <div className="flex flex-col gap-8 p-6">
        <BackLink to={backLink} />
        <PageHeader
          marker={<RaceNumber raceId={race_id} />}
          eyebrow="Race Detail"
          title="レース詳細"
          description={race_id}
        />
        <RaceDetailSkeleton />
      </div>
    );
  }

  if (raceQuery.isError) {
    const is404 = isNotFoundError(raceQuery.error);
    return (
      <div className="flex flex-col gap-8 p-6">
        <BackLink to={backLink} />
        <PageHeader
          marker={<RaceNumber raceId={race_id} />}
          eyebrow="Race Detail"
          title="レース詳細"
          description={race_id}
        />
        <EmptyState
          message={is404 ? '指定レース ID は見つかりません' : 'レース詳細の取得に失敗しました'}
          description={is404 ? undefined : 'バックエンドが起動しているか確認してください。'}
        />
        {is404 && (
          <div className="flex justify-center">
            <Button asChild variant="outline">
              <Link to="/upcoming">Upcoming Races へ戻る</Link>
            </Button>
          </div>
        )}
      </div>
    );
  }

  // raceQuery が success であっても TanStack Query の型は data: RaceDetail | undefined。
  // ここで明示的に narrowing し、以降は race を非 null として扱えるようにする。
  if (!race) {
    return (
      <div className="flex flex-col gap-8 p-6">
        <BackLink to={backLink} />
        <PageHeader
          marker={<RaceNumber raceId={race_id} />}
          eyebrow="Race Detail"
          title="レース詳細"
          description={race_id}
        />
        <RaceDetailSkeleton />
      </div>
    );
  }

  const predictions = predQuery.data?.predictions ?? null;
  const infoCoverage = predQuery.data?.info_coverage ?? null;

  const hasEntries = race.entries.length > 0;

  // **タブが立つのは買い目を描けるときだけ。** RecommendationsCard は取得中・
  // 失敗・0 件では EmptyState を出すので、そのときに出走馬一覧を渡すと消える。
  const recHasTabs =
    aiRequested &&
    !recQuery.isPending &&
    !recQuery.isError &&
    (recQuery.data?.candidates.length ?? 0) > 0;

  // 出走馬一覧の中身。タブにも単独カードにも同じものを出す。
  const entryTable = !aiRequested ? (
    // AI 予想 未実行: 実績データのみ (予想列は空欄)。
    <>
      <p className="mb-3 text-sm text-muted-foreground">
        「予想を見る」で確率と買い目を出します。
      </p>
      <EntryPredictionTable entries={race.entries} predictions={null} raceDate={race.date} />
    </>
  ) : predQuery.isPending ? (
    <TableSkeleton rows={race.entries.length || 8} />
  ) : predQuery.isError ? (
    <>
      <p className="mb-3 text-sm text-muted-foreground">
        {isServiceUnavailableError(predQuery.error)
          ? 'active モデルが見つかりません。確率の列は非表示です。'
          : '予想データを取得できません。確率の列は非表示です。'}
      </p>
      <EntryPredictionTable entries={race.entries} predictions={null} raceDate={race.date} />
    </>
  ) : (
    <EntryPredictionTable
      entries={race.entries}
      predictions={predictions}
      raceDate={race.date}
    />
  );

  const isScrapingShutuba = runShutubaMutation.isPending || runShutubaMutation.isPolling;
  const isPredicting = aiRequested && (predQuery.isFetching || recQuery.isFetching);
  // 出馬表の取得 → 予想 の一連を 1 ボタンで通す。いまどの段階かを 1 本で見せる。
  const stage: 'idle' | 'entries' | 'predict' = isScrapingShutuba
    ? 'entries'
    : isPredicting
      ? 'predict'
      : 'idle';
  const busy = stage !== 'idle';

  function handleRunShutuba(thenPredict: boolean) {
    if (thenPredict) setAutoPredict(true);
    runShutubaMutation.mutate(
      { race_ids: [race_id] },
      {
        onError: async (err) => {
          setAutoPredict(false);
          const msg = await formatErrorMessage(err);
          toast.error('出馬表の取得に失敗しました', {
            description: msg,
            action: { label: '再試行', onClick: () => handleRunShutuba(thenPredict) },
          });
        },
      }
    );
  }

  /**
   * 「予想を見る」= 出馬表が無ければ取得 → 完了を待って予想、まで連鎖させる。
   * 2 回目以降は「予想を更新」。同じ文言のまま挙動を変えない。
   */
  function handleShowPrediction() {
    if (!hasEntries) {
      handleRunShutuba(true);
      return;
    }
    if (!aiRequested) {
      setAiRequested(true);
      setLastRunAt(new Date());
      return;
    }
    predQuery.refetch();
    recQuery.refetch();
    setLastRunAt(new Date());
  }

  return (
    <div className="flex flex-col gap-8 p-6">
      <div className="flex items-center justify-between">
        <BackLink to={backLink} />
        <RaceStepper raceId={race.race_id} date={dateParam} />
      </div>

      <PageHeader
        marker={<RaceNumber raceId={race.race_id} />}
        eyebrow="Race Detail"
        title={race.name ?? `${race.course} ${race.race_class ?? ''}`.trim()}
        description={`${race.date}・${race.surface}${race.distance}m・${race.race_id}`}
      >
        {/* ボタンは 1 行に揃え、注記は下の固定高の行にまとめる。
            ボタンごとに下へ注記を付けると、注記が出た瞬間に高さが変わって
            ボタンの位置がずれてしまう。 */}
        <div className="flex flex-col items-end gap-1">
          <div className="flex items-center gap-2">
            {hasEntries && (
              <Button
                variant="outline"
                size="sm"
                disabled={busy}
                onClick={() => handleRunShutuba(false)}
                title="出馬表を再取得して単勝オッズ・人気・馬場状態を最新化します（発走が近いほど確定値に近づく）"
              >
                <RefreshCw className="mr-1.5 h-4 w-4" />
                オッズ更新
              </Button>
            )}
            <Button size="sm" disabled={busy} onClick={handleShowPrediction}>
              <Sparkles className="mr-1.5 h-4 w-4" />
              {aiRequested ? '予想を更新' : '予想を見る'}
            </Button>
          </div>
          {/* 高さは常に確保する (空でも詰めない) */}
          <div className="flex h-4 items-center gap-3 font-mono text-[10px] tabular-nums text-subtle-foreground">
            {oddsFetchedAt && <span>オッズ {formatClock(oddsFetchedAt)} 時点</span>}
            {lastRunAt && !busy && <span>予想 {formatClock(lastRunAt)}</span>}
          </div>
        </div>
      </PageHeader>

      {/* 進捗は 1 本にまとめ、いまどの段階かを言葉で出す (B-1 ①) */}
      {busy && <RunProgress stage={stage} />}

      {/* Race overview */}
      <Card className="border-t border-border pt-6">
        <CardHeader>
          <CardTitle className="text-label-ja">レース概要</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3 lg:grid-cols-4">
            <MetaItem label="レース ID" value={race.race_id} mono />
            <MetaItem label="レース名" value={race.name ?? '·'} />
            <MetaItem label="開催日" value={race.date} />
            <MetaItem label="競馬場" value={race.course} />
            <MetaItem label="馬場種別" value={race.surface} />
            <MetaItem label="距離" value={race.distance ? `${race.distance} m` : '·'} />
            <MetaItem label="天候" value={race.weather ?? '·'} />
            <MetaItem label="馬場状態" value={race.track_condition ?? '·'} />
            <MetaItem label="クラス" value={race.race_class ?? '·'} />
            <MetaItem label="出走頭数" value={race.n_runners?.toString() ?? '·'} />
            {/* 払戻はレース後にしか確定しない。未確定なら行ごと隠す
                (「単勝払戻 —」が並ぶと動いていないアプリに見えるため)。 */}
            {race.payout_win != null && (
              <MetaItem label="単勝払戻" value={formatYen(race.payout_win)} />
            )}
            {race.payout_place != null && (
              <MetaItem label="複勝払戻" value={race.payout_place} />
            )}
          </dl>
        </CardContent>
      </Card>

      {/* 出馬表が未取得のとき: ボタンで取り込む (自動取得しない) */}
      {!hasEntries && (
        <Card className="border-t border-border pt-6">
          <CardContent>
            <EmptyState
              message="出馬表がまだ取り込まれていません"
              description="下のボタンで出馬表の取得から予想までまとめて実行します。"
            >
              <Button onClick={handleShowPrediction} disabled={busy}>
                <Sparkles className="mr-1.5 h-4 w-4" />
                予想を見る
              </Button>
            </EmptyState>
          </CardContent>
        </Card>
      )}

      {hasEntries && aiRequested && !predQuery.isPending && !predQuery.isError &&
        infoCoverage?.is_low_information && (
          <LowInformationNotice coverage={infoCoverage} />
        )}

      {/* 推奨買目も表より上に置く */}
      {hasEntries && aiRequested && (
        <RecommendationsCard
          entriesTab={hasEntries ? entryTable : undefined}
          raceId={race_id}
          data={recQuery.data}
          isPending={recQuery.isPending}
          isError={recQuery.isError}
          error={recQuery.error}
          runners={race.entries.length}
          overrides={overrides}
          onOverridesChange={setOverrides}
          payouts={race.payouts ?? []}
        />
      )}

      {/* 買い目が取れないときは、出走馬一覧を単独で出す。タブは
          RecommendationsCard が買い目を描けるときだけ立つため。 */}
      {hasEntries && !recHasTabs && (
        <Card className="border-t border-border pt-6">
          <CardHeader>
            <CardTitle className="text-label-ja">出走馬一覧</CardTitle>
          </CardHeader>
          <CardContent>{entryTable}</CardContent>
        </Card>
      )}

    </div>
  );
}

interface BackLinkProps {
  to: string;
}

function BackLink({ to }: BackLinkProps) {
  return (
    <Link
      to={to}
      className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      aria-label="Past Races へ戻る"
    >
      <ChevronLeft className="h-4 w-4" />
      戻る
    </Link>
  );
}
