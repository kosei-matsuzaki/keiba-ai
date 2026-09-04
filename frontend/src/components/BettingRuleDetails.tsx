import { useSettings } from '@/hooks/useSettings';

const COMBO_LABELS: Record<string, string> = {
  馬連: '馬連',
  ワイド: 'ワイド',
  馬単: '馬単',
  三連複: '三連複',
  三連単: '三連単',
};

/**
 * 買い方の説明を **1 箇所** にまとめ、既定では畳んでおく。
 *
 * 以前は「BUY バッジ」「オッズの出所」「確信度」「並び順」「賭け金の決め方」
 * 「点数の決め方」が別々の注記として画面のあちこちに散らばっていた。毎回読む
 * ものではないのに常時出ているせいで、肝心の買い目が埋もれていた。
 * **必要なときに開く 1 つの引き出し**にする。
 *
 * 数字は設定から引く。ここに直書きすると設定を変えたときに嘘になる。
 */
export function BettingRuleDetails() {
  const { data: settings } = useSettings();
  const minOdds = settings?.win_min_odds ?? 1.1;
  const placeMin = settings?.place_min_hit_prob ?? 0.6;
  const floors = settings?.combo_min_hit_prob ?? {};
  const floorText = Object.entries(COMBO_LABELS)
    .map(([key, label]) =>
      floors[key] != null ? `${label} ${(floors[key] * 100).toFixed(1)}%` : null
    )
    .filter(Boolean)
    .join(' / ');

  return (
    <details className="group border-t border-border pt-3 text-xs">
      <summary className="cursor-pointer list-none text-label-ja text-muted-foreground transition-colors hover:text-foreground">
        買い方（券種ごとの条件と点数）
        <span className="ml-2 font-mono text-[10px] text-subtle-foreground group-open:hidden">
          ＋
        </span>
        <span className="ml-2 hidden font-mono text-[10px] text-subtle-foreground group-open:inline">
          −
        </span>
      </summary>

      <div className="mt-3 flex flex-col gap-3">
        <table className="w-full max-w-3xl text-left">
          <thead className="text-subtle-foreground">
            <tr>
              <th className="py-0.5 pr-4 font-normal">券種</th>
              <th className="py-0.5 pr-4 font-normal">買う条件</th>
              <th className="py-0.5 font-normal">点数（1 点 = 100 円）</th>
            </tr>
          </thead>
          <tbody className="text-muted-foreground">
            <tr>
              <td className="py-0.5 pr-4 text-foreground">単勝</td>
              <td className="py-0.5 pr-4">モデル1位の馬。オッズ {minOdds} 倍超のとき</td>
              <td className="py-0.5">確信度で 1〜15 点（25% のとき 5 点）</td>
            </tr>
            <tr>
              <td className="py-0.5 pr-4 text-foreground">複勝</td>
              <td className="py-0.5 pr-4">
                モデル1位の馬。確信度が {(placeMin * 100).toFixed(0)}% 以上のとき
              </td>
              <td className="py-0.5">確信度で 1〜15 点（50% のとき 5 点）</td>
            </tr>
            <tr>
              <td className="py-0.5 pr-4 text-foreground">連系</td>
              <td className="py-0.5 pr-4">
                上位 3 頭で組んだ買い目のうち、確信度が下限以上のもの
              </td>
              <td className="py-0.5">下限を超えた買い目を 1 点ずつ（上限なし）</td>
            </tr>
          </tbody>
        </table>

        {floorText && (
          <p className="text-subtle-foreground">
            連系の下限：{floorText}
            <span className="ml-1">（Settings で変えられます）</span>
          </p>
        )}

        {/* 用語と作り方。**文章にしない** — 読み飛ばせる行の集まりにする */}
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-subtle-foreground">
          <dt className="text-foreground">確信度</dt>
          <dd>
            その買い目が当たる確率。単勝＝1着 / 複勝＝3着以内 / 連系＝その組合せ。
            券種が違っても意味は同じ
          </dd>

          <dt className="text-foreground">点数</dt>
          <dd>
            <span className="font-mono">5 ×（確信度 ÷ 基準）²</span> を 1〜15 点に。
            基準は単勝 25% / 複勝 50%。連系は 1 組合せ 1 点で、
            <strong className="font-medium">点数＝下限を超えた買い目の数</strong>
          </dd>

          <dt className="text-foreground">作り方</dt>
          <dd>
            スコア → 温度較正 → 1〜3着の並びを 1 万回サンプリング → 出た割合。
            出走馬表の 1着確率 / 3着内率 は単勝 / 複勝の確信度と同じ数字
          </dd>

          <dt className="text-foreground">2 つのモデル</dt>
          <dd>
            馬を選ぶのは買い目モデル（回収率で学習。確率は不正確で、本命の確率と
            勝敗の相関 0.07）。確信度を答えるのは確率モデル（同 0.24）。
            <strong className="font-medium">連系は最初から確率モデル</strong>なので
            「確率モデル」列は「同じ」になる
          </dd>

          <dt className="text-foreground">EV</dt>
          <dd>
            使わない。較正済みの確率で EV を条件にすると大穴に寄り、単勝回収率が
            0.93 → 0.70 に落ちる。表の EV は参考値
          </dd>

          <dt className="text-foreground">予算</dt>
          <dd>
            上限であって使い切らない。条件を満たす買い目が少なければ余る
            （連系を 1 点も買わないレースが 4 分の 1 ほど）
          </dd>

          <dt className="text-foreground">実測</dt>
          <dd>
            単勝 0.91 / 複勝 0.92（2024-10〜2026-08・6,244 レース）。
            <strong className="font-medium">どちらも 1.0 未満</strong>
          </dd>
        </dl>
      </div>
    </details>
  );
}
