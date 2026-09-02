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
              <td className="py-0.5">1着確率で 1〜15 点（25% で 5 点）</td>
            </tr>
            <tr>
              <td className="py-0.5 pr-4 text-foreground">複勝</td>
              <td className="py-0.5 pr-4">
                モデル1位の馬。3着内率が {(placeMin * 100).toFixed(0)}% 以上のとき
              </td>
              <td className="py-0.5">3着内率で 1〜15 点（50% で 5 点）</td>
            </tr>
            <tr>
              <td className="py-0.5 pr-4 text-foreground">連系</td>
              <td className="py-0.5 pr-4">
                上位 3 頭で組んだ買い目のうち、的中確率が下限以上のもの
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

        <p className="leading-relaxed text-subtle-foreground">
          使う数字は 2 つだけです。<strong className="font-medium">的中確率</strong>
          （その買い目が当たる確率。買う順序を決める）と
          <strong className="font-medium">確信度</strong>
          （確率専用モデルが見た同じ確率。点数と、複勝を買うかを決める）。
          期待値（EV）は使っていません — 較正済みの確率で EV を条件にすると大穴に寄り、
          実測で単勝回収率が 0.93 → 0.70 に落ちるためです。
        </p>
        <p className="leading-relaxed text-subtle-foreground">
          <strong className="font-medium">予算は上限で、使い切りません。</strong>
          買う条件を満たす買い目が少なければ、予算が余ったまま終わります（連系を
          1 点も買わないレースは 4 分の 1 ほどあります）。前進検証 9 fold の実測は
          単勝 0.93 / 複勝 0.89（確信度で絞ると 0.92）/ 連系 0.87 で、
          <strong className="font-medium">いずれも 1.0 未満</strong>です。
        </p>
      </div>
    </details>
  );
}
