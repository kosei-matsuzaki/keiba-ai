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

        <p className="leading-relaxed text-subtle-foreground">
          <strong className="font-medium">確信度＝その買い目が当たる確率。</strong>
          単勝なら 1 着になる確率、複勝なら 3 着以内に入る確率、連系ならその組合せで
          決まる確率です。券種が違っても意味は同じなので、横に並べて比べられます。
          点数の式は <span className="font-mono">5 ×（確信度 ÷ 基準）²</span> を
          1〜15 点に丸めたもの（基準は単勝 25% / 複勝 50%）。連系だけは 1 組合せ 1 点で、
          <strong className="font-medium">何点買うかは下限を超えた買い目の数</strong>が
          決めます（レースごとに変わり、1 点も買わないレースが 4 分の 1 ほどあります）。
        </p>

        <p className="leading-relaxed text-subtle-foreground">
          <strong className="font-medium">確信度は 2 つのモデルが出します。</strong>
          買い目を選ぶのは「買い目モデル」、確信度を答えるのは「確率モデル」です。
          表の <span className="font-mono">確信度</span> 列は買う順序を決める値、
          <span className="font-mono">確率モデル</span> 列は同じ量を確率専用モデルが
          答えたもの。<strong className="font-medium">連系は最初から確率モデルが
          出している</strong>ので、そこは「同じ」と表示します。単複だけ 2 つ並ぶのは、
          買い目モデルが回収率で学習していて確率が正確でないためです（本命の確率と
          勝敗の相関は 0.07、確率モデルは 0.24）。
        </p>

        <p className="leading-relaxed text-subtle-foreground">
          <strong className="font-medium">確信度の作り方。</strong>
          モデルが出した馬ごとのスコアを較正（温度スケーリング）してから、
          1 着〜3 着の並びを 1 万回サンプリングします。単勝はそのうち 1 着だった割合、
          複勝は 3 着以内に入った割合、連系はその組合せが出た割合です。
          出走馬表の <span className="font-mono">1着確率</span> と
          <span className="font-mono">3着内率</span> は、単勝・複勝の確信度と同じ数字です。
        </p>

        <p className="leading-relaxed text-subtle-foreground">
          <strong className="font-medium">期待値（EV）は使っていません。</strong>
          較正済みの確率で EV を条件にすると大穴に寄り、実測で単勝回収率が
          0.93 → 0.70 に落ちるためです。表の EV 列は参考値です。
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
