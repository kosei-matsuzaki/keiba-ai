import { useSettings } from '@/hooks/useSettings';

/**
 * 連系の券種と、その確信度が何を数えた割合か。
 *
 * **全券種を 1 行ずつ出す。** 「連系」でまとめると、下限も数え方も券種で違うのに
 * 1 行に収まらず、結局表の外に書くことになる。
 */
const COMBO_ROWS: ReadonlyArray<{ key: string; counted: string }> = [
  { key: '馬連', counted: '1-2着がこの2頭' },
  { key: 'ワイド', counted: '3着以内にこの2頭' },
  { key: '馬単', counted: '1着→2着がこの順' },
  { key: '三連複', counted: '3着以内がこの3頭' },
  { key: '三連単', counted: '1-2-3着がこの順' },
];

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
        {/* **券種ごとに全部この表で読み切れること。** 確信度の中身も点数の式も
            券種で変わるので、別行に出すと対応を取りながら読む羽目になる */}
        <table className="w-full text-left">
          <thead className="text-subtle-foreground">
            <tr>
              <th className="py-0.5 pr-4 font-normal">券種</th>
              <th className="py-0.5 pr-4 font-normal">買う条件</th>
              <th className="py-0.5 pr-4 font-normal">確信度＝抽選で数える割合</th>
              <th className="py-0.5 font-normal">点数（1 点 = 100 円）</th>
            </tr>
          </thead>
          <tbody className="text-muted-foreground">
            <tr>
              <td className="py-0.5 pr-4 text-foreground">単勝</td>
              <td className="py-0.5 pr-4">モデル1位。オッズ {minOdds} 倍超</td>
              <td className="py-0.5 pr-4">1着だった割合</td>
              <td className="py-0.5">
                <span className="font-mono">5 ×（確信度 ÷ 25%）²</span> → 1〜15 点
              </td>
            </tr>
            <tr>
              <td className="py-0.5 pr-4 text-foreground">複勝</td>
              <td className="py-0.5 pr-4">
                モデル1位。確信度 {(placeMin * 100).toFixed(0)}% 以上
              </td>
              <td className="py-0.5 pr-4">3着以内だった割合</td>
              <td className="py-0.5">
                <span className="font-mono">5 ×（確信度 ÷ 50%）²</span> → 1〜15 点
              </td>
            </tr>
            {COMBO_ROWS.map(({ key, counted }) => (
              <tr key={key}>
                <td className="py-0.5 pr-4 text-foreground">{key}</td>
                <td className="py-0.5 pr-4">
                  上位3頭の組合せ。確信度{' '}
                  {floors[key] != null ? `${(floors[key] * 100).toFixed(1)}%` : '下限'} 以上
                </td>
                <td className="py-0.5 pr-4">{counted}</td>
                <td className="py-0.5">1 組合せ 1 点（上限なし）</td>
              </tr>
            ))}
          </tbody>
        </table>

        <p className="text-subtle-foreground">
          確信度はモデルのスコアを温度較正し、
          <strong className="font-medium">1〜3着の並びを 1 万回サンプリング</strong>
          して上の割合を数えたもの。出走馬表の 1着確率 / 3着内率 は単勝 / 複勝の
          確信度と同じ数字です。連系の下限は Settings で変えられます。
        </p>

        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-subtle-foreground">
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
        </dl>
      </div>
    </details>
  );
}
