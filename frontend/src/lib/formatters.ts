/**
 * Display formatters used across the app.
 *
 * Goals:
 *   - Single source of truth for number / date / currency rendering, so the
 *     same value never appears as 1.882 in one card and 1.88 in another.
 *   - NaN / null / undefined consistently render as the same placeholder.
 *   - Locale-aware where it matters (digit grouping, percent symbol).
 *
 * Usage from components:
 *   formatRatio(0.882) → "0.882"
 *   formatPercent(0.625) → "62.5%"
 *   formatCount(1690) → "1,690"
 *   formatYen(12560) → "12,560 円"
 *   formatDateTime("2024-12-28T10:00:00Z") → "2024-12-28 10:00"
 *   formatDate("2024-12-28") → "2024-12-28"
 */

const PLACEHOLDER = '—';

const numberFormatter = new Intl.NumberFormat('ja-JP');

function isFinite(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/**
 * 小数 3 桁固定でランキングスコア・回収率の表示に使う。
 * NaN / Infinity / null / undefined は PLACEHOLDER。
 */
export function formatScore(value: number | null | undefined, digits = 3): string {
  if (!isFinite(value)) return PLACEHOLDER;
  return value.toFixed(digits);
}

/**
 * 回収率・倍率系。「1.05」「0.882」のように小数 2 桁固定。
 * 1.0 が損益分岐なのでそれが直感的に読める桁数。
 */
export function formatRatio(value: number | null | undefined, digits = 2): string {
  if (!isFinite(value)) return PLACEHOLDER;
  return value.toFixed(digits);
}

/**
 * 0〜1 の小数を「62.5%」のように表示する。
 * 例: formatPercent(0.625, 1) → "62.5%"。
 */
export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (!isFinite(value)) return PLACEHOLDER;
  return `${(value * 100).toFixed(digits)}%`;
}

/**
 * 整数または小数の 3 桁区切り表示。
 * 例: formatCount(1690) → "1,690"。NaN なら PLACEHOLDER。
 */
export function formatCount(value: number | null | undefined): string {
  if (!isFinite(value)) return PLACEHOLDER;
  return numberFormatter.format(value);
}

/**
 * 金額表示。3 桁区切り + 「円」suffix。
 * 例: formatYen(12560) → "12,560 円"。
 */
export function formatYen(value: number | null | undefined): string {
  if (!isFinite(value)) return PLACEHOLDER;
  return `${numberFormatter.format(value)} 円`;
}

/**
 * オッズ表記 (小数 1 桁)。
 * 例: formatOdds(3.1) → "3.1"。
 */
export function formatOdds(value: number | null | undefined): string {
  if (!isFinite(value)) return PLACEHOLDER;
  return value.toFixed(1);
}

/**
 * ISO 8601 文字列を「2024-12-28 10:00」形式の日時文字列にする。
 * タイムゾーンを変換せず、文字列の先頭 16 文字を整形して返す軽量実装。
 * 不正値や空文字は PLACEHOLDER。
 */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso || typeof iso !== 'string' || iso.length < 16) return PLACEHOLDER;
  // "2024-12-28T10:00:00Z" → "2024-12-28 10:00"
  return `${iso.slice(0, 10)} ${iso.slice(11, 16)}`;
}

/**
 * 「2024-12-28」のような日付のみ表示。
 * ISO 文字列を受けたら頭 10 文字を取る。
 */
export function formatDate(value: string | null | undefined): string {
  if (!value || typeof value !== 'string' || value.length < 10) return PLACEHOLDER;
  return value.slice(0, 10);
}
