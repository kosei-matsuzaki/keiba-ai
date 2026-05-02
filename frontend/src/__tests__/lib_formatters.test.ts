import { describe, it, expect } from 'vitest';

import {
  formatCount,
  formatDate,
  formatDateTime,
  formatOdds,
  formatPercent,
  formatRatio,
  formatScore,
  formatYen,
} from '@/lib/formatters';

const PLACEHOLDER = '—';

describe('formatScore', () => {
  it('renders 3 digits by default', () => {
    expect(formatScore(0.123456)).toBe('0.123');
  });
  it('honours custom digits', () => {
    expect(formatScore(0.123456, 4)).toBe('0.1235');
  });
  it.each([null, undefined, NaN, Infinity, -Infinity])(
    'renders placeholder for %s',
    (input) => {
      expect(formatScore(input as number | null | undefined)).toBe(PLACEHOLDER);
    },
  );
});

describe('formatRatio', () => {
  it('uses 2 digits by default — ROI readability around 1.0', () => {
    expect(formatRatio(0.882)).toBe('0.88');
    expect(formatRatio(1.05)).toBe('1.05');
  });
  it.each([null, undefined, NaN])('renders placeholder for %s', (v) => {
    expect(formatRatio(v as number | null | undefined)).toBe(PLACEHOLDER);
  });
});

describe('formatPercent', () => {
  it('multiplies by 100 and adds %', () => {
    expect(formatPercent(0.625)).toBe('62.5%');
  });
  it('honours digits arg', () => {
    expect(formatPercent(0.625, 2)).toBe('62.50%');
  });
  it('handles 0', () => {
    expect(formatPercent(0)).toBe('0.0%');
  });
  it('handles invalid', () => {
    expect(formatPercent(null)).toBe(PLACEHOLDER);
    expect(formatPercent(NaN)).toBe(PLACEHOLDER);
  });
});

describe('formatCount', () => {
  it('groups large numbers with comma', () => {
    expect(formatCount(1690)).toBe('1,690');
    expect(formatCount(1_234_567)).toBe('1,234,567');
  });
  it('handles 0 and negatives', () => {
    expect(formatCount(0)).toBe('0');
    expect(formatCount(-42)).toBe('-42');
  });
  it('placeholder for invalid', () => {
    expect(formatCount(null)).toBe(PLACEHOLDER);
    expect(formatCount(NaN)).toBe(PLACEHOLDER);
  });
});

describe('formatYen', () => {
  it('appends 円 with grouping', () => {
    expect(formatYen(12560)).toBe('12,560 円');
  });
  it('placeholder for invalid', () => {
    expect(formatYen(null)).toBe(PLACEHOLDER);
  });
});

describe('formatOdds', () => {
  it('1 decimal digit', () => {
    expect(formatOdds(3.14)).toBe('3.1');
    expect(formatOdds(10)).toBe('10.0');
  });
  it('placeholder for invalid', () => {
    expect(formatOdds(null)).toBe(PLACEHOLDER);
    expect(formatOdds(undefined)).toBe(PLACEHOLDER);
  });
});

describe('formatDateTime', () => {
  it('converts ISO 8601 to space-separated form', () => {
    expect(formatDateTime('2024-12-28T10:00:00Z')).toBe('2024-12-28 10:00');
    expect(formatDateTime('2024-12-28T10:30:45.123Z')).toBe('2024-12-28 10:30');
  });
  it('placeholder for too-short string', () => {
    expect(formatDateTime('2024-12-28')).toBe(PLACEHOLDER);
    expect(formatDateTime('')).toBe(PLACEHOLDER);
    expect(formatDateTime(null)).toBe(PLACEHOLDER);
  });
});

describe('formatDate', () => {
  it('truncates to YYYY-MM-DD', () => {
    expect(formatDate('2024-12-28T10:00:00Z')).toBe('2024-12-28');
    expect(formatDate('2024-12-28')).toBe('2024-12-28');
  });
  it('placeholder for short or missing', () => {
    expect(formatDate('2024')).toBe(PLACEHOLDER);
    expect(formatDate(null)).toBe(PLACEHOLDER);
  });
});
