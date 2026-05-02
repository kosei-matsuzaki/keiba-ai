import { HTTPError } from 'ky';
import { describe, expect, it } from 'vitest';

import {
  formatErrorMessage,
  formatErrorMessageSync,
  getStatus,
  isNotFoundError,
  isServiceUnavailableError,
  isValidationError,
} from '@/lib/api';

function makeHTTPError(status: number, body?: unknown): HTTPError {
  const response = new Response(body ? JSON.stringify(body) : '', {
    status,
    headers: body ? { 'content-type': 'application/json' } : undefined,
  });
  // ky's HTTPError signature: (response, request, options)
  return new HTTPError(
    response,
    new Request('https://example.test/'),
    { method: 'GET' } as never,
  );
}

describe('getStatus', () => {
  it('extracts status from ky HTTPError', () => {
    expect(getStatus(makeHTTPError(404))).toBe(404);
    expect(getStatus(makeHTTPError(503))).toBe(503);
  });

  it('extracts status from a plain object with .status', () => {
    expect(getStatus({ status: 404 })).toBe(404);
  });

  it('falls back to parsing status from message', () => {
    expect(getStatus(new Error('Request failed with status 503'))).toBe(503);
  });

  it('returns null for unknown shapes', () => {
    expect(getStatus(null)).toBeNull();
    expect(getStatus(undefined)).toBeNull();
    expect(getStatus('plain string')).toBeNull();
    expect(getStatus(new Error('no digits'))).toBeNull();
  });
});

describe('status type guards', () => {
  it('isNotFoundError detects 404 only', () => {
    expect(isNotFoundError(makeHTTPError(404))).toBe(true);
    expect(isNotFoundError(makeHTTPError(503))).toBe(false);
    expect(isNotFoundError(null)).toBe(false);
  });

  it('isServiceUnavailableError detects 503 only', () => {
    expect(isServiceUnavailableError(makeHTTPError(503))).toBe(true);
    expect(isServiceUnavailableError(makeHTTPError(404))).toBe(false);
  });

  it('isValidationError detects 400 and 422', () => {
    expect(isValidationError(makeHTTPError(400))).toBe(true);
    expect(isValidationError(makeHTTPError(422))).toBe(true);
    expect(isValidationError(makeHTTPError(404))).toBe(false);
  });
});

describe('formatErrorMessage', () => {
  it('returns Japanese mapped message for known status', async () => {
    expect(await formatErrorMessage(makeHTTPError(404))).toContain('対象が見つかりません');
    expect(await formatErrorMessage(makeHTTPError(503))).toContain('利用できません');
  });

  it('appends FastAPI detail when present', async () => {
    const err = makeHTTPError(400, { detail: 'date format invalid' });
    const msg = await formatErrorMessage(err);
    expect(msg).toContain('入力内容に誤りがあります');
    expect(msg).toContain('date format invalid');
  });

  it('handles 422 with array detail (pydantic)', async () => {
    const err = makeHTTPError(422, {
      detail: [{ loc: ['body', 'minutes'], msg: 'value must be ≥ 1', type: 'value_error' }],
    });
    const msg = await formatErrorMessage(err);
    expect(msg).toContain('再確認');
    expect(msg).toContain('value must be ≥ 1');
  });

  it('falls back to message for plain Error', async () => {
    expect(await formatErrorMessage(new Error('Network down'))).toBe('Network down');
  });

  it('handles unknown', async () => {
    expect(await formatErrorMessage('weird value')).toBe('不明なエラーが発生しました');
  });
});

describe('formatErrorMessageSync', () => {
  it('handles known status without async', () => {
    expect(formatErrorMessageSync(makeHTTPError(404))).toContain('見つかりません');
  });

  it('returns plain Error message', () => {
    expect(formatErrorMessageSync(new Error('boom'))).toBe('boom');
  });
});
