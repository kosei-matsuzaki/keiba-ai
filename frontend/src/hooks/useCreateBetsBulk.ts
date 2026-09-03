import { createBetsBulk } from '@/lib/api';
import type { BetRecordBulkIn } from '@/types/api';

import { useBetMutation } from './useBetMutation';

/** 流し/ボックス/フォーメーションを展開した複数点をまとめて登録する。 */
export function useCreateBetsBulk() {
  return useBetMutation(
    (body: BetRecordBulkIn) => createBetsBulk(body),
    (res) => `${res.total} 点を記録しました`
  );
}
