import { createBet } from '@/lib/api';
import type { BetRecordIn } from '@/types/api';

import { useBetMutation } from './useBetMutation';

export function useCreateBet() {
  return useBetMutation((body: BetRecordIn) => createBet(body), '買目を記録しました');
}
