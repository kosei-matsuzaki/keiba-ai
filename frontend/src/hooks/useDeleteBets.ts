import { deleteBets } from '@/lib/api';

import { useBetMutation } from './useBetMutation';

/** 買い方（複数点）単位で購入記録をまとめて削除する。 */
export function useDeleteBets() {
  return useBetMutation((ids: number[]) => deleteBets(ids), '購入記録を削除しました');
}
