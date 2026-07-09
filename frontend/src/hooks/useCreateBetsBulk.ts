import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createBetsBulk, formatErrorMessage } from '@/lib/api';
import { toast } from '@/components/ui/toast';
import type { BetRecordBulkIn } from '@/types/api';

/** 流し/ボックス/フォーメーションを展開した複数点をまとめて登録する。 */
export function useCreateBetsBulk() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (body: BetRecordBulkIn) => createBetsBulk(body),
    retry: false,
    onSuccess: (res) => {
      toast.success(`${res.total} 点を記録しました`);
      queryClient.invalidateQueries({ queryKey: ['bets'] });
    },
    onError: async (err: unknown) => {
      const msg = await formatErrorMessage(err);
      toast.error(msg);
    },
  });
}
