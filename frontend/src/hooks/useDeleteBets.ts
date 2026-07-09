import { useMutation, useQueryClient } from '@tanstack/react-query';
import { deleteBets, formatErrorMessage } from '@/lib/api';
import { toast } from '@/components/ui/toast';

/** 買い方（複数点）単位で購入記録をまとめて削除する。 */
export function useDeleteBets() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (ids: number[]) => deleteBets(ids),
    retry: false,
    onSuccess: () => {
      toast.success('購入記録を削除しました');
      queryClient.invalidateQueries({ queryKey: ['bets'] });
    },
    onError: async (err: unknown) => {
      const msg = await formatErrorMessage(err);
      toast.error(msg);
    },
  });
}
