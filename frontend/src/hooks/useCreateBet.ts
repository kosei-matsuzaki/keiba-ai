import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createBet } from '@/lib/api';
import { toast } from '@/components/ui/toast';
import { formatErrorMessage } from '@/lib/api';
import type { BetRecordIn } from '@/types/api';

export function useCreateBet() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (body: BetRecordIn) => createBet(body),
    retry: false,
    onSuccess: () => {
      toast.success('買目を記録しました');
      // Invalidate bet list cache so any open bets page refreshes
      queryClient.invalidateQueries({ queryKey: ['bets'] });
    },
    onError: async (err: unknown) => {
      const msg = await formatErrorMessage(err);
      toast.error(msg);
    },
  });
}
