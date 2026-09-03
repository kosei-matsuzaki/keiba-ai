import { useMutation, useQueryClient } from '@tanstack/react-query';

import { formatErrorMessage } from '@/lib/api';
import { toast } from '@/lib/toast';

/**
 * 購入記録を変える mutation の共通形。
 *
 * 記録・まとめ記録・削除の 3 つは、成功トーストの文言以外がまったく同じだった
 * (retry なし / 成功したら `['bets']` を無効化 / 失敗したらエラーを整形して
 * トースト)。失敗時の扱いや無効化するキーを変えるときは 3 つとも同時に変わる
 * ので、ここに 1 つだけ置く。
 *
 * `success` を関数で渡せるのは、まとめ記録だけが結果の件数を文言に混ぜるため。
 */
export function useBetMutation<TArgs, TResult>(
  mutationFn: (args: TArgs) => Promise<TResult>,
  success: string | ((result: TResult) => string)
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn,
    retry: false,
    onSuccess: (result: TResult) => {
      toast.success(typeof success === 'function' ? success(result) : success);
      queryClient.invalidateQueries({ queryKey: ['bets'] });
    },
    onError: async (err: unknown) => {
      toast.error(await formatErrorMessage(err));
    },
  });
}
