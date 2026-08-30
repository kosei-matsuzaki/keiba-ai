import { useMutation } from '@tanstack/react-query';
import { evaluateModel } from '@/lib/api';

/**
 * モデルを実運用の賭けルールで測り直す (backtest --persist 相当)。
 *
 * 5,000 レース規模で 10 分前後かかるので、返るのは job_id だけ。進捗は
 * JobProgressCard が polling する。
 */
export function useEvaluateModel() {
  return useMutation({
    mutationFn: (id: number) => evaluateModel(id),
  });
}
