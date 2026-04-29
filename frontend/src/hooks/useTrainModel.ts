import { useMutation } from '@tanstack/react-query';
import { trainModel } from '@/lib/api';
import type { TrainRequest } from '@/types/api';

export function useTrainModel() {
  return useMutation({
    mutationFn: (body: TrainRequest) => trainModel(body),
  });
}
