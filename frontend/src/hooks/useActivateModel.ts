import { useMutation, useQueryClient } from '@tanstack/react-query';
import { activateModel } from '@/lib/api';

export function useActivateModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => activateModel(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });
}
