import { useMutation, useQueryClient } from '@tanstack/react-query';
import { deleteModel } from '@/lib/api';

export function useDeleteModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => deleteModel(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });
}
