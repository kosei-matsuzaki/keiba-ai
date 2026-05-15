import { useMutation, useQueryClient } from '@tanstack/react-query';
import { updateModel } from '@/lib/api';
import type { UpdateModelRequest } from '@/types/api';

export function useUpdateModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, body }: { id: number; body: UpdateModelRequest }) =>
      updateModel(id, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });
}
