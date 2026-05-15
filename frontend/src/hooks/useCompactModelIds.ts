import { useMutation, useQueryClient } from '@tanstack/react-query';
import { compactModelIds } from '@/lib/api';

export function useCompactModelIds() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => compactModelIds(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });
}
