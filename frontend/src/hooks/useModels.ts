import { useQuery } from '@tanstack/react-query';
import { fetchModels } from '@/lib/api';

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: fetchModels,
  });
}
