import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { fetchSettings, updateSettings } from '@/lib/api';
import type { SettingsUpdate } from '@/types/api';

export function useSettings() {
  return useQuery({
    queryKey: ['settings'],
    queryFn: fetchSettings,
  });
}

export function useUpdateSettings() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (body: SettingsUpdate) => updateSettings(body),
    onSuccess: (data) => {
      queryClient.setQueryData(['settings'], data);
    },
  });
}
