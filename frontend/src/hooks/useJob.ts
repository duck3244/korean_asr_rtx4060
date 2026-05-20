import { useQuery } from '@tanstack/react-query';
import { getJob } from '../api/client';
import type { Job } from '../types';

// 전사 작업 상태를 폴링한다. 완료/실패 시 폴링을 멈춘다.
export function useJob(jobId: string | null) {
  return useQuery<Job>({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId as string),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'completed' || status === 'failed' ? false : 1500;
    },
  });
}
