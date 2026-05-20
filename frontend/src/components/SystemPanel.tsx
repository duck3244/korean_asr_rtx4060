import { useQuery } from '@tanstack/react-query';
import { getSystemStatus } from '../api/client';
import type { SystemStatus } from '../types';

export default function SystemPanel() {
  const { data, error } = useQuery<SystemStatus>({
    queryKey: ['system-status'],
    queryFn: getSystemStatus,
    refetchInterval: 5000,
  });

  if (error) {
    return (
      <section className="rounded-xl bg-amber-50 p-4 text-sm text-amber-700">
        백엔드에 연결할 수 없습니다. uvicorn 서버(:8000) 실행을 확인하세요.
      </section>
    );
  }

  if (!data) {
    return (
      <section className="rounded-xl bg-white p-4 text-sm text-slate-400 shadow-sm">
        시스템 상태 확인 중...
      </section>
    );
  }

  const allocated = data.memory.gpu_allocated_gb ?? 0;
  const total = data.total_vram_gb || 1;
  const vramPct = Math.min(100, Math.round((allocated / total) * 100));

  return (
    <section className="rounded-xl bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <span className="flex items-center gap-1.5">
          <span
            className={`h-2 w-2 rounded-full ${
              data.model_loaded ? 'bg-green-500' : 'bg-slate-300'
            }`}
          />
          {data.model_loaded ? '모델 로드됨' : '모델 미로드'}
        </span>
        <span className="text-slate-500">기기: {data.device}</span>
        <span className="text-slate-500">진행 중 작업: {data.active_jobs}</span>
      </div>

      <div className="mt-3">
        <div className="mb-1 flex justify-between text-xs text-slate-500">
          <span>VRAM</span>
          <span>
            {allocated.toFixed(1)} / {total.toFixed(1)} GB
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
          <div
            className="h-full bg-emerald-500"
            style={{ width: `${vramPct}%` }}
          />
        </div>
      </div>
    </section>
  );
}
