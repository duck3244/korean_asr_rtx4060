import type { Job } from '../types';

export default function JobProgress({ job }: { job: Job }) {
  const pct = Math.round(job.progress * 100);

  return (
    <section className="rounded-xl bg-white p-6 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-lg font-semibold">
          {job.status === 'pending' ? '전사 대기 중' : '전사 진행 중'}
        </h2>
        <span className="text-sm text-slate-500">
          {job.processed_chunks}/{job.total_chunks || '?'} 청크
        </span>
      </div>

      <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200">
        <div
          className="h-full bg-blue-600 transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>

      <p className="mt-2 text-sm text-slate-500">
        {pct}% — {job.filename}
      </p>
    </section>
  );
}
