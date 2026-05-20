import type { Job } from '../types';
import { downloadUrl } from '../api/client';

const FORMATS = ['txt', 'srt', 'json', 'csv'] as const;

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function ResultView({ job }: { job: Job }) {
  return (
    <section className="rounded-xl bg-white p-6 shadow-sm">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">전사 결과</h2>
        <div className="flex gap-2">
          {FORMATS.map((fmt) => (
            <a
              key={fmt}
              href={downloadUrl(job.job_id, fmt)}
              className="rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
            >
              {fmt.toUpperCase()}
            </a>
          ))}
        </div>
      </div>

      <p className="whitespace-pre-wrap rounded-lg bg-slate-50 p-4 text-sm leading-relaxed">
        {job.text || '(빈 결과)'}
      </p>

      {job.chunks && job.chunks.length > 0 && (
        <div className="mt-4">
          <h3 className="mb-2 text-sm font-semibold text-slate-600">
            구간별 타임스탬프
          </h3>
          <ul className="divide-y divide-slate-100 text-sm">
            {job.chunks.map((chunk) => (
              <li key={chunk.index} className="flex gap-3 py-1.5">
                <span className="shrink-0 font-mono text-xs text-slate-400">
                  {formatTime(chunk.start_time)}–{formatTime(chunk.end_time)}
                </span>
                <span>{chunk.text}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
