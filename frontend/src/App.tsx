import { useState } from 'react';
import Uploader from './components/Uploader';
import JobProgress from './components/JobProgress';
import ResultView from './components/ResultView';
import SystemPanel from './components/SystemPanel';
import { useJob } from './hooks/useJob';

export default function App() {
  const [jobId, setJobId] = useState<string | null>(null);
  const { data: job, error } = useJob(jobId);

  const isRunning =
    !!job && (job.status === 'pending' || job.status === 'processing');

  return (
    <div className="min-h-screen bg-slate-100 text-slate-800">
      <header className="bg-slate-900 text-white">
        <div className="mx-auto max-w-3xl px-6 py-5">
          <h1 className="text-xl font-bold">🇰🇷 한국어 음성인식</h1>
          <p className="text-sm text-slate-300">RTX 4060 · wav2vec2 기반 ASR</p>
        </div>
      </header>

      <main className="mx-auto max-w-3xl space-y-6 px-6 py-8">
        <SystemPanel />

        <Uploader onUploaded={setJobId} disabled={isRunning} />

        {error && (
          <div className="rounded-lg bg-red-100 px-4 py-3 text-sm text-red-700">
            오류: {(error as Error).message}
          </div>
        )}

        {isRunning && job && <JobProgress job={job} />}

        {job && job.status === 'failed' && (
          <div className="rounded-lg bg-red-100 px-4 py-3 text-sm text-red-700">
            전사 실패: {job.error ?? '알 수 없는 오류'}
          </div>
        )}

        {job && job.status === 'completed' && <ResultView job={job} />}
      </main>
    </div>
  );
}
