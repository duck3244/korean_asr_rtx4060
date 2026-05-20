import { useRef, useState } from 'react';
import { uploadAudio } from '../api/client';

interface Props {
  onUploaded: (jobId: string) => void;
  disabled?: boolean;
}

const ACCEPT = '.wav,.mp3,.m4a,.flac,.ogg,.aac';

export default function Uploader({ onUploaded, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const { job_id } = await uploadAudio(file);
      onUploaded(job_id);
      setFile(null);
      if (inputRef.current) inputRef.current.value = '';
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <section className="rounded-xl bg-white p-6 shadow-sm">
      <h2 className="mb-3 text-lg font-semibold">오디오 파일 전사</h2>

      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const dropped = e.dataTransfer.files?.[0];
          if (dropped) setFile(dropped);
        }}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-10 text-center transition ${
          dragOver ? 'border-blue-500 bg-blue-50' : 'border-slate-300'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <span className="text-sm text-slate-600">
          {file ? `📄 ${file.name}` : '파일을 드래그하거나 클릭해서 선택'}
        </span>
        <span className="mt-1 text-xs text-slate-400">
          지원: WAV, MP3, M4A, FLAC, OGG, AAC (최대 200MB)
        </span>
      </label>

      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}

      <button
        onClick={handleUpload}
        disabled={!file || uploading || disabled}
        className="mt-4 w-full rounded-lg bg-blue-600 px-4 py-2.5 font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        {uploading ? '업로드 중...' : disabled ? '전사 진행 중...' : '전사 시작'}
      </button>
    </section>
  );
}
