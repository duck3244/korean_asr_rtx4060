// FastAPI 백엔드 호출 래퍼. 개발 시 Vite 프록시(/api → :8000)를 통해 동작한다.

import type { Job, SystemStatus } from '../types';

const BASE = '/api';

async function parseError(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    return body?.detail || fallback;
  } catch {
    return fallback;
  }
}

export async function uploadAudio(file: File): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/transcribe`, { method: 'POST', body: form });
  if (!res.ok) {
    throw new Error(await parseError(res, `업로드 실패 (${res.status})`));
  }
  return res.json();
}

export async function getJob(jobId: string): Promise<Job> {
  const res = await fetch(`${BASE}/jobs/${jobId}`);
  if (!res.ok) {
    throw new Error(await parseError(res, `작업 조회 실패 (${res.status})`));
  }
  return res.json();
}

export async function getSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${BASE}/system/status`);
  if (!res.ok) {
    throw new Error(await parseError(res, `시스템 상태 조회 실패 (${res.status})`));
  }
  return res.json();
}

// 결과 다운로드는 <a href>로 직접 연결한다.
export function downloadUrl(jobId: string, format: string): string {
  return `${BASE}/jobs/${jobId}/download?format=${format}`;
}
