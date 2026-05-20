// 백엔드 JSON 응답 형태 (backend/src/api/jobs.py · routes.py 와 대응)

export type JobStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface Chunk {
  index: number;
  start_time: number;
  end_time: number;
  text: string;
  duration: number;
}

export interface Job {
  job_id: string;
  filename: string;
  status: JobStatus;
  progress: number; // 0.0 ~ 1.0
  processed_chunks: number;
  total_chunks: number;
  created_at: number;
  finished_at: number | null;
  error: string | null;
  // status === 'completed' 일 때만 포함
  text?: string;
  chunks?: Chunk[];
  stats?: Record<string, unknown>;
  file_duration?: number;
}

export interface SystemStatus {
  model_loaded: boolean;
  model_name: string;
  device: string;
  total_vram_gb: number;
  memory: {
    cpu_percent?: number;
    cpu_available_gb?: number;
    gpu_allocated_gb?: number;
    gpu_reserved_gb?: number;
    gpu_free_gb?: number;
    gpu_utilization?: number;
  };
  performance: Record<string, unknown>;
  active_jobs: number;
}
