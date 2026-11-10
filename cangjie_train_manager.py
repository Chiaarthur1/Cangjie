"""
Project Cangjie — менеджер фоновых задач обучения [v10.0].

Запускает обучение/проверку как фоновый subprocess, пишет лог, даёт статус.
Поддерживаемые job-типы (CPU-проверяемые без GPU):
  - sanity_loss : быстрая проверка сходимости композитного лосса D-FINE;
  - embed_train : CPU-обучение проекции Chamfer+HOG (PCA) из папки глифов;
  - detector    : полное обучение RT-DETR (требует GPU — честно упадёт).

Без заглушек: статус берётся из реального процесса и лог-файла.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, Optional

_LOG_DIR = "./runs/jobs"


class CangjieTrainManager:
    def __init__(self, log_dir: str = _LOG_DIR):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def start(self, job_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        job_id = uuid.uuid4().hex[:12]
        log_path = os.path.join(self.log_dir, f"{job_id}.log")

        cmd = self._build_command(job_type, params)
        if cmd is None:
            return {"ok": False, "error": f"Неизвестный тип задачи: {job_type}"}

        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"# {job_type} | {time.strftime('%Y-%m-%d %H:%M:%S')}\n# {' '.join(cmd)}\n")
            proc = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )

        self._jobs[job_id] = {
            "id": job_id, "type": job_type, "status": "running",
            "pid": proc.pid, "log_path": log_path, "started_at": time.time(),
            "params": params, "proc": proc,
        }
        return {"ok": True, "job_id": job_id, "type": job_type, "pid": proc.pid}

    @staticmethod
    def _build_command(job_type: str, params: Dict[str, Any]) -> Optional[list]:
        py = sys.executable
        if job_type == "sanity_loss":
            epochs = int(params.get("epochs", 8))
            return [py, "cangjie_rtdetr_train.py", "--sanity-test", "--profile", "laptop", "--epochs", str(epochs)]
        if job_type == "embed_train":
            data = params.get("data", "./data/glyphs")
            out = params.get("out", "runs/embeddings/projection.npz")
            dim = int(params.get("dim", 128))
            return [py, "cangjie_metric_embed.py", "train", "--backbone", "chamfer_hog",
                    "--data", data, "--out", out, "--dim", str(dim)]
        if job_type == "detector":
            profile = params.get("profile", "auto")
            epochs = int(params.get("epochs", 30))
            data = params.get("data", "./cangjie_yolo_dataset/data.yaml")
            return [py, "cangjie_rtdetr_train.py", "--profile", profile,
                    "--run-ultralytics", "--epochs", str(epochs), "--data", data]
        return None

    def _refresh(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs[job_id]
        proc: subprocess.Popen = job["proc"]
        rc = proc.poll()
        if rc is not None:
            job["status"] = "finished" if rc == 0 else "failed"
            job["exit_code"] = rc
        else:
            job["status"] = "running"
        return job

    def status(self, job_id: str) -> Dict[str, Any]:
        if job_id not in self._jobs:
            return {"ok": False, "error": f"Задача {job_id} не найдена"}
        job = self._refresh(job_id)
        tail = self._tail(job["log_path"], 40)
        return {
            "ok": True, "id": job_id, "type": job["type"], "status": job["status"],
            "pid": job["pid"], "exit_code": job.get("exit_code"),
            "elapsed_sec": round(time.time() - job["started_at"], 1),
            "log_tail": tail,
        }

    def list_jobs(self) -> Dict[str, Any]:
        items = []
        for jid, job in list(self._jobs.items()):
            self._refresh(jid)
            items.append({
                "id": jid, "type": job["type"], "status": job["status"],
                "elapsed_sec": round(time.time() - job["started_at"], 1),
            })
        return {"jobs": items}

    @staticmethod
    def _tail(path: str, n: int) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except Exception as e:
            return f"(лог недоступен: {e})"


_shared: Optional[CangjieTrainManager] = None


def get_train_manager() -> CangjieTrainManager:
    global _shared
    if _shared is None:
        _shared = CangjieTrainManager()
    return _shared


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Project Cangjie — фоновые задачи обучения")
    ap.add_argument("--start", choices=["sanity_loss", "embed_train", "detector"], default=None)
    ap.add_argument("--status", default=None, help="job_id")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    mgr = get_train_manager()
    if args.start:
        print(json.dumps(mgr.start(args.start), ensure_ascii=False, indent=2))
    elif args.status:
        print(json.dumps(mgr.status(args.status), ensure_ascii=False, indent=2))
    elif args.list:
        print(json.dumps(mgr.list_jobs(), ensure_ascii=False, indent=2))
