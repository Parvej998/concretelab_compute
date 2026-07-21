"""Run arbitrary Python from the ConcreteLab chat, streaming stdout/stderr back
to the app via the `chat_code_runs` table in near-real time.

Payload: { run_id: uuid, project_id: uuid, code: str }

Streaming: a background thread executes user code while the main thread flushes
buffered stdout/stderr to Supabase every ~1.5s so the chat card shows live
processing steps (epochs, prints, tqdm-style progress) instead of only the
final dump.
"""
from __future__ import annotations

import io
import sys
import threading
import time
import traceback
import contextlib
from datetime import datetime, timezone

from .common import get_client, get_payload, update_status


TABLE = "chat_code_runs"
FLUSH_INTERVAL_SEC = 1.5
STDOUT_TAIL = 40000
STDERR_TAIL = 20000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _TeeStream(io.TextIOBase):
    """Write to an in-memory buffer AND the real fd so GitHub logs still show it."""

    def __init__(self, buf: io.StringIO, mirror) -> None:
        self._buf = buf
        self._mirror = mirror
        self._lock = threading.Lock()

    def write(self, s: str) -> int:  # type: ignore[override]
        with self._lock:
            self._buf.write(s)
        try:
            self._mirror.write(s)
            self._mirror.flush()
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        try:
            self._mirror.flush()
        except Exception:
            pass


def main() -> None:
    payload = get_payload()
    run_id = payload.get("run_id")
    if not run_id:
        print("ERROR: payload missing run_id", file=sys.stderr)
        sys.exit(1)

    sb = get_client()
    row = sb.table(TABLE).select("code").eq("id", run_id).single().execute().data
    code = (row or {}).get("code") or payload.get("code") or ""
    if not code.strip():
        update_status(sb, TABLE, run_id, {
            "status": "failed",
            "error_message": "Empty code payload",
            "completed_at": _now_iso(),
        })
        return

    update_status(sb, TABLE, run_id, {
        "status": "running",
        "stdout": "▶ Runner started — executing code...\n",
    })

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    tee_out = _TeeStream(stdout_buf, sys.__stdout__)
    tee_err = _TeeStream(stderr_buf, sys.__stderr__)

    result: dict = {"status": "completed", "err_msg": None}

    def _run() -> None:
        exec_globals: dict = {"__name__": "__main__", "sb": sb, "supabase": sb}
        try:
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                exec(compile(code, "<chat_code>", "exec"), exec_globals)
        except SystemExit as exc:
            if exc.code not in (0, None):
                result["status"] = "failed"
                result["err_msg"] = f"SystemExit: {exc.code}"
        except Exception as exc:  # noqa: BLE001
            result["status"] = "failed"
            tb = traceback.format_exc()
            result["err_msg"] = f"{type(exc).__name__}: {exc}\n\n{tb[-2000:]}"
            tee_err.write("\n" + tb)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    last_out_len = 0
    last_err_len = 0
    while True:
        worker.join(timeout=FLUSH_INTERVAL_SEC)
        cur_out = stdout_buf.getvalue()
        cur_err = stderr_buf.getvalue()
        if len(cur_out) != last_out_len or len(cur_err) != last_err_len:
            last_out_len = len(cur_out)
            last_err_len = len(cur_err)
            try:
                update_status(sb, TABLE, run_id, {
                    "stdout": cur_out[-STDOUT_TAIL:],
                    "stderr": cur_err[-STDERR_TAIL:],
                })
            except Exception as exc:  # noqa: BLE001
                print(f"[stream] flush failed: {exc}", file=sys.__stderr__)
        if not worker.is_alive():
            break

    stdout = stdout_buf.getvalue()[-STDOUT_TAIL:]
    stderr = stderr_buf.getvalue()[-STDERR_TAIL:]

    update_status(sb, TABLE, run_id, {
        "status": result["status"],
        "stdout": stdout,
        "stderr": stderr,
        "error_message": result["err_msg"],
        "completed_at": _now_iso(),
    })

    print(f"[chat_code_runs] {run_id} {result['status']}")
    if result["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
