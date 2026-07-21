"""Run arbitrary Python from the ConcreteLab chat, streaming stdout/stderr back
to the app via the `chat_code_runs` table.

Payload: { run_id: uuid, project_id: uuid, code: str }
"""
from __future__ import annotations

import io
import os
import sys
import traceback
import contextlib
from datetime import datetime, timezone

from .common import get_client, get_payload, update_status


TABLE = "chat_code_runs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    payload = get_payload()
    run_id = payload.get("run_id")
    if not run_id:
        print("ERROR: payload missing run_id", file=sys.stderr)
        sys.exit(1)

    sb = get_client()
    # Fetch the row (the app already inserted `code`)
    row = sb.table(TABLE).select("code").eq("id", run_id).single().execute().data
    code = (row or {}).get("code") or payload.get("code") or ""
    if not code.strip():
        update_status(sb, TABLE, run_id, {
            "status": "failed",
            "error_message": "Empty code payload",
            "completed_at": _now_iso(),
        })
        return

    update_status(sb, TABLE, run_id, {"status": "running"})

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    status = "completed"
    err_msg: str | None = None

    # Provide a real cwd + a helper `sb` client for user code
    exec_globals: dict = {
        "__name__": "__main__",
        "sb": sb,
        "supabase": sb,
    }

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(compile(code, "<chat_code>", "exec"), exec_globals)
    except SystemExit as exc:
        if exc.code not in (0, None):
            status = "failed"
            err_msg = f"SystemExit: {exc.code}"
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        tb = traceback.format_exc()
        err_msg = f"{type(exc).__name__}: {exc}\n\n{tb[-2000:]}"
        stderr_buf.write("\n" + tb)

    stdout = stdout_buf.getvalue()[-40000:]
    stderr = stderr_buf.getvalue()[-20000:]

    update_status(sb, TABLE, run_id, {
        "status": status,
        "stdout": stdout,
        "stderr": stderr,
        "error_message": err_msg,
        "completed_at": _now_iso(),
    })

    print(f"[chat_code_runs] {run_id} {status}")
    if status == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
