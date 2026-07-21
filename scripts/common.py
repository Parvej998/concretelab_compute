"""Shared helpers for GitHub Actions compute jobs.

Every job reads its config from the `PAYLOAD` env var (JSON dispatched by the app),
uses the Supabase service-role key to fetch the dataset from Storage, and writes
results back to the same run row the app already reads from.
"""
from __future__ import annotations

import io
import json
import os
import sys
import traceback
from typing import Any

import pandas as pd
from supabase import Client, create_client


def get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def get_payload() -> dict[str, Any]:
    raw = os.environ.get("PAYLOAD", "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def load_dataset(sb: Client, dataset_version_id: str) -> pd.DataFrame:
    """Fetch a dataset_version's CSV/XLSX from the `datasets` storage bucket."""
    row = (
        sb.table("dataset_versions")
        .select("storage_path")
        .eq("id", dataset_version_id)
        .single()
        .execute()
        .data
    )
    if not row or not row.get("storage_path"):
        raise RuntimeError(f"No storage_path for dataset_version {dataset_version_id}")
    path = row["storage_path"]
    blob = sb.storage.from_("datasets").download(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(blob))
    return pd.read_csv(io.BytesIO(blob))


def update_status(sb: Client, table: str, run_id: str, patch: dict[str, Any]) -> None:
    sb.table(table).update(patch).eq("id", run_id).execute()


def run_with_status(table: str, fn) -> None:
    """Wrap the main routine: mark running → run → mark succeeded/completed or failed."""
    payload = get_payload()
    run_id = payload.get("run_id") or payload.get("job_id")
    if not run_id:
        print("ERROR: payload missing run_id", file=sys.stderr)
        sys.exit(1)

    sb = get_client()
    update_status(sb, table, run_id, {"status": "running"})
    try:
        result_patch = fn(sb, payload, run_id) or {}
        patch = {"status": "completed", **result_patch}
        update_status(sb, table, run_id, patch)
        print(f"[{table}] run {run_id} completed")
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        update_status(
            sb,
            table,
            run_id,
            {"status": "failed", "error_message": f"{exc}\n\n{tb[-1500:]}"},
        )
        sys.exit(1)
