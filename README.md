# ConcreteLab Compute Repo

This is the **public** GitHub repository that runs training, SHAP, and optimization jobs for ConcreteLab via GitHub Actions.

Everything sensitive (Supabase service-role key, dataset content) is injected at runtime — nothing sensitive is committed here.

## One-time setup

1. Create a **new empty public repo** on GitHub (e.g. `yourname/concretelab-compute`).
2. Copy the entire contents of this `compute-repo/` folder to the root of that repo and push.
3. In the new repo, go to **Settings → Secrets and variables → Actions → New repository secret** and add:

   | Name | Value |
   |---|---|
   | `SUPABASE_URL` | Your Lovable Cloud Supabase URL (same as `VITE_SUPABASE_URL` in the app `.env`) |
   | `SUPABASE_SERVICE_ROLE_KEY` | Service-role key for your Lovable Cloud project |

4. In the ConcreteLab app, save two Lovable Cloud secrets so the backend can fire dispatches:
   - `GITHUB_ACTIONS_PAT` — a GitHub Personal Access Token (classic) with `repo` scope
   - `GITHUB_ACTIONS_REPO` — the repo name as `owner/repo` (e.g. `yourname/concretelab-compute`)

5. Done. Clicking **Train Model / Run SHAP / Run Optimization** in the app now dispatches a `repository_dispatch` event that this repo handles automatically.

## How it works

```
App button click
  → TanStack server fn (dispatchRun)
    → GitHub REST API repository_dispatch (event_type: train-request | shap-request | optimize-request)
      → this repo's workflow runs
        → scripts/{train,shap,optimize}.py
          → fetches dataset from Supabase Storage
          → runs the ML job
          → writes results + status back to Supabase
```

Realtime subscriptions in the app pick up the status change and render results with **no page refresh**.

## Local dev

```bash
pip install -r requirements.txt
export SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=...
python -m scripts.train '{"run_id": "..."}'
```
