# Reproducible MLOps Pipeline (DVC • CI • Docker • Airflow • API)

Production-minded ML project structure focused on reproducibility, automation, and deployable artifacts.

## What this repo demonstrates
- **Reproducible pipeline** with DVC (versioned data + pipeline stages)
- **Automated checks** via GitHub Actions (tests / quality gates)
- **Containerized runtime** with Docker for consistent execution
- **Orchestration-ready** workflows (Airflow assets)
- **Inference entrypoint** (API/app scaffold)

## Repository structure
- `.github/workflows` — CI pipelines (GitHub Actions)
- `.dvc`, `dvc.yaml`, `dvc.lock` — DVC pipeline + tracked artifacts
- `data` — dataset tracked via DVC (pulled, not committed directly)
- `src` — training / pipeline code
- `tests` — unit tests for training/pipeline components
- `airflow` — orchestration assets
- `api` — API/service scaffold
- `app.py` — application entrypoint
- `requirements.txt` — Python dependencies

## Quickstart (local)
### 1) Setup
- Create and activate a virtual environment
- Install dependencies:
  - `pip install -r requirements.txt`

### 2) Pull data/artifacts (if DVC remote is configured)
- Install DVC (if needed):
  - `pip install dvc`
- Pull tracked data/artifacts:
  - `dvc pull`

### 3) Reproduce the pipeline
- Run the pipeline:
  - `dvc repro`

### 4) Run tests
- `pytest -q`

## Docker (optional)
- Build:
  - `docker build -t mlops-app .`
- Run:
  - `docker run --rm mlops-app`

## Notes
This repository is intentionally structured to be easy for reviewers to audit: deterministic reruns, explicit pipeline stages, and automation-friendly tooling.
