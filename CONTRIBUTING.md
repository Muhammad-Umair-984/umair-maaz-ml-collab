# Contributing

## Team responsibilities

- Umair owns the model pipeline, configurations, and experiments. He also handles environment and release preparation.
- Maaz owns data checks, DVC data updates, pre-commit hooks, and CI.
- Both members write code, open pull requests, review each other's work, and run experiments.

## Branches and pull requests

- `main` contains approved releases.
- `staging` contains a release candidate awaiting independent reproduction.
- `dev` combines completed work.
- Start `feat/<name>` and `data/<name>` branches from `dev`; open pull requests back into `dev`.
- Start personal `exp/<member>-<idea>` branches from `dev`. Never merge an experiment branch directly. Apply or cherry-pick a winning change onto a new `feat/` branch.
- Start an urgent `fix/<name>` branch from `main`. Merge it through a reviewed pull request into `main`, then bring the fix back into `dev`.
- Promote releases through reviewed pull requests from `dev` to `staging`, then `staging` to `main`.
- After the permitted initial import in Phase 2, do not push directly to `dev`, `staging`, or `main`.
- Each pull request needs the other member's review and at least one approval. Required CI checks must pass once CI is installed.

We use **squash merges for pull requests into `dev`**. This keeps one clear commit for each completed change on the integration branch. Delete short-lived feature and data branches after merging.

## Commit messages

Use Conventional Commits, such as `feat: add training baseline`, `data: remove duplicate rows`, `fix: correct metric calculation`, and `exp: try a different model depth`.

Each member commits under their own Git identity. Do not commit on behalf of a teammate.

## Data and model files

Keep datasets and trained models out of Git. Track them with DVC and use a shared DVC remote. Run `dvc push` **before** `git push` whenever data or model files change, so teammates can retrieve the content referenced by the Git commit. Never commit credentials or `.env` files.

Reviewers should run the changed code. For a data change, retrieve the DVC data in a fresh clone. For a pipeline change, reproduce its metrics before approving.
