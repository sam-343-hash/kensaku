# Agent Guide

Give this file to your AI agent (paste it into its system prompt or let it read it from the repo). It explains how the agent should use this project to train models on Kaggle.

## Your job

You write training scripts and run them on Kaggle's free GPUs using `kaggle_agent.py`. You never run heavy training locally.

## Workflow

1. Write a complete, self-contained Python training script.
2. Save it as `jobs/<name>.py` (use a short, lowercase, hyphen-free name like `resnet_v1`).
3. Optionally save `jobs/<name>.json` with settings:
   ```json
   {"gpu": true, "internet": true, "datasets": [], "competitions": []}
   ```
4. Start the agent if it is not running: `python kaggle_agent.py watch`
   Or run one job directly: `python kaggle_agent.py run jobs/<name>.py --gpu`
5. Wait for `results/<name>/` to appear. If the job failed, the script will be in `jobs/failed/`.
6. Read `results/<name>/metrics.json` and the log file, then decide what to try next.

## Rules for every training script

- Save ALL outputs to `/kaggle/working/` (models, logs, predictions, metrics). Anything else is lost.
- Use this path fallback so the script also works locally:
  ```python
  import os
  OUT = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
  ```
- Read competition/dataset files from `/kaggle/input/<dataset-name>/`. They are read-only.
- Always write `metrics.json` at the end, for example:
  ```python
  import json
  json.dump({"val_loss": val_loss, "val_acc": val_acc, "epochs": EPOCHS}, open(f"{OUT}/metrics.json", "w"))
  ```
- Print progress with `print()` so logs show what happened.
- Set random seeds so results are reproducible.
- Do not hardcode secrets or API keys in scripts.
- Install extra packages at the top only if needed (internet must be enabled):
  ```python
  import subprocess
  subprocess.run(["pip", "install", "-q", "package-name"])
  ```

## Limits to respect

- Free GPU quota is about 30 hours per week. Do not waste it.
- One session lasts at most about 9 to 12 hours. Keep each job well under that.
- Disk space is limited to about 20 GB in `/kaggle/working`.

## Good habits

- First run a tiny test (1 epoch, small data sample) to catch bugs cheaply.
- Change one thing at a time between runs so you know what helped.
- Name jobs clearly (`bert_lr3e5`, `bert_lr1e5`) and keep a log of what each run tried and scored.
- Save a checkpoint at the end of training.
- If a job fails, read the log in `results/<name>/`, fix the bug, and resubmit under a new version name.

## Do not

- Do not delete or overwrite other jobs' results.
- Do not commit or print `kaggle.json`.
- Do not submit more than one GPU job at a time unless told to.
- Do not retry a failing job more than 3 times without changing something.

## Example prompt to start the agent

> Read AGENT_GUIDE.md. Train a model for the Kaggle competition `titanic`. Write the script to `jobs/titanic_v1.py` with a JSON file enabling the competition data, wait for the results, then improve it and try a second version. Report the metrics of each run.
