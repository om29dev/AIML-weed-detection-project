"""
MH-Weed16 YOLO11 Resumable Hyperparameter Evolution Runner
Accelerated for Intel Arc XPU hardware.
"""

import os
import sys
import json
import time
import shutil
import pathlib
import argparse
import pandas as pd
import yaml
import torch
from ultralytics import YOLO

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

def sync_ndjson_to_csv(ndjson_path: pathlib.Path, csv_path: pathlib.Path, best_hyp_path: pathlib.Path, space: dict):
    """Parse Ultralytics tune_results.ndjson and export to CSV and best_hyperparameters.yaml."""
    if not ndjson_path.exists():
        return None
    records = []
    with open(ndjson_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                gen = entry.get('iteration', len(records) + 1)
                fit = entry.get('fitness', 0.0)
                hyps = entry.get('hyperparameters', {})
                metrics = entry.get('metrics', {})
                records.append({
                    'generation': gen,
                    'fitness': fit,
                    'mAP50': metrics.get('metrics/mAP50(B)', metrics.get('mAP50', 0.0)),
                    'mAP50-95': metrics.get('metrics/mAP50-95(B)', metrics.get('mAP50-95', 0.0)),
                    'precision': metrics.get('metrics/precision(B)', metrics.get('precision', 0.0)),
                    'recall': metrics.get('metrics/recall(B)', metrics.get('recall', 0.0)),
                    **hyps
                })
            except Exception as e:
                print(f"[Warning] Failed parsing line in {ndjson_path}: {e}")
    if not records:
        return None
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Synced {len(df)} trials to {csv_path}")

    # Update best_hyperparameters.yaml
    best_row = df.sort_values(by='fitness', ascending=False).iloc[0]
    best_params = {k: float(best_row[k]) for k in space.keys() if k in best_row}
    with open(best_hyp_path, 'w', encoding='utf-8') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print(f"Saved optimal hyperparameters to {best_hyp_path} (Fitness: {best_row['fitness']:.4f})")
    return df

def count_completed_iterations(ndjson_path: pathlib.Path) -> int:
    """Return count of valid completed iterations in ndjson."""
    if not ndjson_path.exists():
        return 0
    count = 0
    with open(ndjson_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description="Resumable Hyperparameter Evolution on Intel Arc XPU")
    parser.add_argument("--iterations", type=int, default=20, help="Number of evolution generations (default: 20)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per evolution trial (default: 30)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--device", type=str, default="", help="Device override ('xpu', 'cpu', etc.)")
    args = parser.parse_args()

    # Hardware detection
    if args.device:
        device = args.device
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = 'xpu'
    elif torch.cuda.is_available():
        device = '0'
    else:
        device = 'cpu'

    print("=" * 65)
    print("MH-WEED16 RESUMABLE HYPERPARAMETER EVOLUTION")
    print(f"- Target Model:       YOLO11 Nano (yolo11n.pt)")
    print(f"- Total Iterations:   {args.iterations}")
    print(f"- Epochs per Trial:   {args.epochs}")
    print(f"- Batch Size:         {args.batch}")
    print(f"- Image Size:         {args.imgsz}")
    print(f"- Compute Device:     {device}")
    print(f"- Workers:            {args.workers}")
    print("=" * 65)

    space = {
        'lr0': (1e-4, 1e-2),
        'lrf': (0.01, 0.2),
        'momentum': (0.85, 0.98),
        'weight_decay': (1e-5, 1e-3),
        'warmup_epochs': (1.0, 4.0),
        'box': (5.0, 10.0),
        'cls': (0.3, 1.5),
        'dfl': (1.0, 2.5),
        'hsv_h': (0.01, 0.05),
        'hsv_s': (0.4, 0.8),
        'hsv_v': (0.3, 0.6),
        'degrees': (0.0, 90.0),
        'flipud': (0.2, 0.7),
        'fliplr': (0.3, 0.7),
        'mosaic': (0.5, 1.0),
        'mixup': (0.0, 0.2)
    }

    tune_dir = ROOT_DIR / "runs" / "detect" / "tune"
    ndjson_file = tune_dir / "tune_results.ndjson"
    csv_file = ROOT_DIR / "tune_results.csv"
    best_hyp_file = ROOT_DIR / "best_hyperparameters.yaml"

    completed = count_completed_iterations(ndjson_file)
    print(f"Checkpoint check: Found {completed}/{args.iterations} completed iterations in {ndjson_file}")

    if completed >= args.iterations:
        print(f"[Done] All {completed} evolution iterations have already finished!")
        sync_ndjson_to_csv(ndjson_file, csv_file, best_hyp_file, space)
        return

    resume_flag = (completed > 0)
    if resume_flag:
        print(f"[Resume] Resuming genetic evolution from iteration {completed + 1}/{args.iterations}...")
    else:
        print(f"[Start] Beginning brand-new evolution run for {args.iterations} iterations...")

    # Instantiate YOLO model
    model = YOLO("yolo11n.pt")

    # Run native Ultralytics tuning (omitting project ensures default runs/detect)
    model.tune(
        data="mhweed16.yaml",
        epochs=args.epochs,
        iterations=args.iterations,
        optimizer="AdamW",
        plots=True,
        save=True,
        val=True,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        space=space,
        name="tune",
        resume=resume_flag,
        exist_ok=True,
        amp=False  # Disabled for Intel Arc XPU stability
    )

    # Final sync
    sync_ndjson_to_csv(ndjson_file, csv_file, best_hyp_file, space)
    print("\nHyperparameter evolution complete!")

if __name__ == "__main__":
    main()
