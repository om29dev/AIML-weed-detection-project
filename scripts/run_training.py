"""
MH-Weed16 YOLO11 Resumable Final Training Runner
Accelerated for Intel Arc XPU hardware.
"""

import os
import sys
import pathlib
import argparse
import pandas as pd
import yaml
import torch
from ultralytics import YOLO

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description="Resumable Final Model Training on Intel Arc XPU")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--name", type=str, default="mhweed16_yolo11n_final", help="Experiment name")
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

    run_dir = ROOT_DIR / "runs" / "detect" / args.name
    weights_dir = run_dir / "weights"
    last_weights = weights_dir / "last.pt"
    best_weights = weights_dir / "best.pt"
    results_csv = run_dir / "results.csv"
    best_hyp_file = ROOT_DIR / "best_hyperparameters.yaml"

    print("=" * 65)
    print(f"MH-WEED16 RESUMABLE FINAL TRAINING: {args.name}")
    print(f"- Target Architecture: YOLO11 Nano (yolo11n.pt)")
    print(f"- Total Epochs:        {args.epochs}")
    print(f"- Batch Size:          {args.batch}")
    print(f"- Image Resolution:    {args.imgsz}")
    print(f"- Compute Device:      {device}")
    print(f"- Workers:             {args.workers}")
    print(f"- Run Output Dir:      {run_dir}")
    print("=" * 65)

    # Check if already completed all epochs
    if best_weights.exists() and results_csv.exists():
        try:
            df_res = pd.read_csv(results_csv)
            df_res.columns = df_res.columns.str.strip()
            completed_epochs = len(df_res)
            if completed_epochs >= args.epochs:
                print(f"[Done] Training already completed all {completed_epochs} epochs!")
                print(f"Best model weights: {best_weights}")
                return
        except Exception:
            pass

    # Check for resumable checkpoint
    if last_weights.exists():
        print(f"[Resume] Found existing checkpoint: {last_weights}")
        print("Resuming training from last saved epoch...")
        model = YOLO(str(last_weights))
        train_results = model.train(resume=True)
    else:
        print("[Start] Initiating fresh training run...")
        model = YOLO("yolo11n.pt")

        # Load optimal hyperparameters
        evolved_args = {}
        if best_hyp_file.exists():
            with open(best_hyp_file, 'r', encoding='utf-8') as f:
                evolved_args = yaml.safe_load(f) or {}
            print(f"Loaded evolved hyperparameters from: {best_hyp_file}")
        else:
            print("No evolved config found; applying default agricultural augmentations.")
            evolved_args = {
                'degrees': 90.0,
                'flipud': 0.5,
                'fliplr': 0.5,
                'mosaic': 1.0
            }

        train_results = model.train(
            data="mhweed16.yaml",
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            workers=args.workers,
            name=args.name,
            exist_ok=True,
            amp=False,  # Explicitly disabled for Intel Arc XPU stability
            patience=args.patience,
            **evolved_args
        )

    print("\nTraining completed successfully!")
    print(f"Artifacts saved to: {run_dir}")

if __name__ == "__main__":
    main()
