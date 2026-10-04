"""
Training runner for MH-Weed16 YOLO11 detector.
Supports Intel Arc XPU, CUDA, or CPU backends.
"""

import argparse
import pathlib
import pandas as pd
import torch
from ultralytics import YOLO

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 on MH-Weed16")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--workers", type=int, default=2, help="Data loader workers")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--name", type=str, default="mhweed16_yolo11n", help="Run name")
    parser.add_argument("--device", type=str, default="", help="Device override ('xpu', '0', 'cpu')")
    args = parser.parse_args()

    # Detect compute device
    if args.device:
        device = args.device
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = 'xpu'
    elif torch.cuda.is_available():
        device = '0'
    else:
        device = 'cpu'

    run_dir = ROOT_DIR / "runs" / "detect" / args.name
    last_weights = run_dir / "weights" / "last.pt"
    best_weights = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"

    print("=" * 60)
    print(f"Training YOLO11 on MH-Weed16 ({args.name})")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch} | Image Size: {args.imgsz}")
    print(f"Run dir: {run_dir}")
    print("=" * 60)

    # If already finished
    if best_weights.exists() and results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            if len(df) >= args.epochs:
                print(f"Training already completed ({len(df)} epochs). Best model: {best_weights}")
                return
        except Exception:
            pass

    # Resume or start fresh
    if last_weights.exists():
        print(f"Resuming training from {last_weights}...")
        model = YOLO(str(last_weights))
        model.train(resume=True)
    else:
        print("Starting training with yolo11n.pt...")
        model = YOLO("yolo11n.pt")
        model.train(
            data=str(ROOT_DIR / "mhweed16.yaml"),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            workers=args.workers,
            name=args.name,
            exist_ok=True,
            amp=False,  # Set False for Intel Arc XPU stability
            patience=args.patience,
            lr0=0.01,
            lrf=0.01,
            degrees=45.0,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0
        )

    print(f"Training finished. Best weights: {best_weights}")

if __name__ == "__main__":
    main()
