import argparse
import sys
from pathlib import Path
from rich import print

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.g1_motion_dataset import G1MotionDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="../GMR-master/export_smplx_retargeted")
    parser.add_argument("--window_size", type=int, default=120)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    ds = G1MotionDataset(
        root_dir=args.root_dir,
        window_size=args.window_size,
        stride=args.stride,
        normalize=True,
        train=True,
        train_split=0.9,
    )

    print(f"[green]Dataset size (num windows):[/green] {len(ds)}")
    sample = ds[0]
    print("[green]Sample keys:[/green]", list(sample.keys()))
    print("[green]State shape:[/green]", sample["state"].shape)
    print("[green]Cond shape:[/green]", sample["cond"].shape)
    print("[green]FPS:[/green]", sample["fps"])
    print("[green]Seq name:[/green]", sample.get("seq_name"))


if __name__ == "__main__":
    main()
