import sys
from pathlib import Path
from rich import print

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.g1_motion_dataset import G1MotionDataset
from utils.general import load_config


def main():

    yml = load_config("./config/inspect.yaml")
    dataset_yml = yml["dataset"]

    root_dir      = yml["root_dir"]

    window_size     = dataset_yml["window_size"]
    stride          = dataset_yml["stride"]
    min_seq_len     = dataset_yml["min_seq_len"]
    normalize       = dataset_yml["normalize"]
    train           = dataset_yml["train"]
    train_split     = dataset_yml["train_split"]
    preload         = dataset_yml["preload"]     

    dataset = G1MotionDataset(
        root_dir=root_dir,
        window_size=window_size,
        stride=stride,
        min_seq_len=min_seq_len,
        normalize=normalize,
        train=train,
        train_split=train_split,
        preload=preload
    )

    print(f"[green]Dataset size (num windows):[/green] {len(dataset)}")
    sample = dataset[0]
    print("[green]Sample keys:[/green]", list(sample.keys()))
    print("[green]State shape:[/green]", sample["state"].shape)
    print("[green]Cond shape:[/green]", sample["cond"].shape)
    print("[green]FPS:[/green]", sample["fps"])
    print("[green]Seq name:[/green]", sample.get("seq_name"))


if __name__ == "__main__":
    main()
