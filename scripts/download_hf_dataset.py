"""
Download retargeted motion dataset from HuggingFace.

Downloads the dataset and organizes it for preprocessing.

Usage:
    python scripts/download_hf_dataset.py --repo_id <huggingface_repo_id> [--output_dir ./data/hf_dataset]

Example:
    python scripts/download_hf_dataset.py --repo_id username/retargeted-motion-dataset
"""

import os
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download retargeted motion dataset from HuggingFace")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace dataset repo ID (e.g. 'username/dataset-name')")
    parser.add_argument("--output_dir", type=str, default="./data/hf_dataset",
                        help="Directory to save the downloaded dataset")
    parser.add_argument("--robot", type=str, default="unitree_g1",
                        help="Robot name to filter (default: unitree_g1)")
    parser.add_argument("--revision", type=str, default="main",
                        help="Git revision (branch/tag/commit) to download")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Install it with:")
        print("  pip install huggingface_hub")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {args.repo_id}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Robot filter: {args.robot}")
    print(f"Revision: {args.revision}")
    print()

    # Download the full dataset (or filter by robot if possible)
    # HuggingFace snapshot_download gets the full repo
    try:
        local_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            revision=args.revision,
            # Only download data for the target robot + scripts
            allow_patterns=[
                f"data/{args.robot}/**",
                "scripts/**",
                "README.md",
                "*.py",
            ],
        )
        print(f"\nDownload complete: {local_path}")
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nIf this is a private/gated dataset, you may need to:")
        print("  1. Log in: huggingface-cli login")
        print("  2. Accept the dataset terms on HuggingFace")
        sys.exit(1)

    # Verify the structure
    data_dir = output_dir / "data" / args.robot
    if data_dir.exists():
        datasets_found = [d.name for d in data_dir.iterdir() if d.is_dir()]
        print(f"\nDatasets found for {args.robot}:")
        for ds in sorted(datasets_found):
            # Count motions
            ds_dir = data_dir / ds
            motions = [m.name for m in ds_dir.iterdir() if m.is_dir()]
            print(f"  {ds}: {len(motions)} motions")
    else:
        # Maybe the data is at the root level (no data/ prefix)
        # Try alternative structures
        print(f"\nWARNING: Expected structure data/{args.robot}/ not found.")
        print("The dataset may use a different directory layout.")
        print(f"Contents of {output_dir}:")
        for item in sorted(output_dir.iterdir()):
            print(f"  {item.name}")

    print(f"\nDataset ready at: {output_dir.resolve()}")
    print(f"\nNext step: Run preprocessing:")
    print(f"  python scripts/preprocess_hf_data.py \\")
    print(f"    --input_dir {output_dir}/data/{args.robot} \\")
    print(f"    --output_dir ./data/hf_preprocessed")


if __name__ == "__main__":
    main()
