from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

def find_nan_files(data_dir: Path, pattern: str = "*.npy") -> list[Path]:
    """
    Scan `data_dir` for files matching `pattern` and return those containing any NaN values.

    Args:
        data_dir: Directory to search for .npy files.
        pattern: Glob pattern (default '*.npy').

    Returns:
        List of Paths to files containing NaNs.
    """
    nan_files: list[Path] = []
    for file_path in tqdm(list(data_dir.glob(pattern)), desc="Scanning for NaNs"):
        try:
            arr = np.load(str(file_path))
            if np.isnan(arr).any():
                nan_files.append(file_path)
        except Exception as e:
            print(f"⚠️  Could not load {file_path.name}: {e}")
    return nan_files


def delete_files(file_list: list[Path], verbose: bool = True) -> int:
    """
    Delete each file in `file_list`.

    Args:
        file_list: List of Path objects to delete.
        verbose: If True, print deletion progress.

    Returns:
        Number of files successfully deleted.
    """
    deleted = 0
    for file_path in file_list:
        try:
            file_path.unlink()
            deleted += 1
            if verbose:
                print(f"🗑️  Deleted {file_path.name}")
        except Exception as e:
            print(f"⚠️  Failed to delete {file_path.name}: {e}")
    return deleted


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean .npy ECG arrays by removing files containing NaNs.")
    ap.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing .npy files to scan",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*.npy",
        help="Glob pattern to select files (default: '*.npy')",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="List files that would be deleted without deleting them",
    )
    args = ap.parse_args()

    # Validate directory
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find NaN files
    nan_files = find_nan_files(data_dir, pattern=args.pattern)
    total = len(list(data_dir.glob(args.pattern)))
    print(f"Found {len(nan_files)} files with NaNs out of {total} in {data_dir}")

    if args.dry_run:
        for path in nan_files:
            print(path.name)
        return

    # Delete them
    deleted = delete_files(nan_files)
    remaining = total - deleted
    print(f"Deleted {deleted} files. Remaining .npy files: {remaining}")

if __name__ == "__main__":
    main()
