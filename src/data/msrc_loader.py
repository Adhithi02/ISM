"""
MSR Cambridge block-trace loader.

Expected MSRC CSV format (SNIA IOTTA trace 388):
    Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime
    - Timestamp : Windows filetime (100 ns ticks since 1601-01-01)
    - Hostname  : string
    - DiskNumber: int
    - Type      : "Read" or "Write"
    - Offset    : bytes (byte offset on disk)
    - Size      : bytes (request size)
    - ResponseTime : Windows filetime ticks (duration)

Units and Conversions:
    - Timestamp: Windows filetime ticks -> Unix seconds: (ticks - 116444736000000000) / 10,000,000
    - Offset: byte offset -> extent ID: Offset // (1024 * 1024) (1 MiB extent)
    - Size: bytes
    - ResponseTime: ticks -> milliseconds: ticks / 10,000.0
    - Type: "Read" -> is_read = 1, "Write" -> is_read = 0

If files are missing in data/msrc_raw, the loader explicitly notifies the user and stops.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# MSRC column names as in the SNIA trace files
MSRC_COLUMNS = [
    "Timestamp", "Hostname", "DiskNumber", "Type",
    "Offset", "Size", "ResponseTime",
]

_WIN_EPOCH_OFFSET = 116444736000000000
_TICKS_PER_SECOND = 10_000_000
_TICKS_PER_MS = 10_000


def _win_ticks_to_seconds(ticks: np.ndarray) -> np.ndarray:
    """Convert Windows filetime ticks to Unix seconds (float64)."""
    return (ticks.astype(np.float64) - _WIN_EPOCH_OFFSET) / _TICKS_PER_SECOND


def _response_ticks_to_ms(ticks: np.ndarray) -> np.ndarray:
    """Convert response-time ticks to milliseconds."""
    return ticks.astype(np.float64) / _TICKS_PER_MS


def find_volume_file(raw_dir: Path, vol: str) -> Optional[Path]:
    """Find CSV or CSV.GZ file for a volume in raw_dir."""
    candidates = [
        raw_dir / f"{vol}.csv",
        raw_dir / f"{vol}.csv.gz",
        raw_dir / f"{vol}.csv.GZ",
        raw_dir / f"{vol}",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def count_file_rows(file_path: Path) -> int:
    """Fast row counter for plain or gzipped CSV."""
    count = 0
    if file_path.suffix.lower() == ".gz":
        with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            for _ in f:
                count += 1
    else:
        with open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            for _ in f:
                count += 1
    # Subtract header row if present
    return max(0, count - 1) if count > 0 else 0


def inspect_raw_volume(file_path: Path, vol_name: str, extent_size_bytes: int = 1_048_576) -> Dict[str, Any]:
    """
    Inspect raw volume file: print header, units, first 5 rows, and total row count.
    """
    print(f"\n{'=' * 60}")
    print(f"MSRC Raw Volume Inspection: {vol_name} ({file_path.name})")
    print(f"{'=' * 60}")
    print(f"File Path: {file_path.resolve()}")
    print(f"File Size: {file_path.stat().st_size / (1024*1024):.2f} MiB")

    # Read first 5 rows
    df_raw = pd.read_csv(file_path, nrows=5, names=MSRC_COLUMNS, header=0)
    print("\n[Header & Column Names]")
    print(list(df_raw.columns))

    print("\n[Data Units & Schema]")
    print("  Timestamp   : Windows filetime ticks (100 ns since 1601-01-01)")
    print("  Hostname    : Machine hostname string")
    print("  DiskNumber  : Target disk index (integer)")
    print("  Type        : Operation type ('Read' or 'Write')")
    print("  Offset      : Byte offset on disk (divide by 1048576 for 1 MiB extent ID)")
    print("  Size        : Request size in bytes")
    print("  ResponseTime: Duration in Windows 100 ns ticks (divide by 10000 for ms)")

    print("\n[First 5 Raw Rows (as in file)]")
    print(df_raw.to_string(index=False))

    # Converted preview
    df_conv = pd.DataFrame({
        "timestamp_s": _win_ticks_to_seconds(df_raw["Timestamp"].values),
        "extent": (df_raw["Offset"].values // extent_size_bytes).astype(np.int32),
        "is_read": (df_raw["Type"].str.strip().str.lower() == "read").astype(np.int8),
        "size_bytes": df_raw["Size"].values,
        "response_ms": _response_ticks_to_ms(df_raw["ResponseTime"].values),
        "DiskNumber": df_raw["DiskNumber"].values,
    })
    print("\n[First 5 Converted Rows (transformed for ISM)]")
    print(df_conv.to_string(index=False))

    print("\nCounting total rows in trace file...")
    total_rows = count_file_rows(file_path)
    print(f"Total Rows: {total_rows:,}")

    return {
        "volume": vol_name,
        "file": str(file_path),
        "total_rows": total_rows,
        "header": list(df_raw.columns),
        "sample_converted": df_conv.to_dict(orient="records"),
        "extent_size_bytes": extent_size_bytes,
    }


def load_volume_from_csv(
    csv_path: Path,
    parquet_path: Path,
    chunksize: int = 500_000,
    extent_size_bytes: int = 1_048_576,
) -> Optional[pd.DataFrame]:
    """Read MSRC CSV in chunks, convert, and cache as Parquet."""
    if parquet_path.exists():
        print(f"  Cache hit: {parquet_path}")
        return pd.read_parquet(parquet_path)

    if not csv_path.exists():
        return None

    print(f"  Parsing {csv_path.name} in chunks (chunksize={chunksize:,})...")
    frames: List[pd.DataFrame] = []
    for i, chunk in enumerate(pd.read_csv(
        csv_path,
        names=MSRC_COLUMNS,
        header=0,
        chunksize=chunksize,
        dtype={
            "Timestamp": np.int64,
            "Hostname": str,
            "DiskNumber": np.int32,
            "Type": str,
            "Offset": np.int64,
            "Size": np.int64,
            "ResponseTime": np.int64,
        },
    )):
        chunk["timestamp_s"] = _win_ticks_to_seconds(chunk["Timestamp"].values)
        chunk["response_ms"] = _response_ticks_to_ms(chunk["ResponseTime"].values)
        chunk["extent"] = (chunk["Offset"] // extent_size_bytes).astype(np.int32)
        chunk["is_read"] = (chunk["Type"].str.strip().str.lower() == "read").astype(np.int8)
        chunk = chunk[["timestamp_s", "extent", "is_read", "Size", "response_ms", "DiskNumber"]]
        chunk = chunk.rename(columns={"Size": "size_bytes"})
        frames.append(chunk)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp_s").reset_index(drop=True)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    print(f"  Cached {len(df):,} rows -> {parquet_path}")
    return df


def load_msrc_volumes(
    config: Dict[str, Any],
    stop_if_missing: bool = True,
) -> Tuple[Dict[str, Optional[pd.DataFrame]], List[str]]:
    """
    Check and load requested MSRC volumes.
    If files are missing and stop_if_missing is True:
        Returns loaded dict and list of missing volume names.
    """
    msrc_cfg = config["data"]["msrc"]
    volumes: List[str] = msrc_cfg["volumes"]
    raw_dir = Path(msrc_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(msrc_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Optional[pd.DataFrame]] = {}
    missing: List[str] = []

    for vol in volumes:
        parquet_path = cache_dir / f"{vol}.parquet"
        source = find_volume_file(raw_dir, vol)

        if parquet_path.exists():
            print(f"  [Cache] Found cached {vol}.parquet")
            results[vol] = pd.read_parquet(parquet_path)
            continue

        if source is not None:
            # Inspect first
            extent_size_bytes = int(config["data"]["extent_size_bytes"])
            inspect_raw_volume(source, vol, extent_size_bytes=extent_size_bytes)
            df = load_volume_from_csv(source, parquet_path, extent_size_bytes=extent_size_bytes)
            results[vol] = df
        else:
            missing.append(vol)
            results[vol] = None

    if missing and stop_if_missing:
        print(f"\n[MSRC NOTICE] Missing raw trace files for volumes: {missing}")
        print(f"Expected location: {raw_dir.resolve()}")
        print(f"Expected files: {[f'{v}.csv.gz or {v}.csv' for v in missing]}")

    return results, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect MSRC traces")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("MSRC Cambridge Trace Inspector & Loader")
    print("=" * 60)

    results, missing = load_msrc_volumes(config, stop_if_missing=True)
    loaded = [v for v, d in results.items() if d is not None]

    if missing:
        print(f"\nSTOPPED: {len(missing)} MSRC volume file(s) missing: {missing}")
        print("Please place the trace files in data/msrc_raw/ to proceed with MSRC.")
    else:
        print(f"\nAll requested MSRC volumes loaded successfully: {loaded}")


if __name__ == "__main__":
    main()
