"""Fresh MSRC inspection using the configured extent size, never old caches."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from src.data.msrc_loader import load_msrc_volumes

ROOT = Path(__file__).parent


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    cfg = copy.deepcopy(cfg)
    cfg["data"]["msrc"]["volumes"] = ["hm_0", "prn_0"]
    cfg["data"]["msrc"]["cache_dir"] = str(ROOT / "results" / "msrc_16m_parquet")
    loaded, missing = load_msrc_volumes(cfg, stop_if_missing=False)
    summary = {"extent_size_bytes": cfg["data"]["extent_size_bytes"], "missing": missing, "volumes": {}}
    for volume, frame in loaded.items():
        if frame is None:
            summary["volumes"][volume] = {"rows": 0, "status": "missing"}
            continue
        summary["volumes"][volume] = {
            "rows": int(len(frame)),
            "extents": int(frame["extent"].nunique()),
            "first_five": frame.head(5).to_dict(orient="records"),
            "columns": list(frame.columns),
            "status": "loaded_fresh",
        }
    (ROOT / "results" / "msrc_fresh.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
