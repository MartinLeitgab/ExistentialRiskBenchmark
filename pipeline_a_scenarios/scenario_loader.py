"""Load scenario JSON lists with required-field validation."""

import json
import os
from typing import Any, Dict, List

_VALID_PAIR_TYPES_DESCRIPTION = (
    "inst_vs_pro, inst_vs_anti, pro_vs_anti, "
    "inst_vs_inst, pro_vs_pro, anti_vs_anti"
)


def load_scenarios(path: str) -> List[Dict[str, Any]]:
    """Load scenarios from JSON; fail if any row is missing ``pair_type``."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    scenarios = _load_json_with_bom_handling(path)
    missing = [s["id"] for s in scenarios if not s.get("pair_type")]
    if missing:
        raise ValueError(
            f"scenarios missing required pair_type: {missing}. "
            f"Valid values: {_VALID_PAIR_TYPES_DESCRIPTION}"
        )
    return scenarios


def _load_json_with_bom_handling(file_path: str) -> Any:
    """Parse JSON with UTF-8 BOM tolerance (matches legacy create_prototypes behaviour)."""
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeError):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            with open(file_path, "rb") as f:
                raw_data = f.read()
                if raw_data.startswith(b"\xef\xbb\xbf"):
                    raw_data = raw_data[3:]
                return json.loads(raw_data.decode("utf-8"))
