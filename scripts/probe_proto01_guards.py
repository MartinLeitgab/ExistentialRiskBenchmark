"""Guard-isolation probe for proto_01 (issue #46 follow-up).

Runs four scenario variants that isolate the two structural defects identified in
`data/findings.md` §16 — temporal dominance and mandate collapse — against the three
frontier target models on the locked production variant.

Deliberately does NOT use `prompt_validation.run_validation_study`: that path
constructs `UnifiedLLMClient` with the default `enable_cache=True`, which serves every
repeat draw from the first response's cache entry (`findings.md` §8a). This probe
disables the cache so the repeats are genuine independent samples.

Writes one JSON line per response as it arrives so a mid-flight kill loses nothing.

Usage:
    python -u scripts/probe_proto01_guards.py [--reps 10] [--out outputs/proto_01_guard_probe]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline_a_scenarios.utils.cost_tracker import CostTracker  # noqa: E402
from pipeline_a_scenarios.utils.llm_client import UnifiedLLMClient  # noqa: E402
from pipeline_a_scenarios.utils.prompt_generator import generate_prompt  # noqa: E402

SCENARIOS_PATH = REPO_ROOT / "data" / "scenarios" / "proto_01_guard_probe.json"

# Locked production variant: fp-abs-ftc-auto-t10-reg-0-0-0
# (first_person / no exfiltration / free-text-with-choice / autonomous system prompt /
#  temperature 1.0 / regular action order / no ethical framing / no value conflict /
#  no goal specification) — see docs/forward_plan.md "Prompt dimension decisions".
VARIANT_ID = "fp-abs-ftc-auto-t10-reg-0-0-0"
DIMENSIONS = {
    "framing": "first_person",
    "exfiltration": "absent",
    "response_format": "free_text_with_choice",
    "system_prompt": "autonomous",
    "temperature": 1.0,
    "action_order": "regular",
    "ethical_framing": "absent",
    "value_conflict": "absent",
    "goal_specification": "absent",
}

MODELS = [
    ("anthropic", "claude-opus-5"),
    ("openai", "gpt-5.6-sol"),
    ("google", "gemini-3.1-pro-preview"),
]

TEMPERATURE = 1.0
MAX_TOKENS = 3000  # per PR #62: Opus reasons before answering; 500 truncates the tag

ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE | re.DOTALL)

_write_lock = threading.Lock()


def parse_choice(text: str) -> str | None:
    """Return the final <answer> letter, or None when the tag never arrives."""
    matches = ANSWER_RE.findall(text or "")
    return matches[-1].upper() if matches else None


def run_model(
    provider: str, model: str, scenarios: list[dict], reps: int, out_path: Path
) -> list[dict]:
    """Run every (scenario × rep) for one model sequentially on its own client."""
    client = UnifiedLLMClient(
        provider=provider,
        model=model,
        enable_cache=False,
        cost_tracker=CostTracker(user_id="proto01_probe"),
    )
    rows: list[dict] = []

    for scenario in scenarios:
        built = generate_prompt(
            context=scenario["context"],
            action_a=scenario["action_a"],
            action_b=scenario["action_b"],
            dimensions=DIMENSIONS,
        )
        for rep in range(reps):
            try:
                response = client.generate(
                    prompt=built["user_prompt"],
                    system_prompt=built["system_prompt"],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                # UnifiedLLMClient.generate() returns the body under "content".
                text = response.get("content") or ""
                row = {
                    "scenario_id": scenario["id"],
                    "guards": scenario["guards"],
                    "model": model,
                    "provider": provider,
                    "rep": rep,
                    "variant_id": VARIANT_ID,
                    "temperature": TEMPERATURE,
                    "parsed_choice": parse_choice(text),
                    "choice_preference": None,
                    "response_text": text,
                    "usage": response.get("usage", {}),
                    "error": None,
                }
                choice = row["parsed_choice"]
                if choice == "A":
                    row["choice_preference"] = scenario["action_a_preference"]
                elif choice == "B":
                    row["choice_preference"] = scenario["action_b_preference"]
            except (
                Exception
            ) as exc:  # noqa: BLE001 — record and continue; one dead cell must not kill the run
                row = {
                    "scenario_id": scenario["id"],
                    "guards": scenario["guards"],
                    "model": model,
                    "provider": provider,
                    "rep": rep,
                    "variant_id": VARIANT_ID,
                    "temperature": TEMPERATURE,
                    "parsed_choice": None,
                    "choice_preference": None,
                    "response_text": "",
                    "usage": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }

            rows.append(row)
            with _write_lock:
                with out_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                status = row["error"] or row["parsed_choice"] or "NO_TAG"
                print(
                    f"  {model:26s} {scenario['id']:22s} rep{rep:02d} -> {status}",
                    flush=True,
                )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reps", type=int, default=10, help="independent draws per scenario × model"
    )
    parser.add_argument("--out", type=str, default="outputs/proto_01_guard_probe")
    parser.add_argument("--scenarios", type=str, default=str(SCENARIOS_PATH))
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "responses.jsonl"
    out_path.write_text("", encoding="utf-8")

    total = len(scenarios) * len(MODELS) * args.reps
    print(
        f"{len(scenarios)} scenarios × {len(MODELS)} models × {args.reps} reps = {total} calls\n"
        f"variant={VARIANT_ID} temperature={TEMPERATURE} max_tokens={MAX_TOKENS} cache=DISABLED\n",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = [
            pool.submit(run_model, provider, model, scenarios, args.reps, out_path)
            for provider, model in MODELS
        ]
        for future in futures:
            future.result()

    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    errors = [r for r in rows if r["error"]]
    no_tag = [r for r in rows if not r["error"] and r["parsed_choice"] is None]
    print(f"\nwrote {len(rows)} rows to {out_path}")
    print(f"errors: {len(errors)}  |  missing <answer> tag: {len(no_tag)}")
    if errors:
        print("first error:", errors[0]["error"])


if __name__ == "__main__":
    main()
