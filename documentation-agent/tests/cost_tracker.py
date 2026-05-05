import json
import tempfile
from pathlib import Path
from pydantic_ai import RunUsage
from collections import defaultdict

MODEL_PRICES = {
    "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-4o": {"input": 2.50, "output": 10.00},
    "google-gla:gemini-2.5-flash": {"input": 0.3, "output": 2.50},
    "google-gla:gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "google-gla:gemini-2.5-flash-lite-preview-09-2025": {"input": 0.1, "output": 0.40},
}

COST_FILE = Path(tempfile.gettempdir()) / "pytest_cost_tracker.jsonl"

def calculate_cost(model_name, input_tokens, output_tokens):
    prices = MODEL_PRICES[model_name.lower()]
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost


def reset_cost_file():
    COST_FILE.unlink(missing_ok=True)


def capture_usage(model, result):
    usage = result.usage()
    entry = {
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    with open(COST_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def display_total_usage():
    print()

    if not COST_FILE.exists():
        print("Total cost: $0.000000")
        return

    totals = {}
    for line in COST_FILE.read_text().splitlines():
        entry = json.loads(line)
        model = entry["model"]
        if model not in totals:
            totals[model] = {"input_tokens": 0, "output_tokens": 0}
        totals[model]["input_tokens"] += entry["input_tokens"]
        totals[model]["output_tokens"] += entry["output_tokens"]

    total_cost = 0
    for model, tokens in totals.items():
        cost = calculate_cost(model, tokens["input_tokens"], tokens["output_tokens"])
        print(f"{model}: ${cost:.6f}")
        total_cost += cost

    print(f"Total cost: ${total_cost:.6f}")