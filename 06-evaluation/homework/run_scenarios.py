import csv
import json
import time

from recipe_agent import agent
import dotenv
dotenv.load_dotenv()

MODEL_PRICES = {
    "google-gla:gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "google-gla:gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}



def calculate_cost(usage, model="google-gla:gemini-2.5-flash"):
    prices = MODEL_PRICES[model]
    input_cost = (usage.input_tokens / 1_000_000) * prices["input"]
    output_cost = (usage.output_tokens / 1_000_000) * prices["output"]
    return round(input_cost + output_cost, 6)


def run_all():
    # filename = 'scenarios.csv'
    filename = 'synthetic_scenarios.csv'  
    with open(filename) as f:

        scenarios = list(csv.DictReader(f))

    results = []

    for i, scenario in enumerate(scenarios):
        question = scenario['question']
        print(f"[{i+1}/{len(scenarios)}] {question}")

        start = time.time()
        result = agent.run_sync(question)
        elapsed = time.time() - start

        usage = result.usage()
        cost = calculate_cost(usage)

        results.append({
            'question': question,
            'category': scenario['category'],
            'type': scenario['type'],
            'output': result.output,
            'execution_time': round(elapsed, 2),
            'tokens': {
                'input_tokens': usage.input_tokens,
                'output_tokens': usage.output_tokens,
                'total_tokens': usage.total_tokens,
            },
            'cost': cost,
        })

        print(f"  Done in {elapsed:.1f}s (${cost})")

    with open('synthetic_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    total_cost = sum(r['cost'] for r in results)
    print(f"\nSaved {len(results)} results to synthetic_results.json")
    print(f"Total cost: ${total_cost:.4f}")

if __name__ == "__main__":
    run_all()
