"""Pre-commit safety review via OpenAI gpt-4o."""
import argparse
import subprocess
import sys

SYSTEM_PROMPT = (
    "You are a Senior MLOps Engineer doing a pre-commit safety review. "
    "Flag CRITICAL if you find: unsafe model promotion logic, missing rollback "
    "conditions, or undefined SLOs. Respond ONLY with PASS or CRITICAL: <reason>."
)

def get_staged_diff() -> str:
    result = subprocess.run(
        ["git", "diff", "--cached"],
        capture_output=True, text=True
    )
    return result.stdout

def review_with_openai(diff: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": diff},
        ],
    )
    return response.choices[0].message.content.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    diff = get_staged_diff()

    if args.dry_run:
        print(f"Dry run: {len(diff)} chars staged. Skipping OpenAI review.")
        sys.exit(0)

    result = review_with_openai(diff)
    print(result)
    sys.exit(1 if result.startswith("CRITICAL") else 0)

if __name__ == "__main__":
    main()
