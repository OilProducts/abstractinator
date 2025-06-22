import argparse
import json

from lm_eval import evaluator
import components.hae_lm  # registers HierarchicalAELM with lm_eval

DEFAULT_TASKS = [
    "lambada_openai",
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "winogrande",
]


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation tasks")
    parser.add_argument("--model", default="hf", help="Model type (e.g. hf)")
    parser.add_argument(
        "--model_args",
        default=None,
        help="Model arguments string passed to the evaluation harness",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Which evaluation tasks to run",
    )
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--device", default=None, help="Device string for PyTorch")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation examples (mainly for quick tests)",
    )
    args = parser.parse_args()

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
