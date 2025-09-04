import argparse
import json
import logging

from lm_eval import evaluator

logger = logging.getLogger(__name__)

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
        "--config",
        type=str,
        default=None,
        help=("Optional config path (unused for HF models)."),
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
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Logging level (debug, info, warning, error, critical)",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # This entrypoint runs external harness models (e.g., HF). For models from
    # this repo, write custom evaluation using AbstractinatorPyramid directly.
    model_param = args.model
    model_args_param = args.model_args

    results = evaluator.simple_evaluate(
        model=model_param,
        model_args=model_args_param,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )

    logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
