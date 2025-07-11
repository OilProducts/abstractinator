import argparse
import json
import importlib.util
import logging
from configs.base_config import ExpConfig

from lm_eval import evaluator, utils
import components.hae_lm  # registers HierarchicalAELM with lm_eval

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
        help=("Path to configuration file for hier_ae models. If omitted, the "
              "checkpoint's stored config is used"),
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

    if args.model == "hier_ae":
        model_args = utils.simple_parse_args_string(args.model_args or "")
        checkpoint = model_args.get("checkpoint")
        if checkpoint is None:
            raise ValueError("model_args must include 'checkpoint' for hier_ae model")

        if args.config:
            spec = importlib.util.spec_from_file_location("config_module", args.config)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            exp_config = config_module.exp_config
            device = args.device or getattr(config_module, "DEVICE", None)
        else:
            ckpt = torch.load(checkpoint, map_location="cpu")
            if "exp_config" not in ckpt:
                raise ValueError("Checkpoint missing exp_config; provide --config")
            cfg = ckpt["exp_config"]
            if isinstance(cfg, dict):
                exp_config = ExpConfig(**cfg)
            else:
                exp_config = cfg
            device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

        lm = components.hae_lm.HierarchicalAELM(
            checkpoint=checkpoint,
            config=exp_config,
            device=device,
        )
        model_param = lm
        model_args_param = None
    else:
        model_param = args.model
        model_args_param = args.model_args

    results = evaluator.simple_evaluate(
        model=model_param,
        model_args=model_args_param,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device if args.model != "hier_ae" else None,
        limit=args.limit,
    )

    logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
