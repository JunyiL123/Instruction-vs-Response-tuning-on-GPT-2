import argparse
from pathlib import Path

from .common import ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description="GPT-2 response-ranking and tuning study")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("prepare")
    subs.add_parser("audit")
    train_parser = subs.add_parser("train")
    train_parser.add_argument("--mode", required=True, choices=["response", "instruction"])
    train_parser.add_argument("--smoke-steps", type=int, default=0)
    evaluation = subs.add_parser("evaluate")
    evaluation.add_argument("--condition", required=True, choices=["base", "response", "instruction"])
    evaluation.add_argument("--allow-unreviewed", action="store_true")
    subs.add_parser("blind")
    reporting = subs.add_parser("report")
    reporting.add_argument("--final", action="store_true")
    review_parser = subs.add_parser("review")
    review_parser.add_argument("--kind", choices=["negatives", "generation"], required=True)
    review_parser.add_argument("--port", type=int, default=8765)
    grade_parser = subs.add_parser("apply-generation-grades")
    grade_parser.add_argument("--grades", type=Path, required=True)
    run_parser = subs.add_parser("run")
    run_parser.add_argument("--allow-unreviewed", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = args.root.resolve()
    if args.command == "prepare":
        from .data import prepare
        prepare(cfg, root)
    elif args.command == "audit":
        from .data import verify, validated_pairs
        manifest = verify(root)
        pairs, reviewed = validated_pairs(root, True)
        print(f"Split hashes and disjoint duplicate groups verified: {manifest['split_counts']}")
        print(f"{len(pairs)} frozen ranking examples; negative adjudication complete: {reviewed}")
    elif args.command == "train":
        from .train import train
        train(cfg, root, args.mode, args.device, args.smoke_steps)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(cfg, root, args.condition, args.device, args.allow_unreviewed)
    elif args.command == "blind":
        from .evaluate import blind
        blind(root, cfg["seed"])
    elif args.command == "report":
        from .report import report
        report(cfg, root, args.final)
    elif args.command == "review":
        from .review import serve
        serve(root, args.kind, args.port)
    elif args.command == "apply-generation-grades":
        from .grade import apply_generation_grades
        apply_generation_grades(root, args.grades)
    elif args.command == "run":
        from .run import run
        run(args.config, root, args.device, args.allow_unreviewed)


if __name__ == "__main__":
    main()
