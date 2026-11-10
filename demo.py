#!/usr/bin/env python3
"""Run the final SteelNet demo or verify all 7,320 public observations."""

import argparse
from pathlib import Path
import torch
from steelnet.io import ROOT, load_yaml
from steelnet.evaluate import CONFIG, METHODS, SCENARIOS, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", default="run", choices=("run", "validate-data")
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--paths", type=int, default=20)
    parser.add_argument(
        "--quick", action="store_true", help="One path and 30 periods per scenario"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "validate-data":
        from steelnet.validate_data import validate

        validate()
        return
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; use --device cpu or install a CUDA-enabled PyTorch build."
        )
    methods = args.methods.split(",")
    scenarios = args.scenarios.split(",")
    if not 1 <= args.paths <= 20:
        parser.error("--paths must be 1..20")
    if not set(methods) <= set(METHODS) or len(set(methods)) != len(methods):
        parser.error("Invalid --methods")
    if not set(scenarios) <= set(SCENARIOS) or len(set(scenarios)) != len(scenarios):
        parser.error("Invalid --scenarios")
    summary = evaluate(
        args.output,
        load_yaml(args.config),
        methods,
        n_paths=1 if args.quick else args.paths,
        horizon=30 if args.quick else None,
        device=device,
        scenarios=scenarios,
    )
    print(
        summary[
            [
                "scenario",
                "method",
                "discounted_total_cost",
                "total_cost",
                "BW_LE",
                "BW_SME",
            ]
        ].to_string(index=False)
    )
    print(f'Report: {args.output.resolve()/"report.html"}')


if __name__ == "__main__":
    main()
