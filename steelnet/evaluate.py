"""Evaluate the final three-method comparison without training."""

from pathlib import Path
import time
import hashlib
import numpy as np
import pandas as pd
from algorithm.paper_heuristic import PaperBaseStockHeuristic
from demand.poisson import generate as poisson
from demand.stochastic_jump import generate as jump
from demand.real_20 import load_real20
from .io import ROOT, load_yaml, dump_json
from .rollout import make_env, LearnedPolicyAdapter, HeuristicAdapter, run_episode
from .costs import aggregate

CONFIG = ROOT / "configs/demo.yaml"
METHODS = ("happo", "mappo", "heuristic")
SCENARIOS = ("stationary", "stochastic_jump", "real20")


def scenario_banks(config, n_paths, horizon, scenarios=SCENARIOS):
    a, b = config["evaluation"]["synthetic_seeds"]
    banks = {}
    if "stationary" in scenarios:
        banks["stationary"] = poisson(n_paths=20, horizon=horizon, seed=a)[:n_paths]
    if "stochastic_jump" in scenarios:
        banks["stochastic_jump"] = jump(n_paths=20, horizon=horizon, seed=b)[0][
            :n_paths
        ]
    if "real20" in scenarios:
        banks["real20"] = load_real20()[0][:n_paths, :horizon]
    return banks


def provider_for(method, config, device):
    p = config["policy"]
    if method == "heuristic":
        return HeuristicAdapter(
            PaperBaseStockHeuristic(
                platform_order_max=p["order_max"],
                supplier_order_max=p["supplier_order_max"],
                **config["heuristic"],
            ),
            name=method,
        )
    return LearnedPolicyAdapter.from_checkpoint(
        ROOT / f"checkpoints/{method}.pt",
        env=make_env(**config["environment"], alpha=p["alpha"][method]),
        algorithm=method,
        order_max=p["order_max"],
        supplier_order_max=p["supplier_order_max"],
        hidden_size=p["hidden_size"],
        device=device,
    )


def evaluate(
    output,
    config=None,
    methods=METHODS,
    n_paths=20,
    horizon=None,
    device="cpu",
    scenarios=SCENARIOS,
):
    config = config or load_yaml(CONFIG)
    horizon = horizon or config["environment"]["horizon"]
    if not 1 <= n_paths <= 20 or not 1 <= horizon <= 366:
        raise ValueError("paths 1..20; horizon 1..366")
    if not methods or not set(methods) <= set(METHODS):
        raise ValueError("Unknown evaluation method")
    if not scenarios or not set(scenarios) <= set(SCENARIOS):
        raise ValueError("Unknown scenario")
    start = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_steps = []
    episodes = []
    banks = scenario_banks(config, n_paths, horizon, scenarios)
    for method in methods:
        provider = provider_for(method, config, device)
        for scenario, bank in banks.items():
            print(
                f"[evaluate] {method} / {scenario}: {len(bank)} x {horizon}", flush=True
            )
            for index, demand in enumerate(bank):
                steps = run_episode(provider, demand, config, method, scenario, index)
                metrics = aggregate(
                    steps,
                    gamma=config["evaluation"]["gamma"],
                    holding_cost=config["environment"]["holding_cost"],
                    backlog_cost=config["environment"]["backlog_cost"],
                )
                metrics.update(method=method, scenario=scenario, episode=index)
                all_steps.extend(steps)
                episodes.append(metrics)
    frame = pd.DataFrame(episodes)
    summary = (
        frame.groupby(["scenario", "method"], sort=False)
        .mean(numeric_only=True)
        .drop(columns="episode")
        .reset_index()
    )
    counts = (
        frame.groupby(["scenario", "method"]).size().rename("n_episodes").reset_index()
    )
    summary = summary.merge(counts, on=["scenario", "method"])
    pd.DataFrame(all_steps).to_csv(output / "steps.csv", index=False)
    frame.to_csv(output / "episodes.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    hashes = {
        method: hashlib.sha256(
            (ROOT / f"checkpoints/{method}.pt").read_bytes()
        ).hexdigest()
        for method in methods
        if method != "heuristic"
    }
    dump_json(
        output / "run.json",
        {
            "protocol": config,
            "split": "test",
            "device": device,
            "seeds": [config["evaluation"]["policy_seed"]],
            "methods": list(methods),
            "scenarios": list(banks),
            "paths_per_scenario": n_paths,
            "horizon": horizon,
            "gamma": config["evaluation"]["gamma"],
            "primary_metric": "discounted_total_cost",
            "checkpoint_sha256": hashes,
            "comparison": "final HAPPO versus fixed MAPPO and causal heuristic",
            "training_performed": False,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        },
    )
    from .report import make_report

    make_report(output)
    return summary
