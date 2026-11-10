"""Offline report for the explicit manuscript demo evaluation protocol."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ("happo", "mappo", "heuristic")
LABELS = {"happo": "HAPPO", "mappo": "Fixed MAPPO", "heuristic": "Heuristic (C.3)"}
SHORT_LABELS = {
    "happo": "HAPPO",
    "mappo": "Fixed MAPPO",
    "heuristic": "Heuristic\n(C.3)",
}
COLORS = {"happo": "#147d92", "mappo": "#e3913c", "heuristic": "#6b8e52"}
SCENARIOS = {
    "stationary": "Poisson",
    "stochastic_jump": "Stochastic jump",
    "real20": "Real20",
}
NODE_COLORS = {"platform": "#147d92", "le": "#e3913c", "sme": "#8b94a6"}
NODE_LABELS = {"platform": "Platform", "le": "LE", "sme": "SME"}


def _require(frame, names, filename):
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing columns: {', '.join(missing)}")


def _validate(summary, episodes, steps, gamma):
    keys = ["scenario", "method"]
    costs = ["discounted_total_cost", "total_cost"]
    components = [
        f"discounted_{kind}_cost_{node}"
        for node in NODE_COLORS
        for kind in ("holding", "backlog")
    ]
    _require(
        summary,
        keys + costs + components + ["BW_LE", "BW_SME", "n_episodes"],
        "summary.csv",
    )
    _require(
        episodes,
        keys + costs + ["episode", "n_steps", "discount_gamma"],
        "episodes.csv",
    )
    _require(steps, keys + ["episode", "t", "system_cost"], "steps.csv")
    if summary.empty or episodes.empty or steps.empty:
        raise ValueError("The report requires nonempty evaluation tables")
    if summary.duplicated(keys).any():
        raise ValueError("summary.csv must contain one row per scenario/method")
    if not np.isfinite(summary[costs + components].to_numpy(dtype=float)).all():
        raise ValueError("Cost objectives and node components must be finite")
    if not np.allclose(episodes.discount_gamma, gamma, rtol=0, atol=1e-12):
        raise ValueError("run.json gamma differs from episode discount_gamma")
    if not np.allclose(
        summary[components].sum(axis=1),
        summary.discounted_total_cost,
        rtol=1e-9,
        atol=1e-6,
    ):
        raise ValueError(
            "Discounted node cost components do not sum to the primary objective"
        )
    ep_groups = episodes.groupby(keys, sort=False)
    raw_counts = steps.groupby(keys + ["episode"], sort=False).size()
    for _, row in summary.iterrows():
        group = ep_groups.get_group((row.scenario, row.method))
        if len(group) != int(row.n_episodes):
            raise ValueError("Summary episode counts do not match episodes.csv")
        for cost in costs:
            if not np.isclose(
                float(row[cost]), float(group[cost].mean()), rtol=1e-9, atol=1e-6
            ):
                raise ValueError(
                    f"Summary {cost} does not match the mean of episodes.csv"
                )
    for _, row in episodes.iterrows():
        if int(raw_counts.loc[(row.scenario, row.method, row.episode)]) != int(
            row.n_steps
        ):
            raise ValueError("Period-record counts do not match episode lengths")


def _comparison_figure(
    summary, scenarios, methods, metrics, titles, path, cost_scale=False
):
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(13.8, 4.8), layout="constrained", squeeze=False
    )
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(scenarios))
    for ax, metric, title in zip(axes[0], metrics, titles):
        for index, method in enumerate(methods):
            series = (
                summary[summary.method == method]
                .set_index("scenario")
                .reindex(scenarios)[metric]
            )
            values = series.to_numpy(dtype=float)
            if cost_scale:
                values = values / 1e6
            positions = x + (index - (len(methods) - 1) / 2) * width
            ax.bar(
                positions,
                values,
                width=width,
                color=COLORS.get(method, "#64748b"),
                label=LABELS.get(method, method),
            )
            for position, value in zip(positions, values):
                if not np.isfinite(value):
                    ax.text(
                        position,
                        0.02,
                        "N/A",
                        transform=ax.get_xaxis_transform(),
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_xticks(x, [SCENARIOS.get(scenario, scenario) for scenario in scenarios])
        ax.set_xlim(-0.6, len(scenarios) - 0.4)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(
            (
                "Cost (million model units)"
                if cost_scale
                else "Replenishment variance / allocated-order variance"
            ),
            fontsize=9,
        )
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=min(4, len(methods)),
        frameon=False,
        fontsize=9,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _component_figure(summary, scenarios, methods, gamma, path):
    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(5 * len(scenarios), 5.2),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    x = np.arange(len(methods))
    for ax, scenario in zip(axes[0], scenarios):
        frame = (
            summary[summary.scenario == scenario].set_index("method").reindex(methods)
        )
        bottom = np.zeros(len(methods))
        for node, color in NODE_COLORS.items():
            for kind, hatch in (("holding", None), ("backlog", "///")):
                values = (
                    frame[f"discounted_{kind}_cost_{node}"].to_numpy(dtype=float) / 1e6
                )
                ax.bar(
                    x,
                    values,
                    bottom=bottom,
                    width=0.68,
                    color=color,
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.4,
                    label=f"{NODE_LABELS[node]} {kind}",
                )
                bottom += values
        ax.set_xticks(
            x, [SHORT_LABELS.get(method, method) for method in methods], fontsize=9
        )
        ax.set_title(SCENARIOS.get(scenario, scenario), fontsize=11)
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
    # Set the shared scale after every panel is drawn; freezing it while the
    # first panel is built would clip larger bars in subsequent scenarios.
    maximum = float(summary.discounted_total_cost.max()) / 1e6
    axes[0, 0].set_ylim(0, maximum * 1.08 if maximum > 0 else 1)
    axes[0, 0].set_ylabel(f"Discounted cost, gamma={gamma:g} (million model units)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncol=3, frameon=False, fontsize=9
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_report(output):
    """Render existing paper_protocol tables; never runs policies or training."""
    output = Path(output)
    summary = pd.read_csv(output / "summary.csv")
    episodes = pd.read_csv(output / "episodes.csv")
    steps = pd.read_csv(
        output / "steps.csv",
        usecols=["scenario", "method", "episode", "t", "system_cost"],
    )
    meta = json.loads((output / "run.json").read_text(encoding="utf-8"))
    gamma = float(meta["gamma"])
    _validate(summary, episodes, steps, gamma)
    methods = [method for method in METHOD_ORDER if method in set(summary.method)]
    methods += [method for method in summary.method.unique() if method not in methods]
    scenarios = list(summary.scenario.unique())
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#334155",
            "figure.facecolor": "white",
        }
    ):
        _comparison_figure(
            summary,
            scenarios,
            methods,
            ["discounted_total_cost", "total_cost"],
            [
                f"Primary: discounted total cost (gamma={gamma:g})",
                "Auxiliary: undiscounted total cost",
            ],
            figures / "paper_costs.png",
            cost_scale=True,
        )
        _component_figure(
            summary, scenarios, methods, gamma, figures / "paper_components.png"
        )
        _comparison_figure(
            summary,
            scenarios,
            methods,
            ["BW_LE", "BW_SME"],
            ["LE branch bullwhip", "SME branch bullwhip"],
            figures / "paper_bullwhip.png",
        )

    def image(name):
        return "data:image/png;base64," + base64.b64encode(
            (figures / name).read_bytes()
        ).decode("ascii")

    def escaped(value):
        return html.escape(str(value))

    columns = [
        "scenario",
        "method",
        "n_episodes",
        "discounted_total_cost",
        "total_cost",
        "BW_LE",
        "BW_SME",
    ]
    optional = [
        name
        for name in ("aggregate_sme_order_share", "cumulative_service_level")
        if name in summary
    ]
    table = summary[columns + optional].copy()
    table["scenario"] = table.scenario.map(lambda value: SCENARIOS.get(value, value))
    table["method"] = table.method.map(lambda value: LABELS.get(value, value))
    table = table.rename(
        columns={
            "scenario": "Scenario",
            "method": "Method",
            "n_episodes": "Episodes",
            "discounted_total_cost": f"Primary cost (gamma={gamma:g})",
            "total_cost": "Undiscounted cost",
            "BW_LE": "LE bullwhip",
            "BW_SME": "SME bullwhip",
            "aggregate_sme_order_share": "SME allocated-order share",
            "cumulative_service_level": "Cumulative service",
        }
    )
    table_html = table.to_html(
        index=False, border=0, na_rep="N/A", float_format=lambda value: f"{value:,.4f}"
    )
    protocol = meta.get("protocol", {})
    env = protocol.get("environment", {})
    policy = protocol.get("policy", {})
    alpha = policy.get("alpha", {})
    split = escaped(meta.get("split", "unspecified"))
    seeds = escaped(", ".join(map(str, meta.get("seeds", []))))
    device = escaped(meta.get("device", "unspecified"))
    horizon = escaped(
        meta.get("horizon", ", ".join(map(str, sorted(episodes.n_steps.unique()))))
    )
    paths = escaped(meta.get("paths_per_scenario", "see episode counts"))
    assumptions = "".join(
        f"<li>{escaped(value)}</li>" for value in protocol.get("assumptions", [])
    )
    assumption_html = (
        f"<details><summary>Recorded protocol assumptions</summary><ul>{assumptions}</ul></details>"
        if assumptions
        else ""
    )
    environment_html = (
        f"Holding-cost coefficients: {escaped(env.get('holding_cost', 'see run.json'))}; "
        f"backlog-cost coefficients: {escaped(env.get('backlog_cost', 'see run.json'))}, in platform, LE, SME order. "
        f"Lead times: {escaped(env.get('lead_time', 'see run.json'))}. "
        f"Platform/supplier order caps: {escaped(policy.get('order_max', 'see run.json'))} / "
        f"{escaped(policy.get('supplier_order_max', 'see run.json'))}."
    )
    alpha_html = f"HAPPO alpha={escaped(alpha.get('happo', 'see protocol'))}; MAPPO alpha={escaped(alpha.get('mappo', 'see protocol'))}."
    refinement_html = "<p>The final HAPPO policy was continued with a remaining-time centralized critic. MAPPO is a fixed pretrained baseline. Actor observations and evaluation costs follow the shared configuration. Model checksums and provenance are in checkpoints/manifest.json and docs/MODEL_CARD.md.</p>"
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SteelNet demo results</title><style>
body{{margin:0;background:#f4f7fa;color:#17283d;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:42px 24px 65px}}h1{{font-size:38px;line-height:1.2;margin:12px 0 18px}}h2{{margin-top:36px;font-size:25px}}p{{max-width:1040px;color:#536478}}.eyebrow{{color:#147d92;font-weight:700;font-size:12px;letter-spacing:.12em}}.stats{{display:flex;gap:32px;flex-wrap:wrap;padding:20px 0;margin:24px 0;border-block:1px solid #dbe2eb}}.stats b{{display:block;font-size:24px}}.stats span{{font-size:13px;color:#536478}}.primary{{background:#e9f4f6;padding:16px 20px;border-left:4px solid #147d92}}img{{width:100%;height:auto;background:white;border-radius:5px}}.table{{overflow:auto;background:white;padding:12px}}table{{border-collapse:collapse;font-size:12px;width:100%}}th,td{{padding:9px;border-bottom:1px solid #e1e7ef;text-align:right}}th{{white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}details{{margin-top:20px;color:#536478}}summary{{cursor:pointer;font-weight:600}}footer{{margin-top:35px;font-size:13px;color:#65758a}}code{{color:#147d92}}
</style></head><body><main><div class="eyebrow">STEELNET · FINAL DEMO</div>
<h1>Procurement, allocation and system cost</h1>
<p>Each method is evaluated on the same scenario paths, supply-chain dynamics, action caps and cost coefficients. HAPPO and MAPPO jointly select platform orders, SME allocation and supplier replenishment. The third comparator is a causal rolling base-stock heuristic with gap-based allocation.</p>
<div class="stats"><span><b>{paths}</b>Paths per scenario</span><span><b>{horizon}</b>Periods per path</span><span><b>{seeds}</b>Training seed(s)</span><span><b>{device}</b>Evaluation device</span></div>
<p class="primary"><strong>Primary objective:</strong> discounted inventory-holding plus backlog cost, &Sigma;<sub>t=0</sub><sup>H−1</sup> {gamma:g}<sup>t</sup>C<sub>t</sub>. The first period has weight 1. Undiscounted total cost is an auxiliary result. Both use the same recorded period costs.</p>
<h2>Primary and auxiliary costs</h2><img alt="Discounted primary system cost and undiscounted auxiliary system cost by scenario and method" src="{image('paper_costs.png')}">
<h2>Which nodes incur the cost?</h2><p>Each bar sums to the discounted primary cost. Colors identify nodes; hatching distinguishes backlog cost from inventory-holding cost.</p><img alt="Discounted cost decomposed by platform, LE and SME holding and backlog components" src="{image('paper_components.png')}">
<h2>Branch bullwhip</h2><p>For each supplier, branch bullwhip is the variance of its replenishment orders divided by the variance of platform orders allocated to that supplier, as in Eq. (8). A zero denominator gives N/A, not zero.</p><img alt="LE and SME branch bullwhip by scenario and method" src="{image('paper_bullwhip.png')}">
<h2>Computed results</h2><div class="table">{table_html}</div><p>Table costs are in model units; cost figures use millions of those units. Each summary row is the arithmetic mean of its path episodes. Bullwhip means use finite episode ratios. Cumulative service, when shown, measures demand fulfilled by the episode end. No statistical interval is inferred from a single training seed.</p>
<h2>Evaluation settings</h2><p>{environment_html} {alpha_html} Reward weights affect policy training; all methods are scored with the same system-cost objective.</p>{refinement_html}{assumption_html}
<footer>Split: {split} · Recorded evaluation runtime: {escaped(meta.get('elapsed_seconds', 'unspecified'))} s · Source tables: summary.csv, episodes.csv, steps.csv; protocol: run.json. This report renders the saved evaluation and does not rerun policies. All figures are embedded for offline viewing.</footer>
</main></body></html>"""
    report_path = output / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path
