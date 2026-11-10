# Configuration and metrics

`configs/demo.yaml` is the shared evaluation configuration. Parameter order for node vectors is platform, LE, SME.

| Parameter | Default |
| --- | --- |
| Horizon | 366 periods |
| Paths per scenario | 20 |
| Initial inventory and backlog | Zero; empty pipelines |
| Lead times | LE branch 1, SME branch 2, customer 0 |
| Holding cost coefficients | 100, 20, 20 |
| Backlog cost coefficients | 300, 200, 200 |
| Procurement/replenishment caps | 220 |
| Primary cost discount factor | 0.99 |
| Observation scale | 1,000 |
| HAPPO / MAPPO reward mixing | 0.2 / 0.7 |
| Poisson / stochastic-jump evaluation seeds | 3026 / 3027 |

Each supplier branch uses its stated lead time for both manufacturer-to-supplier and supplier-to-platform transport. All agents choose actions from the decision-time state before current-period arrivals and shipments are applied. Unfulfilled demand remains in backlog.

The period system cost is the sum of each node's end-of-period inventory multiplied by its holding coefficient and backlog multiplied by its backlog coefficient. Purchase payments and random shipping losses are disabled in this comparison. No additional terminal penalty is added.

- `discounted_total_cost`: sum of `gamma**t * system_cost`, with the first period indexed by zero. This is the primary objective.
- `total_cost`: undiscounted sum of the same period costs.
- `BW_LE`, `BW_SME`: supplier replenishment-order variance divided by the variance of platform orders allocated to that supplier. A zero denominator produces an undefined value (`NaN`), not zero.
- `cumulative_service_level`: demand fulfilled by episode end divided by cumulative demand; it is not an immediate-fill-rate measure.

Summary costs are arithmetic means across path episodes. Bullwhip averages use defined episode ratios. The report checks that node cost components sum to the primary objective and that summaries match per-episode records.

Poisson demand uses 20 means evenly spaced from 92 to 132. The jump generator is in `demand/stochastic_jump.py`; Real20 loads all public paths with the documented scale conversion. For a fixed horizon, `--paths N` takes the first N paths from the same 20-path bank. `--quick` sets one path and 30 periods per scenario.

Use `python demo.py --config configs/demo.yaml` to specify a configuration, or `python demo.py --help` for runtime options. Changing the environment or actor dimensions may invalidate the supplied checkpoints. The saved `run.json` records the evaluated configuration, model hashes, device and elapsed time.
