# Included models

This package evaluates two pretrained three-agent policies and a causal rolling base-stock heuristic. No model fitting occurs when running `demo.py`.

| Model | Checkpoint | Role in the comparison |
| --- | --- | --- |
| HAPPO | `checkpoints/happo.pt` | Final policy after a 14-iteration continuation with a remaining-time centralized critic |
| MAPPO | `checkpoints/mappo.pt` | Fixed pretrained baseline, unchanged during the final HAPPO continuation |

Both policies use independent role-specific actors with two 64-unit feedforward layers and a 64-unit GRU. The platform emits procurement and SME allocation; each supplier emits replenishment. Gaussian latent actions are transformed through a sigmoid and executed with the same quantity caps and integer rules. Evaluation uses deterministic actions and resets recurrent state at every episode.

The models originate from synthetic jump-demand training with seed 2026, including a causal imitation warm-start, short PPO training and actor-head calibration. The final HAPPO continuation additionally used known remaining-episode time in its centralized critic, state/time residual features and an unclipped value loss. This time input is not an additional actor observation or future-demand information. Training and validation used synthetic paths; the released real-demand paths were not used for fitting or checkpoint selection.

MAPPO remains the fixed baseline requested for this demonstration. The two final checkpoints have different continuation histories; this is a comparison of the supplied policies, not an assertion of identical final training budgets. The release reports one training seed and does not infer statistical significance across training seeds.

`checkpoints/manifest.json` records both SHA-256 checksums. The original checkpoint bytes are preserved, including saved model metadata. HAPPO and MAPPO use reward-mixing weights 0.2 and 0.7 respectively; all methods are evaluated with the same system-cost definition.

The heuristic in `algorithm/paper_heuristic.py` uses a 14-period causal rolling forecast, safety factor 1.645, and gap-based routing. The platform uses current demand and its available operating information; supplier forecasts use previously received orders. It does not access future demand or SME private state through the platform's routing rule. Its full parameters are in `configs/demo.yaml`.
