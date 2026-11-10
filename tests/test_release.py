"""Release integrity and deterministic episode-reset checks."""

import hashlib
import json
import unittest

import numpy as np
import pandas as pd
import torch

from demand.real_20 import DATA
from steelnet.evaluate import CONFIG, provider_for
from steelnet.io import ROOT, load_yaml
from steelnet.rollout import run_episode
from steelnet.validate_data import validate


class ReleaseTests(unittest.TestCase):
    def test_public_data_matches_workbook(self):
        validate()
        workbook = pd.read_excel(DATA / "real_20.xlsx", sheet_name="paths_wide")
        csv = pd.read_csv(DATA / "real_demand_wide.csv", float_precision="round_trip")
        np.testing.assert_array_equal(workbook.to_numpy(), csv.to_numpy())

    def test_final_checkpoint_checksums(self):
        manifest = json.loads((ROOT / "checkpoints/manifest.json").read_text())
        for model in manifest["models"]:
            self.assertEqual(
                hashlib.sha256((ROOT / model["file"]).read_bytes()).hexdigest(),
                model["sha256"],
            )

    def test_recurrent_state_resets_between_episodes(self):
        torch.set_num_threads(1)
        config = load_yaml(CONFIG)
        demand = np.array([94, 113, 88, 104, 98, 120, 105, 96], dtype=np.float32)
        columns = [
            "platform_order",
            "beta_sme",
            "replenishment_le",
            "replenishment_sme",
            "system_cost",
        ]
        for method in ("happo", "mappo", "heuristic"):
            with self.subTest(method=method):
                policy = provider_for(method, config, "cpu")
                first = pd.DataFrame(
                    run_episode(policy, demand, config, method, "test", 0)
                )
                second = pd.DataFrame(
                    run_episode(policy, demand, config, method, "test", 1)
                )
                np.testing.assert_array_equal(
                    first[columns].to_numpy(), second[columns].to_numpy()
                )


if __name__ == "__main__":
    unittest.main()
