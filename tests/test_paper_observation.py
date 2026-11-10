"""Verify manuscript actor information without changing the legacy dynamics."""
import unittest

import numpy as np

from envs.state import AGENT_NAMES, LE, PLATFORM, SME
from envs.steelnet_env import Env


class PaperObservationTests(unittest.TestCase):
    def make_env(self, **kwargs):
        options = dict(
            episode_len=4,
            lead_time=(1, 3, 0),
            initial_inventory=(11, 22, 33),
            initial_pipeline_order=2,
            random_shipping_loss=False,
            normalize_obs=False,
        )
        options.update(kwargs)
        return Env(**options)

    def test_default_legacy_layout_matches_recorded_fields(self):
        env = self.make_env()
        observations = env.reset(demand_list=[101, 102, 103, 104])
        self.assertFalse(env.paper_observation)
        self.assertEqual(env.obs_dim, 12)
        self.assertEqual(env.obs_dims, [12, 12, 12])
        expected = [
            [11, 0, 101, 0, 0, 0.5, 2, 0, 0, 2, 2, 2],
            [22, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
            [33, 0, 0, 0, 0, 6, 2, 2, 2, 0, 0, 0],
        ]
        for actual, reference in zip(observations, expected):
            np.testing.assert_array_equal(actual, reference)

    def test_paper_platform_appends_le_and_preserves_all_transitions(self):
        legacy = self.make_env(paper_observation=False)
        paper = self.make_env(paper_observation=True)
        old_obs = legacy.reset(demand_list=[101, 102, 103, 104])
        new_obs = paper.reset(demand_list=[101, 102, 103, 104])
        self.assertEqual(paper.obs_dims, [24, 12, 12])
        self.assertEqual(paper.obs_dims_by_name, dict(zip(AGENT_NAMES, [24, 12, 12])))
        for action in ([[30, 0.4], [17], [19]], [[50, 0.2], [8], [20]], [[10, 0.7], [35], [12]], [[60, 0.5], [10], [30]]):
            np.testing.assert_array_equal(new_obs[PLATFORM][:legacy.obs_dim], old_obs[PLATFORM])
            np.testing.assert_array_equal(new_obs[PLATFORM][legacy.obs_dim:], old_obs[LE])
            np.testing.assert_array_equal(new_obs[LE], old_obs[LE])
            np.testing.assert_array_equal(new_obs[SME], old_obs[SME])
            np.testing.assert_array_equal(paper.global_state(), legacy.global_state())
            old_obs, old_rewards, old_done, old_info = legacy.step(action)
            new_obs, new_rewards, new_done, new_info = paper.step(action)
            np.testing.assert_array_equal(new_rewards, old_rewards)
            self.assertEqual(new_done, old_done)
            self.assertEqual(new_info, old_info)

    def test_supplier_fields_are_lagged_executed_actions_and_shipments(self):
        env = self.make_env(paper_observation=True)
        before = env.reset(demand_list=[101, 102, 103, 104])
        for supplier in (LE, SME):
            np.testing.assert_array_equal(before[supplier][2:5], [0, 0, 0])
        after, _, _, infos = env.step([[30, 0.4], [17], [19]])
        info = infos[0]
        for supplier in (LE, SME):
            np.testing.assert_array_equal(after[supplier][2:5], [
                info["platform_orders"][supplier],
                info["replenishments"][supplier],
                info["shipments_to_platform"][supplier],
            ])
            self.assertEqual(after[supplier][5], sum(env.supplier_to_platform[supplier]))
            np.testing.assert_array_equal(
                after[supplier][6:6 + env.max_lead_time],
                env._pad_pipeline(env.m_to_supplier[supplier]),
            )
            # Returned observations are snapshots, not views of the new actions.
            np.testing.assert_array_equal(before[supplier][2:5], [0, 0, 0])
        self.assertEqual(after[PLATFORM][2], 102)

    def test_supplier_observation_does_not_receive_current_customer_demand(self):
        left = self.make_env(paper_observation=True)
        right = self.make_env(paper_observation=True)
        left_obs = left.reset(demand_list=[101, 102, 103, 104])
        right_obs = right.reset(demand_list=[999, 998, 997, 996])
        for supplier in (LE, SME):
            np.testing.assert_array_equal(left_obs[supplier], right_obs[supplier])
        self.assertNotEqual(left_obs[PLATFORM][2], right_obs[PLATFORM][2])

    def test_sme_private_state_is_not_appended_to_platform(self):
        env = self.make_env(paper_observation=True)
        before = env.reset(demand_list=[101, 102, 103, 104])
        env.inventory[SME] += 100
        env.backlog[SME] += 200
        env.last_replenishments[SME] = 300
        env.last_shipments[SME] = 400
        env.m_to_supplier[SME] = [500, 600, 700]
        after = env._build_obs(False)
        np.testing.assert_array_equal(after[PLATFORM], before[PLATFORM])
        np.testing.assert_array_equal(after[LE], before[LE])
        self.assertFalse(np.array_equal(after[SME], before[SME]))
        env.inventory[LE] += 10
        changed = env._build_obs(False)
        self.assertEqual(changed[PLATFORM][env.obs_dim], before[LE][0] + 10)

    def test_normalization_is_identical_for_copied_le_block(self):
        env = self.make_env(paper_observation=True, normalize_obs=True, obs_scale=100)
        observations = env.reset(demand_list=[101, 102, 103, 104])
        np.testing.assert_array_equal(observations[PLATFORM][env.obs_dim:], observations[LE])
        self.assertAlmostEqual(float(observations[PLATFORM][0]), 0.11, places=6)
        self.assertEqual(observations[PLATFORM][5], 0.5)
        self.assertAlmostEqual(float(observations[LE][0]), 0.22, places=6)

    def test_dimensions_cover_lead_times_without_changing_global_state(self):
        for lead_times in ((0, 0, 0), (1, 2, 0), (1, 3, 0), (2, 4, 1)):
            for paper in (False, True):
                with self.subTest(lead_times=lead_times, paper=paper):
                    env = self.make_env(lead_time=lead_times, paper_observation=paper)
                    observations = env.reset(demand_list=[101, 102, 103, 104])
                    self.assertEqual([len(obs) for obs in observations], env.obs_dims)
                    self.assertEqual(len(env.global_state()), env.state_dim)
                    self.assertEqual(env.state_dim, 18 + 5 * max(lead_times))


if __name__ == "__main__":
    unittest.main()
