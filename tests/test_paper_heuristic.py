"""Check causality, accounting and comparable action limits of the C.3 rule."""
from collections import deque
from types import SimpleNamespace
import unittest

import numpy as np

from algorithm.paper_heuristic import PaperBaseStockHeuristic
from envs.state import LE, PLATFORM, SME


def fake_env(demands=(100.0, 120.0, 80.0)):
    return SimpleNamespace(
        step_num=0, inventory=[0.0, 0.0, 0.0], backlog=[0.0, 0.0, 0.0],
        m_to_supplier={LE: deque([0.0]), SME: deque([0.0, 0.0])},
        supplier_to_platform={LE: deque([0.0]), SME: deque([0.0, 0.0])},
        m_to_supplier_lead_times={LE: 1, SME: 2},
        supplier_to_platform_lead_times={LE: 1, SME: 2},
        last_platform_orders={LE: 0.0, SME: 0.0}, last_shipments={LE: 0.0, SME: 0.0},
        _get_demand=lambda t: demands[t],
    )


class PaperHeuristicTests(unittest.TestCase):
    def test_platform_inventory_position_and_review_override(self):
        env = fake_env()
        env.inventory[PLATFORM], env.backlog[PLATFORM] = 20.0, 5.0
        env.supplier_to_platform = {LE: deque([10.0]), SME: deque([15.0, 5.0])}
        rule = PaperBaseStockHeuristic(platform_review_period=1.0, safety_z=0.0)
        actions = rule.act_from_env(env)
        self.assertEqual(actions[0][0], 55.0)  # 100 - (20 - 5 + 30)
        self.assertEqual(rule.last_diagnostics['platform_inventory_position'], 45.0)

    def test_supplier_uses_previous_allocations_and_own_inventory_position(self):
        env = fake_env()
        rule = PaperBaseStockHeuristic(safety_z=0.0)
        first = rule.act_from_env(env)
        self.assertEqual(first[1:], [[0.0], [0.0]])
        env.step_num = 1
        env.last_platform_orders = {LE: 40.0, SME: 60.0}
        env.last_shipments = {LE: 30.0, SME: 50.0}
        env.inventory[LE], env.backlog[LE] = 10.0, 5.0
        env.m_to_supplier[LE] = deque([15.0])
        actions = rule.act_from_env(env)
        self.assertEqual(actions[1][0], 60.0)  # 40 * (1+1) - (10-5+15)
        self.assertEqual(actions[2][0], 180.0)  # 60 * (1+2)
        self.assertEqual(rule.last_diagnostics['sme_outstanding_proxy'], 10.0)

    def test_platform_cannot_read_sme_private_inventory_or_backlog(self):
        outputs = []
        for stock, backlog in ((0.0, 0.0), (900.0, 300.0)):
            env = fake_env()
            env.inventory[SME], env.backlog[SME] = stock, backlog
            outputs.append(PaperBaseStockHeuristic().act_from_env(env)[0])
        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_future_demand_does_not_affect_current_action(self):
        outputs = [PaperBaseStockHeuristic().act_from_env(fake_env(d))
                   for d in ((100.0, 0.0, 0.0), (100.0, 99999.0, 99999.0))]
        self.assertEqual(outputs[0], outputs[1])

    def test_caps_minimum_share_and_fallback(self):
        env = fake_env((1e6, 1e6, 1e6))
        env.backlog = [1e6, 1e6, 1e6]
        rule = PaperBaseStockHeuristic(sme_min_share=0.25)
        actions = rule.act_from_env(env)
        self.assertTrue(all(0 <= actions[i][0] <= 220 for i in range(3)))
        self.assertGreaterEqual(actions[0][1], 0.25)
        zero = PaperBaseStockHeuristic(beta_fallback=0.5).act_from_env(fake_env((0, 0, 0)))
        self.assertEqual(zero[0][1], 0.5)

    def test_causal_window_idempotence_and_reset(self):
        env = fake_env()
        rule = PaperBaseStockHeuristic(forecast_window=2)
        initial = rule.act_from_env(env)
        for t in (1, 2):
            env.step_num = t
            first = rule.act_from_env(env)
            self.assertEqual(rule.act_from_env(env), first)
        self.assertEqual(rule.last_diagnostics['forecast_mean'], 100.0)
        self.assertAlmostEqual(rule.last_diagnostics['forecast_std'], np.std([120, 80], ddof=1))
        rule.reset()
        env.step_num = 0
        self.assertEqual(rule.act_from_env(env), initial)

    def test_automatic_review_counts_both_branch_links(self):
        env = fake_env()
        rule = PaperBaseStockHeuristic()
        actions = rule.act_from_env(env)
        beta = actions[0][1]
        self.assertAlmostEqual(rule.last_diagnostics['platform_review_period'], 1 + (1-beta)*2 + beta*4)


if __name__ == '__main__':
    unittest.main()
