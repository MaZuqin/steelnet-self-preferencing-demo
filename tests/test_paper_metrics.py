import unittest

import numpy as np

from steelnet.costs import NODES, aggregate
from steelnet.metrics import aggregate_episode_metrics


class DiscountedCostTests(unittest.TestCase):
    @staticmethod
    def rows(holding=(100, 20, 20), backlog=(300, 200, 200)):
        inventory = [(1, 2, 3), (2, 1, 0), (0, 0, 1)]
        pending = [(0, 1, 0), (1, 0, 2), (0, 2, 0)]
        rows = []
        for t, (stocks, orders) in enumerate(zip(inventory, pending)):
            row = {
                "t": t,
                "demand": 10 + t,
                "system_cost": float(np.dot(stocks, holding) + np.dot(orders, backlog)),
                "platform_order": 20 + 2 * t,
                "platform_order_le": 10 + t,
                "platform_order_sme": 10 + t,
                "replenishment_le": 20 + 2 * t,
                "replenishment_sme": 5 + 0.5 * t,
            }
            row.update({f"inventory_{node}": value for node, value in zip(NODES, stocks)})
            row.update({f"backlog_{node}": value for node, value in zip(NODES, orders)})
            rows.append(row)
        return rows

    def test_hand_calculated_discount_starts_at_zero(self):
        metrics = aggregate(self.rows(), gamma=0.5)
        self.assertEqual(metrics["total_cost"], 1740)
        self.assertEqual(metrics["discounted_total_cost"], 400 + 0.5 * 920 + 0.25 * 420)
        self.assertEqual(metrics["discount_gamma"], 0.5)
        self.assertEqual(metrics["discounted_total_cost_platform"], 350)
        self.assertEqual(metrics["discounted_total_cost_le"], 350)
        self.assertEqual(metrics["discounted_total_cost_sme"], 265)
        self.assertEqual(metrics["discounted_holding_cost"], 315)
        self.assertEqual(metrics["discounted_backlog_cost"], 650)

    def test_gamma_one_and_every_decomposition_conserve_cost(self):
        for gamma in (0.0, 0.5, 0.99, 1.0):
            metrics = aggregate(self.rows(), gamma=gamma)
            for prefix in ("", "discounted_"):
                total = metrics[f"{prefix}total_cost"]
                self.assertAlmostEqual(total, sum(metrics[f"{prefix}total_cost_{node}"] for node in NODES))
                self.assertAlmostEqual(total, metrics[f"{prefix}holding_cost"] + metrics[f"{prefix}backlog_cost"])
                for node in NODES:
                    self.assertAlmostEqual(metrics[f"{prefix}total_cost_{node}"], metrics[f"{prefix}holding_cost_{node}"] + metrics[f"{prefix}backlog_cost_{node}"])
            if gamma == 1.0:
                self.assertEqual(metrics["discounted_total_cost"], metrics["total_cost"])
                for node in NODES:
                    self.assertEqual(metrics[f"discounted_total_cost_{node}"], metrics[f"total_cost_{node}"])
            if gamma == 0.0:
                self.assertEqual(metrics["discounted_total_cost"], 400)

    def test_all_original_metrics_remain_unchanged(self):
        rows = self.rows()
        original = aggregate_episode_metrics(rows)
        extended = aggregate(rows)
        for name, value in original.items():
            np.testing.assert_equal(extended[name], value, err_msg=name)

    def test_custom_coefficients_and_cost_mismatch(self):
        rows = self.rows(holding=(1, 2, 3), backlog=(4, 5, 6))
        metrics = aggregate(rows, holding_cost=(1, 2, 3), backlog_cost=(4, 5, 6))
        self.assertEqual(metrics["holding_cost"], 21)
        self.assertEqual(metrics["backlog_cost"], 31)
        with self.assertRaisesRegex(AssertionError, "does not match system_cost"):
            aggregate(rows)

    def test_nonzero_initial_period_and_missing_state_rejected(self):
        rows = self.rows()
        rows[0]["t"] = 1
        with self.assertRaisesRegex(ValueError, "starting at t=0"):
            aggregate(rows)
        rows = self.rows()
        del rows[1]["inventory_sme"]
        with self.assertRaises(KeyError):
            aggregate(rows)


if __name__ == "__main__":
    unittest.main()
