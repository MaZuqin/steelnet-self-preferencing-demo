"""Regression checks for the manuscript's branch-level bullwhip definition."""
import unittest
import numpy as np

from steelnet.metrics import aggregate_episode_metrics


class PaperBullwhipTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [
            {
                'demand':10.0*k,'platform_order':3.0*k,
                'platform_order_le':2.0*k,'platform_order_sme':1.0*k,
                'replenishment_le':6.0*k,'replenishment_sme':0.5*k,
                'system_cost':10.0*k,
            }
            for k in range(1,5)
        ]

    def test_paper_ratio_and_distinct_allocation_ratio(self):
        metrics=aggregate_episode_metrics(self.rows())
        self.assertAlmostEqual(metrics['BW_LE'],9.0)
        self.assertAlmostEqual(metrics['BW_SME'],0.25)
        self.assertAlmostEqual(metrics['BW1'],9.0)
        self.assertAlmostEqual(metrics['BW3'],0.25)
        self.assertAlmostEqual(metrics['allocation_to_customer_variance_ratio_LE'],0.04)
        self.assertAlmostEqual(metrics['allocation_to_customer_variance_ratio_SME'],0.01)
        self.assertAlmostEqual(metrics['BW_total'],0.09)
        self.assertEqual(metrics['total_cost'],100.0)

    def test_constant_customer_demand_does_not_invalidate_branch_ratio(self):
        rows=self.rows()
        for row in rows: row['demand']=100.0
        metrics=aggregate_episode_metrics(rows)
        self.assertAlmostEqual(metrics['BW_LE'],9.0)
        self.assertAlmostEqual(metrics['BW_SME'],0.25)
        self.assertTrue(np.isnan(metrics['allocation_to_customer_variance_ratio_LE']))

    def test_constant_received_orders_have_undefined_bullwhip(self):
        rows=self.rows()
        for row in rows: row['platform_order_sme']=2.0
        self.assertTrue(np.isnan(aggregate_episode_metrics(rows)['BW_SME']))

    def test_missing_replenishment_is_not_reported_as_zero(self):
        rows=self.rows()
        for row in rows: del row['replenishment_le']
        metrics=aggregate_episode_metrics(rows)
        self.assertTrue(np.isnan(metrics['BW_LE']))
        self.assertAlmostEqual(metrics['allocation_to_customer_variance_ratio_LE'],0.04)


if __name__=='__main__': unittest.main()
