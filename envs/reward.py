def node_cost_breakdown(node, inventory, backlog, holding_cost, backlog_cost):
    """节点成本 = 库存持有成本 + 订单积压成本。"""

    holding = holding_cost[node] * inventory[node]
    backlog_penalty = backlog_cost[node] * backlog[node]
    return {
        "inventory_holding_cost": float(holding),
        "order_backlog_cost": float(backlog_penalty),
        "total": float(holding + backlog_penalty),
    }


def mixed_train_rewards(local_rewards, alpha):
    """把各节点负成本转换为局部-全局混合训练奖励。

    local_rewards 中的每个值已经是 -cost_i。
    训练奖励：
    train_reward_i = -[(1-alpha) * system_cost + alpha * cost_i]
    """

    local_costs = [-float(local_reward) for local_reward in local_rewards]
    system_cost = float(sum(local_costs))
    return [
        [-((1.0 - alpha) * system_cost + alpha * local_cost)]
        for local_cost in local_costs
    ]
