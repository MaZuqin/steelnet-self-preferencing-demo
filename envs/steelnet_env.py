import random
from collections.abc import Mapping

import numpy as np

try:
    from .reward import mixed_train_rewards, node_cost_breakdown
    from .state import (
        ACTION_DIM,
        ACTION_DIMS,
        AGENT_NAMES,
        AGENT_NUM,
        ALPHA,
        B,
        DEFAULT_BETA_SME,
        EPISODE_LEN,
        H,
        INITIAL_PIPELINE_ORDER,
        INTEGER_SHIPPING_LOSS,
        LE,
        MANUFACTURER_CAPACITY,
        NORMALIZE_OBS,
        OBS_SCALE,
        PLATFORM,
        PLATFORM_INIT_INVENTORY,
        PURCHASE_DISCOUNT_TIERS,
        PURCHASE_PRICE,
        RANDOM_SHIPPING_LOSS,
        SME,
        SME_INIT_INVENTORY,
        SUPPLIERS,
        LE_INIT_INVENTORY,
        discount_tiers as _discount_tiers,
        lead_time_values as _lead_time_values,
        node_values as _node_values,
        transport_loss_rates as _transport_loss_rates,
    )
except ImportError:
    from reward import mixed_train_rewards, node_cost_breakdown
    from state import (
        ACTION_DIM,
        ACTION_DIMS,
        AGENT_NAMES,
        AGENT_NUM,
        ALPHA,
        B,
        DEFAULT_BETA_SME,
        EPISODE_LEN,
        H,
        INITIAL_PIPELINE_ORDER,
        INTEGER_SHIPPING_LOSS,
        LE,
        MANUFACTURER_CAPACITY,
        NORMALIZE_OBS,
        OBS_SCALE,
        PLATFORM,
        PLATFORM_INIT_INVENTORY,
        PURCHASE_DISCOUNT_TIERS,
        PURCHASE_PRICE,
        RANDOM_SHIPPING_LOSS,
        SME,
        SME_INIT_INVENTORY,
        SUPPLIERS,
        LE_INIT_INVENTORY,
        discount_tiers as _discount_tiers,
        lead_time_values as _lead_time_values,
        node_values as _node_values,
        transport_loss_rates as _transport_loss_rates,
    )


# ==============================================================================
# SteelNet 基础环境
#
# 本文件只描述 Y 型供应链的状态转移。
# 参数整理放在 state.py，成本和训练奖励放在 reward.py。
# 需求过程、智能体算法、实验配置和绘图逻辑放在各自目录中。
# ==============================================================================

# ==============================================================================
# 需求接口
#
# 这里保留 serial.py 中 get_training_data / get_eval_data 的格式。
# 但具体需求生成建议放在 demand/ 目录中，在实验脚本里传给 reset(demand_list=...)。
# ==============================================================================

def get_training_data():
    """返回一条训练需求序列。

    真实项目中建议在 experiments/ 里调用 demand/stationary.py 等模块，
    然后把生成好的 demand_list 传给 env.reset(demand_list=...)。
    """

    return [0.0 for _ in range(EPISODE_LEN)]


def get_eval_data():
    """返回评估需求数据。

    这里给空实现，保持和 serial.py 一样的接口位置。
    """

    return 0, []



class Env(object):
    """SteelNet Y 型基础环境。

    - 平台 P 输出总订货量 O_t，并决定分配给 SMEs 的比例 beta_t；
    - LE/SME 供给池决定向制造商 M 的补货申请；
    - 本环境不包含履约努力 effort，供应商动作只表示补货申请量；
    - 平台 backlog 表示尚未发给客户的订单；
    - LE/SME backlog 表示尚未发给平台的订单；
    - inventory 表示各节点当前持有的货物积压；
    - LE 支路、SME 支路、P->C 的提前期和损耗分开记录；
    - reward = -(库存持有成本 + 订单积压成本)。
    """

    def __init__(
        self,
        episode_len=EPISODE_LEN,
        initial_inventory=None,
        lead_time=None,
        platform_to_customer_lead_time=None,
        holding_cost=None,
        backlog_cost=None,
        purchase_price=None,
        purchase_discount=None,
        purchase_discount_tiers=None,
        include_purchase_cost_in_reward=False,
        default_beta_sme=DEFAULT_BETA_SME,
        alpha=ALPHA,
        random_shipping_loss=RANDOM_SHIPPING_LOSS,
        transport_lost_rate=None,
        branch_lost_rate=None,
        lost_rate=None,
        integer_shipping_loss=INTEGER_SHIPPING_LOSS,
        normalize_obs=NORMALIZE_OBS,
        obs_scale=OBS_SCALE,
        manufacturer_capacity=MANUFACTURER_CAPACITY,
        initial_pipeline_order=INITIAL_PIPELINE_ORDER,
        paper_observation=False,
    ):
        if manufacturer_capacity is not None:
            raise ValueError(
                "Base SteelNet assumes infinite manufacturer supply; "
                "use manufacturer_capacity=None."
            )
        self.manufacturer_capacity = None
        self.initial_pipeline_order = float(max(initial_pipeline_order, 0.0))

        self.initial_inventory = _node_values(
            initial_inventory,
            [PLATFORM_INIT_INVENTORY, LE_INIT_INVENTORY, SME_INIT_INVENTORY],
        )
        self.branch_lead_times, self.platform_to_customer_lead_time = _lead_time_values(
            lead_time, platform_to_customer_lead_time
        )
        self.m_to_supplier_lead_times = dict(self.branch_lead_times)
        self.supplier_to_platform_lead_times = dict(self.branch_lead_times)
        self.max_lead_time = max(
            self.branch_lead_times[LE],
            self.branch_lead_times[SME],
            self.platform_to_customer_lead_time,
        )
        self.holding_cost = _node_values(holding_cost, H)
        self.backlog_cost = _node_values(backlog_cost, B)
        self.purchase_price = _node_values(purchase_price, PURCHASE_PRICE)
        self.purchase_discount_tiers = _discount_tiers(
            self._fixed_purchase_discount_tiers(purchase_discount)
            if purchase_discount is not None
            else purchase_discount_tiers,
            PURCHASE_DISCOUNT_TIERS,
        )
        self.include_purchase_cost_in_reward = bool(include_purchase_cost_in_reward)
        self.default_beta_sme = float(np.clip(default_beta_sme, 0.0, 1.0))
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.random_shipping_loss = bool(random_shipping_loss)
        self.transport_lost_rate = _transport_loss_rates(
            transport_lost_rate, branch_lost_rate, lost_rate
        )
        self.integer_shipping_loss = bool(integer_shipping_loss)
        self.default_normalize = bool(normalize_obs)
        self.paper_observation = bool(paper_observation)

        # ------------------------------------------------------------------
        # serial.py 中要求定义的几个成员变量。
        # ------------------------------------------------------------------
        self.agent_num = AGENT_NUM
        # Keep the historical scalar as the width of one padded local block.
        # Actors must use obs_dims for the optional heterogeneous Eq. (9c) view.
        self.obs_dim = 6 + 2 * self.max_lead_time
        self.obs_dims = [self.obs_dim for _ in range(self.agent_num)]
        if self.paper_observation:
            self.obs_dims[PLATFORM] += self.obs_dim
        self.obs_dims_by_name = dict(zip(AGENT_NAMES, self.obs_dims))
        # The existing global_state layout is unaffected by actor information.
        self.state_dim = 18 + 5 * self.max_lead_time
        fallback_obs_scale = max(
            self.initial_inventory[PLATFORM],
            self.initial_inventory[LE],
            self.initial_inventory[SME],
            1.0,
        )
        self.obs_scale = float(obs_scale) if float(obs_scale) > 0.0 else float(fallback_obs_scale)
        # 兼容 serial.py 的单一 action_dim；真实异质动作维度看 action_dims。
        self.action_dim = ACTION_DIM
        self.action_dims = list(ACTION_DIMS)
        self.order_action_upper_bound = None

        self.eval_episode_len = int(episode_len)
        self.episode_max_steps = int(episode_len)

        # ------------------------------------------------------------------
        # SteelNet 状态变量。
        # ------------------------------------------------------------------
        self.inventory = []
        self.backlog = []

        # M -> j 的在途队列和 j -> P 的在途队列。
        # 队列长度由各支路 lead time 单独决定。
        self.m_to_supplier = {}
        self.supplier_to_platform = {}
        self.platform_to_customer = []

        self.demand_list = []
        self.step_num = 0
        self.train = True
        self.normalize = self.default_normalize

        self._reset_step_records()

        # 用于评估时记录动作波动，接口位置和 serial.py 保持一致。
        self.action_history = [[] for _ in range(AGENT_NUM)]
        self.record_act_sta = [[] for _ in range(AGENT_NUM)]
        self.eval_bw_res = [0.0 for _ in range(AGENT_NUM)]

        self.n_eval, self.eval_data = get_eval_data()
        self.eval_index = 0

    def _reset_step_records(self):
        """清空上一期动作、发货和统计记录。"""

        self.last_platform_order = 0.0
        self.last_beta_sme = self.default_beta_sme
        self.last_platform_orders = {LE: 0.0, SME: 0.0}
        self.last_replenishments = {LE: 0.0, SME: 0.0}
        self.last_shipments = {LE: 0.0, SME: 0.0}
        self.last_customer_shipment = 0.0
        self.last_customer_arrival = 0.0

        self.current_orders = []
        self.current_info = {}
        self.current_shipping_loss = {}

    def reset(self, train=True, normalize=None, demand_list=None):
        """重置环境。

        参数：
        - train：是否训练模式。评估模式会尝试读取 self.eval_data。
        - normalize：是否对数量观测做归一化。
        - demand_list：外部需求序列。推荐由 demand/ 目录下的文件生成后传入。
        """

        self.step_num = 0
        self.train = train
        if normalize is None:
            normalize = self.default_normalize
        self.normalize = normalize

        self.inventory = [
            self.initial_inventory[PLATFORM],
            self.initial_inventory[LE],
            self.initial_inventory[SME],
        ]
        self.backlog = [0.0 for _ in range(AGENT_NUM)]

        self.m_to_supplier = {
            LE: self._empty_pipeline(
                self.m_to_supplier_lead_times[LE], self.initial_pipeline_order
            ),
            SME: self._empty_pipeline(
                self.m_to_supplier_lead_times[SME], self.initial_pipeline_order
            ),
        }
        self.supplier_to_platform = {
            LE: self._empty_pipeline(
                self.supplier_to_platform_lead_times[LE], self.initial_pipeline_order
            ),
            SME: self._empty_pipeline(
                self.supplier_to_platform_lead_times[SME], self.initial_pipeline_order
            ),
        }
        self.platform_to_customer = self._empty_pipeline(
            self.platform_to_customer_lead_time, self.initial_pipeline_order
        )

        self._reset_step_records()
        self.action_history = [[] for _ in range(AGENT_NUM)]

        if demand_list is not None:
            self.demand_list = list(demand_list)
        elif train:
            self.demand_list = [0.0 for _ in range(self.episode_max_steps)]
        elif self.n_eval > 0:
            self.demand_list = self.eval_data[self.eval_index]
            self.eval_index += 1
            if self.eval_index == self.n_eval:
                self.eval_index = 0
        else:
            self.demand_list = [0.0 for _ in range(self.episode_max_steps)]

        sub_agent_obs = self.get_reset_obs(normalize)
        return sub_agent_obs

    def step(self, actions, one_hot=False):
        """环境交互一步。

        返回格式保持 serial.py 风格：
        [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
        """

        action = self.action_map(actions, one_hot=one_hot)
        reward = self.state_update(action)
        sub_agent_obs = self.get_step_obs(action)
        sub_agent_reward = self.get_processed_rewards(reward)

        done = self.step_num >= self.episode_max_steps
        sub_agent_done = [done for _ in range(self.agent_num)]
        sub_agent_info = [self.current_info for _ in range(self.agent_num)]

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def get_eval_num(self):
        return self.n_eval

    def get_eval_bw_res(self):
        """返回评估期动作波动指标。

        这里记录的是：
        - 平台订货量 O_t 的变异系数；
        - LE 补货申请的变异系数；
        - SME 补货申请的变异系数。
        """

        return self.eval_bw_res

    def get_demand(self):
        if self.step_num <= 0:
            return [self._get_demand(0)]
        return [self._get_demand(self.step_num - 1)]

    def get_orders(self):
        """返回当前实际动作，便于训练时打印。"""

        return self.current_orders

    def get_inventory(self):
        return self.inventory

    def global_state(self, normalize=None):
        """返回 CTDE critic 使用的全局状态。

        actor 仍然使用各自局部观测；critic 在训练阶段可以使用这个联合状态。
        字段顺序固定，后续接网络时不要随意改。
        """

        if normalize is None:
            normalize = self.normalize

        values = [
            # 当前外生需求和平台上一期决策。
            self._q(self._get_demand(self.step_num), normalize),
            self._q(self.last_platform_order, normalize),
            self.last_beta_sme,
            # 客户侧交付和 P->C 在途。
            self._q(self.last_customer_shipment, normalize),
            self._q(self.last_customer_arrival, normalize),
            self._q(sum(self.platform_to_customer), normalize),
            # 三个节点的库存和订单积压。
            self._q(self.inventory[PLATFORM], normalize),
            self._q(self.backlog[PLATFORM], normalize),
            self._q(self.inventory[LE], normalize),
            self._q(self.backlog[LE], normalize),
            self._q(self.inventory[SME], normalize),
            self._q(self.backlog[SME], normalize),
            # 上一期平台分单、供给池补货和供给池发货。
            self._q(self.last_platform_orders[LE], normalize),
            self._q(self.last_platform_orders[SME], normalize),
            self._q(self.last_replenishments[LE], normalize),
            self._q(self.last_replenishments[SME], normalize),
            self._q(self.last_shipments[LE], normalize),
            self._q(self.last_shipments[SME], normalize),
        ]
        values += self._q_list(self._pad_pipeline(self.supplier_to_platform[LE]), normalize)
        values += self._q_list(self._pad_pipeline(self.supplier_to_platform[SME]), normalize)
        values += self._q_list(self._pad_pipeline(self.platform_to_customer), normalize)
        values += self._q_list(self._pad_pipeline(self.m_to_supplier[LE]), normalize)
        values += self._q_list(self._pad_pipeline(self.m_to_supplier[SME]), normalize)

        return np.asarray(values, dtype=np.float32)

    def action_map(self, actions, one_hot=False):
        """把智能体输出映射为实际决策量。

        输出格式：
        [O_t, beta_sme, le_replenishment, sme_replenishment]

        one_hot=False 时，推荐输入：
        [
            [O_t, beta_sme],
            [le_replenishment],
            [sme_replenishment],
        ]
        其中 O_t、le_replenishment、sme_replenishment 会被四舍五入成整数。
        beta_sme 保持连续。

        one_hot=True 时仅作为兼容：
        - 平台动作可以传 [q_logits, beta_sme]；
        - 供应商动作可以传补货量 logits；
        - logits 的长度由算法输出层决定，环境不设置订货上限。
        推荐后续 HAPPO/MAPPO 直接使用 one_hot=False 的标量动作。
        """

        if isinstance(actions, Mapping):
            platform_action = actions.get(
                "platform", actions.get(PLATFORM, [0.0, self.default_beta_sme])
            )
            le_action = actions.get("le_supplier", actions.get(LE, [0.0]))
            sme_action = actions.get("sme_supplier", actions.get(SME, [0.0]))
            actions = [platform_action, le_action, sme_action]

        platform_order, beta_sme = self._decode_platform_action(
            actions[PLATFORM], one_hot=one_hot
        )
        le_replenishment = self._decode_order_action(actions[LE], one_hot=one_hot)
        sme_replenishment = self._decode_order_action(actions[SME], one_hot=one_hot)

        mapped_actions = [
            int(platform_order),
            float(beta_sme),
            int(le_replenishment),
            int(sme_replenishment),
        ]
        self.current_orders = mapped_actions
        return mapped_actions

    def _decode_platform_action(self, platform_action, one_hot=False):
        """解析平台动作。

        one_hot=False:
            platform_action = [O_t, beta_sme]
            O_t 直接表示平台根据客户需求形成的订货量，环境转成非负整数。

        one_hot=True:
            推荐 platform_action = [q_logits, beta_sme]
            q_logits 用 argmax 得到整数 O_t，beta_sme 保持连续。
            如果只传一个 logits 向量，则 beta_sme 使用默认值。
        """

        if one_hot:
            if isinstance(platform_action, (list, tuple)) and len(platform_action) == 2:
                q_logits = platform_action[0]
                beta_raw = self._as_array(platform_action[1])
                beta_sme = beta_raw[0] if beta_raw.size >= 1 else self.default_beta_sme
            else:
                q_logits = platform_action
                beta_sme = self.default_beta_sme

            q_idx = int(np.argmax(q_logits))
            platform_order = self._to_order_int(q_idx)
        else:
            arr = self._as_array(platform_action)
            q_value = arr[0] if arr.size >= 1 else 0.0
            beta_sme = arr[1] if arr.size >= 2 else self.default_beta_sme
            platform_order = self._to_order_int(q_value)

        beta_sme = float(np.clip(beta_sme, 0.0, 1.0))
        return platform_order, beta_sme

    def _decode_order_action(self, order_action, one_hot=False):
        """解析 LE/SME 的补货动作，输出非负整数补货量。"""

        if one_hot:
            order_value = int(np.argmax(order_action))
        else:
            arr = self._as_array(order_action)
            order_value = arr[0] if arr.size >= 1 else 0.0
        return self._to_order_int(order_value)

    @staticmethod
    def _to_order_int(value):
        """把订货动作转成非负整数，不设置最高上限。"""

        return int(round(max(float(value), 0.0)))

    def get_reset_obs(self, normalize):
        """返回初始观测。"""

        return self._build_obs(normalize)

    def get_step_obs(self, action):
        """返回 step 后观测。"""

        return self._build_obs(self.normalize)

    def get_processed_rewards(self, reward):
        """处理奖励，格式参考 serial.py。

        reward 输入已经是各节点的负成本，成本项只包含库存持有和订单积压。
        训练时使用：
        train_reward_i = -[(1-alpha) * system_cost + alpha * cost_i]
        评估时直接返回局部奖励。
        """

        if self.train:
            processed_rewards = mixed_train_rewards(reward, self.alpha)
        else:
            processed_rewards = [[i] for i in reward]

        return processed_rewards

    def state_update(self, action):
        """按照论文中的 Y 型平台供应链逻辑更新状态。

        action = [O_t, beta_sme, q_LM,t, q_SM,t]
        """

        platform_order, beta_sme, le_replenishment, sme_replenishment = action
        demand_t = self._get_demand(self.step_num)
        self.current_shipping_loss = {}

        # 1. 本期在途到货先进入库存。
        arrivals_from_m = {
            LE: self._pop_pipeline(self.m_to_supplier[LE]),
            SME: self._pop_pipeline(self.m_to_supplier[SME]),
        }
        arrivals_to_platform = {
            LE: self._pop_pipeline(self.supplier_to_platform[LE]),
            SME: self._pop_pipeline(self.supplier_to_platform[SME]),
        }
        arrival_to_customer = self._pop_pipeline(self.platform_to_customer)
        self.inventory[LE] += arrivals_from_m[LE]
        self.inventory[SME] += arrivals_from_m[SME]
        self.inventory[PLATFORM] += arrivals_to_platform[LE] + arrivals_to_platform[SME]

        # 2. 平台将整数 O_t 按连续 beta_t 分配到 LEs 和 SMEs。
        # beta 本身保持连续；实际支路订单量取整数，并保证两支路之和等于 O_t。
        sme_order = int(round(platform_order * beta_sme))
        sme_order = int(np.clip(sme_order, 0, platform_order))
        le_order = int(platform_order - sme_order)
        platform_orders = {
            LE: le_order,
            SME: sme_order,
        }

        # 3. L/S 向制造商补货，补货进入 M -> j 在途队列。
        replenishments = {
            LE: le_replenishment,
            SME: sme_replenishment,
        }
        for supplier in SUPPLIERS:
            self._schedule_m_to_supplier(
                supplier=supplier,
                amount=replenishments[supplier],
                lead_time=self.m_to_supplier_lead_times[supplier],
            )

        # 4. L/S 根据库存和积压订单向平台发货，发货进入 j -> P 在途队列。
        shipments_to_platform = {}
        for supplier in SUPPLIERS:
            pending = platform_orders[supplier] + self.backlog[supplier]
            shipment = min(self.inventory[supplier], pending)
            self.inventory[supplier] -= shipment
            self.backlog[supplier] = max(pending - shipment, 0.0)

            shipments_to_platform[supplier] = shipment
            self._schedule_supplier_to_platform(
                supplier=supplier,
                amount=shipment,
                lead_time=self.supplier_to_platform_lead_times[supplier],
            )

        # 5. 平台满足终端需求。
        # 基础模型采用“延迟满足”：本期没发出的客户需求进入平台订单积压。
        # 这里不做丢单；backlog 就是后续必须继续满足的客户订单。
        customer_pending = demand_t + self.backlog[PLATFORM]
        customer_shipment = min(self.inventory[PLATFORM], customer_pending)
        customer_backlog = max(customer_pending - customer_shipment, 0.0)
        self.inventory[PLATFORM] -= customer_shipment
        self.backlog[PLATFORM] = customer_backlog
        customer_arrival = arrival_to_customer + self._schedule_platform_to_customer(
            customer_shipment
        )

        # 6. 基础奖励：只惩罚货物积压和订单积压。
        cost_breakdown = {
            AGENT_NAMES[node]: self._node_cost_breakdown(node)
            for node in range(AGENT_NUM)
        }
        purchase_info = self._purchase_info(platform_orders, shipments_to_platform)
        for breakdown in cost_breakdown.values():
            breakdown.setdefault("purchase_order_cost", 0.0)
            breakdown.setdefault("purchase_order_gross_amount", 0.0)
            breakdown.setdefault("purchase_discount_saving", 0.0)
        if self.include_purchase_cost_in_reward:
            platform_breakdown = cost_breakdown[AGENT_NAMES[PLATFORM]]
            purchase_order_cost = float(purchase_info["total_order_net_amount"])
            platform_breakdown["purchase_order_cost"] = purchase_order_cost
            platform_breakdown["purchase_order_gross_amount"] = float(
                purchase_info["total_order_gross_amount"]
            )
            platform_breakdown["purchase_discount_saving"] = float(
                purchase_info["total_order_discount_saving"]
            )
            platform_breakdown["total"] = float(platform_breakdown["total"]) + purchase_order_cost
        local_costs = [
            cost_breakdown[AGENT_NAMES[PLATFORM]]["total"],
            cost_breakdown[AGENT_NAMES[LE]]["total"],
            cost_breakdown[AGENT_NAMES[SME]]["total"],
        ]
        rewards = [-float(cost) for cost in local_costs]
        # 7. 保存状态，供观测和外部统计使用。
        self.last_platform_order = platform_order
        self.last_beta_sme = beta_sme
        self.last_platform_orders = platform_orders
        self.last_replenishments = replenishments
        self.last_shipments = shipments_to_platform
        self.last_customer_shipment = customer_shipment
        self.last_customer_arrival = customer_arrival

        self.action_history[PLATFORM].append(platform_order)
        self.action_history[LE].append(le_replenishment)
        self.action_history[SME].append(sme_replenishment)

        self.current_info = {
            "t": self.step_num,
            "demand": demand_t,
            "O_t": platform_order,
            "platform_order": platform_order,
            "beta_sme": beta_sme,
            "order_action_upper_bound": self.order_action_upper_bound,
            "platform_orders": dict(platform_orders),
            "replenishments": dict(replenishments),
            "shipments_to_platform": dict(shipments_to_platform),
            "arrivals_from_m": dict(arrivals_from_m),
            "arrivals_to_platform": dict(arrivals_to_platform),
            "arrival_to_customer": arrival_to_customer,
            "customer_pending": customer_pending,
            "customer_shipment": customer_shipment,
            "customer_arrival": customer_arrival,
            "platform_backlog": self.backlog[PLATFORM],
            "lead_times": self._lead_time_info(),
            "manufacturer_supply": {
                "mode": "infinite",
                "capacity": self.manufacturer_capacity,
            },
            "initial_pipeline_order": self.initial_pipeline_order,
            "transport_lost_rate": dict(self.transport_lost_rate),
            "shipping_loss": dict(self.current_shipping_loss),
            "purchase": purchase_info,
            "cost_coefficients": {
                "holding": dict(zip(AGENT_NAMES, self.holding_cost)),
                "backlog": dict(zip(AGENT_NAMES, self.backlog_cost)),
                "purchase_price": dict(zip(AGENT_NAMES, self.purchase_price)),
                "purchase_discount_tiers": self._discount_tiers_info(),
            },
            "cost_breakdown": cost_breakdown,
            "local_costs": dict(zip(AGENT_NAMES, local_costs)),
            "system_cost": float(sum(local_costs)),
            "sme_order_share": self._safe_share(platform_orders[SME], platform_order),
            "sme_shipment_share": self._safe_share(
                shipments_to_platform[SME],
                shipments_to_platform[LE] + shipments_to_platform[SME],
            ),
        }

        self.step_num += 1
        self._record_eval_action_stats()

        return rewards

    def _build_obs(self, normalize):
        """构造局部观测；可选按式 (9c) 向平台追加关联 LE 观测。"""

        demand_t = self._get_demand(self.step_num)

        platform_obs = [
            self._q(self.inventory[PLATFORM], normalize),
            self._q(self.backlog[PLATFORM], normalize),
            self._q(demand_t, normalize),
            self._q(sum(self.platform_to_customer), normalize),
            self._q(self.last_platform_order, normalize),
            self.last_beta_sme,
        ]
        platform_obs += self._q_list(
            self._pad_pipeline(self.supplier_to_platform[LE]), normalize
        )
        platform_obs += self._q_list(
            self._pad_pipeline(self.supplier_to_platform[SME]), normalize
        )

        le_obs = self._supplier_obs(LE, normalize)
        sme_obs = self._supplier_obs(SME, normalize)
        if self.paper_observation:
            platform_obs += le_obs

        return [
            np.reshape(np.asarray(platform_obs, dtype=np.float32), (self.obs_dims[PLATFORM],)),
            np.reshape(np.asarray(le_obs, dtype=np.float32), (self.obs_dims[LE],)),
            np.reshape(np.asarray(sme_obs, dtype=np.float32), (self.obs_dims[SME],)),
        ]

    def _supplier_obs(self, supplier, normalize):
        """式 (9b)：历史订单、补货和发货仅取上一期已执行值。

        同期平台动作在三个主体同步决策后才写入 last_*，不进入本期
        供应商观测。末尾零填充保留旧检查点的输入宽度，不增加信息。
        """

        obs = [
            self._q(self.inventory[supplier], normalize),
            self._q(self.backlog[supplier], normalize),
            self._q(self.last_platform_orders[supplier], normalize),
            self._q(self.last_replenishments[supplier], normalize),
            self._q(self.last_shipments[supplier], normalize),
            self._q(sum(self.supplier_to_platform[supplier]), normalize),
        ]
        obs += self._q_list(self._pad_pipeline(self.m_to_supplier[supplier]), normalize)
        obs += [0.0 for _ in range(self.max_lead_time)]
        return obs

    def _get_demand(self, t):
        if t >= len(self.demand_list):
            return 0.0
        return float(max(self.demand_list[t], 0.0))

    def _node_cost(self, node):
        """节点成本 = 库存持有成本 + 订单积压成本。"""

        return self._node_cost_breakdown(node)["total"]

    def _node_cost_breakdown(self, node):
        """拆分节点的库存持有成本和订单积压成本。"""

        return node_cost_breakdown(
            node=node,
            inventory=self.inventory,
            backlog=self.backlog,
            holding_cost=self.holding_cost,
            backlog_cost=self.backlog_cost,
        )

    def _purchase_info(self, platform_orders, shipments_to_platform):
        """计算采购价格与折扣统计。

        include_purchase_cost_in_reward=True 时，订单口径的折扣后采购金额
        会计入平台节点成本，进而影响 system_cost 和 reward。
        """

        by_order = {}
        by_shipment = {}
        for supplier in SUPPLIERS:
            price = float(self.purchase_price[supplier])
            order_qty = float(platform_orders[supplier])
            shipment_qty = float(shipments_to_platform[supplier])
            order_discount = self._purchase_discount(supplier, order_qty)
            shipment_discount = self._purchase_discount(supplier, shipment_qty)
            order_effective_price = price * (1.0 - order_discount)
            shipment_effective_price = price * (1.0 - shipment_discount)
            by_order[AGENT_NAMES[supplier]] = {
                "quantity": order_qty,
                "unit_price": price,
                "discount": order_discount,
                "discount_tier": self._purchase_discount_tier(supplier, order_qty),
                "effective_unit_price": order_effective_price,
                "gross_amount": price * order_qty,
                "discount_saving": price * order_discount * order_qty,
                "net_amount": order_effective_price * order_qty,
            }
            by_shipment[AGENT_NAMES[supplier]] = {
                "quantity": shipment_qty,
                "unit_price": price,
                "discount": shipment_discount,
                "discount_tier": self._purchase_discount_tier(supplier, shipment_qty),
                "effective_unit_price": shipment_effective_price,
                "gross_amount": price * shipment_qty,
                "discount_saving": price * shipment_discount * shipment_qty,
                "net_amount": shipment_effective_price * shipment_qty,
            }

        return {
            "discount_scheme": "all_units_quantity_discount",
            "by_order": by_order,
            "by_shipment": by_shipment,
            "total_order_gross_amount": float(
                sum(item["gross_amount"] for item in by_order.values())
            ),
            "total_order_discount_saving": float(
                sum(item["discount_saving"] for item in by_order.values())
            ),
            "total_order_net_amount": float(
                sum(item["net_amount"] for item in by_order.values())
            ),
            "total_shipment_gross_amount": float(
                sum(item["gross_amount"] for item in by_shipment.values())
            ),
            "total_shipment_discount_saving": float(
                sum(item["discount_saving"] for item in by_shipment.values())
            ),
            "total_shipment_net_amount": float(
                sum(item["net_amount"] for item in by_shipment.values())
            ),
            "reward_includes_purchase_cost": bool(self.include_purchase_cost_in_reward),
        }

    def _purchase_discount(self, supplier, quantity):
        """根据本期采购量读取对应供应商的阶梯折扣。"""

        return self._purchase_discount_tier(supplier, quantity)["discount"]

    def _purchase_discount_tier(self, supplier, quantity):
        """返回本期采购量命中的折扣档。"""

        quantity = float(max(quantity, 0.0))
        selected_breakpoint = 0.0
        selected_discount = 0.0
        for breakpoint, discount in self.purchase_discount_tiers[supplier]:
            if quantity >= breakpoint:
                selected_breakpoint = breakpoint
                selected_discount = discount
            else:
                break
        return {
            "breakpoint": float(selected_breakpoint),
            "discount": float(selected_discount),
        }

    def _discount_tiers_info(self):
        """返回 info 中更易读的阶梯折扣表。"""

        return {
            AGENT_NAMES[supplier]: [
                {"breakpoint": float(breakpoint), "discount": float(discount)}
                for breakpoint, discount in self.purchase_discount_tiers[supplier]
            ]
            for supplier in SUPPLIERS
        }

    @staticmethod
    def _fixed_purchase_discount_tiers(purchase_discount):
        """把旧的固定折扣参数转换为 0 档阶梯折扣。"""

        if isinstance(purchase_discount, Mapping):
            return {
                "le": [(0, purchase_discount.get("le", purchase_discount.get("le_supplier", purchase_discount.get(LE, 0.0))))],
                "sme": [(0, purchase_discount.get("sme", purchase_discount.get("sme_supplier", purchase_discount.get(SME, 0.0))))],
            }

        values = list(purchase_discount)
        if len(values) == 2:
            return [[(0, values[0])], [(0, values[1])]]
        if len(values) == 3:
            return [[(0, values[LE])], [(0, values[SME])]]
        raise ValueError("purchase_discount 必须包含 LE 和 SME 两个固定折扣值。")

    def _lead_time_info(self):
        """返回各支路提前期，便于实验记录。"""

        return {
            "le_branch": self.branch_lead_times[LE],
            "sme_branch": self.branch_lead_times[SME],
            "platform_to_customer": self.platform_to_customer_lead_time,
        }

    def _schedule_m_to_supplier(self, supplier, amount, lead_time):
        delivered = self._apply_shipping_loss(
            f"M->{AGENT_NAMES[supplier]}",
            amount,
            self.transport_lost_rate["le_branch" if supplier == LE else "sme_branch"],
        )
        if lead_time <= 0:
            self.inventory[supplier] += delivered
        else:
            self.m_to_supplier[supplier].append(delivered)

    def _schedule_supplier_to_platform(self, supplier, amount, lead_time):
        delivered = self._apply_shipping_loss(
            f"{AGENT_NAMES[supplier]}->P",
            amount,
            self.transport_lost_rate["le_branch" if supplier == LE else "sme_branch"],
        )
        if lead_time <= 0:
            self.inventory[PLATFORM] += delivered
        else:
            self.supplier_to_platform[supplier].append(delivered)

    def _schedule_platform_to_customer(self, amount):
        """安排平台到客户的发货。

        lead time 为 0 时，本期发出就视为本期到达客户；否则进入 P->C 在途队列。
        平台到客户的运输损耗率固定为 0，只保留提前期。
        """

        delivered = self._apply_shipping_loss(
            "P->customer", amount, self.transport_lost_rate["platform_to_customer"]
        )
        if self.platform_to_customer_lead_time <= 0:
            return delivered

        self.platform_to_customer.append(delivered)
        return 0.0

    def _apply_shipping_loss(self, arc, amount, max_loss_rate):
        sent = float(max(amount, 0.0))
        if sent <= 0.0:
            return 0.0

        max_loss_rate = float(np.clip(max_loss_rate, 0.0, 1.0))
        if self.random_shipping_loss and max_loss_rate > 0.0:
            max_loss_step = int(round(max_loss_rate * 100))
            loss_step = random.randint(0, max_loss_step) if max_loss_step > 0 else 0
            loss_rate = loss_step / 100.0
            survival_rate = 1.0 - loss_rate
        else:
            loss_rate = 0.0
            survival_rate = 1.0

        arrival = sent * survival_rate
        if self.random_shipping_loss and max_loss_rate > 0.0 and self.integer_shipping_loss:
            arrival = float(int(arrival))

        self.current_shipping_loss[arc] = {
            "sent": sent,
            "arrival": float(arrival),
            "lost_qty": float(sent - arrival),
            "loss_rate": float(loss_rate),
            "survival_rate": float(survival_rate),
        }
        return float(arrival)

    def _record_eval_action_stats(self):
        if self.train or self.step_num != self.eval_episode_len - 1:
            return

        for k in range(AGENT_NUM):
            mean_action = np.mean(self.action_history[k])
            if mean_action < 1e-6:
                self.record_act_sta[k].append(0.0)
            else:
                self.record_act_sta[k].append(
                    float(np.std(self.action_history[k]) / mean_action)
                )

        if self.n_eval > 0 and self.eval_index == 0:
            self.eval_bw_res = [
                float(np.mean(self.record_act_sta[i])) for i in range(AGENT_NUM)
            ]
            self.record_act_sta = [[] for _ in range(AGENT_NUM)]

    @staticmethod
    def _empty_pipeline(lead_time, initial_value=0.0):
        return [
            float(initial_value)
            for _ in range(max(int(lead_time), 0))
        ]

    @staticmethod
    def _pop_pipeline(queue):
        if not queue:
            return 0.0
        return float(queue.pop(0))

    @staticmethod
    def _as_array(action):
        if isinstance(action, Mapping):
            action = action.get("action", [])
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def _pad_pipeline(self, queue):
        padded = list(queue[: self.max_lead_time])
        while len(padded) < self.max_lead_time:
            padded.append(0.0)
        return padded

    def _q(self, value, normalize):
        if not normalize:
            return float(value)
        return float(value) / self.obs_scale

    def _q_list(self, values, normalize):
        return [self._q(v, normalize) for v in values]

    @staticmethod
    def _safe_share(part, total):
        if total <= 1e-8:
            return 0.0
        return float(part / total)


# 兼容你之前的写法，也方便后面把文件改名为 steelnet_env.py。
SteelNetEnv = Env


if __name__ == "__main__":
    env = Env()
    obs = env.reset(demand_list=[100.0, 105.0, 95.0, 120.0, 110.0])
    print("obs_dim:", env.obs_dim)
    print("action_dim:", env.action_dim)

    for _ in range(5):
        actions = [
            [120, 0.5],   # 平台：[整数 O_t, 连续 beta_sme]
            [100.0],      # LE 向 M 的补货申请
            [80.0],       # SME 向 M 的补货申请
        ]
        obs, reward, done, info = env.step(actions, one_hot=False)
        print(info[0])
        if all(done):
            break
