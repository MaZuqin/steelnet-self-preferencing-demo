from collections.abc import Mapping

import numpy as np


# 智能体顺序：平台 P、头部供给池 LEs、中小供给池 SMEs。
PLATFORM = 0
LE = 1
SME = 2
AGENT_NAMES = ["platform", "le_supplier", "sme_supplier"]

AGENT_NUM = 3
SUPPLIERS = [LE, SME]

# Base experiment assumptions.
# None means the manufacturer has unlimited supply capacity.
MANUFACTURER_CAPACITY = None

# Pipelines start empty in the base experiment.
INITIAL_PIPELINE_ORDER = 0.0

# episode 长度。
EPISODE_LEN = 366

# 动作设置。
# 平台动作：[O_t, beta_sme]；LE/SME 动作：[replenishment]。
ACTION_DIMS = [2, 1, 1]
ACTION_DIMS_BY_NAME = dict(zip(AGENT_NAMES, ACTION_DIMS))
# 兼容旧代码中单一 action_dim 的写法；真实异质动作维度看 ACTION_DIMS。
ACTION_DIM = ACTION_DIMS[PLATFORM]

# LE 支路包含 M->LE 和 LE->Platform，两段使用同一个 lead time。
# SME 支路包含 M->SME 和 SME->Platform，两段使用同一个 lead time。
LE_BRANCH_LEAD_TIME = 1
SME_BRANCH_LEAD_TIME = 2
PLATFORM_TO_CUSTOMER_LEAD_TIME = 0

# 初始库存。
PLATFORM_INIT_INVENTORY = 160.0
LE_INIT_INVENTORY = 140.0
SME_INIT_INVENTORY = 90.0

# 平台默认分配比例。
DEFAULT_BETA_SME = 0.5

# 成本默认值，顺序均为 P、LE、SME。
H = [100.0, 18, 22]
B = [300.0, 180, 220]

# 平台向供给池采购的基础价格，顺序为 P、LE、SME。
# P 本身不作为采购来源，所以第一个值保留为 0。
# 这些参数只用于 info 统计，不进入当前基础奖励函数。
#
# 设计思路：
# - LEs 代表成熟头部供给池，可靠性更强，基础价格略高，阶梯折扣较温和；
# - SMEs 代表中小供给池，为吸引平台分单，基础价格略低，阶梯折扣更明显。
PURCHASE_PRICE = [0.0, 12.0, 10.8]

# 阶梯折扣采用 all-units quantity discount：
# 当本期采购量达到某个断点后，该折扣作用于该供应商本期全部采购量。
# 例如 SME 订单量为 220 时，命中 200 档，折扣为 10%。
PURCHASE_DISCOUNT_TIERS = {
    "le": [(0, 0.00), (100, 0.02), (200, 0.04), (300, 0.06), (500, 0.08)],
    "sme": [(0, 0.03), (100, 0.06), (200, 0.10), (300, 0.14), (500, 0.18)],
}

# 局部-全局混合奖励参数。
ALPHA = 0.0

# 运输损耗设置。P->Customer 损耗固定为 0。
RANDOM_SHIPPING_LOSS = False
LE_BRANCH_LOST_RATE = 0.05
SME_BRANCH_LOST_RATE = 0.10
PLATFORM_TO_CUSTOMER_LOST_RATE = 0.0
INTEGER_SHIPPING_LOSS = True

# 观测归一化。
NORMALIZE_OBS = True
# 数量型观测统一按千单位缩放；训练脚本仍可通过 obs_scale 覆盖。
OBS_SCALE = 1000.0


def node_values(value, default):
    """把 P、LE、SME 三个节点参数整理成 list。"""

    if value is None:
        return [float(v) for v in default]
    if isinstance(value, Mapping):
        return [
            float(value.get("platform", value.get("p", default[PLATFORM]))),
            float(value.get("le", value.get("le_supplier", default[LE]))),
            float(value.get("sme", value.get("sme_supplier", default[SME]))),
        ]

    values = list(value)
    if len(values) != AGENT_NUM:
        raise ValueError("节点参数必须包含 P、LE、SME 三个值。")
    return [float(v) for v in values]


def discount_tiers(value=None, default=None):
    """整理 LE/SME 的阶梯折扣表。

    支持传入：
    - None：使用默认折扣表；
    - {"le": [(breakpoint, discount), ...], "sme": [...]}；
    - [le_tiers, sme_tiers]；
    - [platform_tiers, le_tiers, sme_tiers]，其中 platform_tiers 会被忽略。
    """

    if default is None:
        default = PURCHASE_DISCOUNT_TIERS

    if value is None:
        raw = {
            LE: default["le"],
            SME: default["sme"],
        }
    elif isinstance(value, Mapping):
        raw = {
            LE: value.get("le", value.get("le_supplier", value.get(LE, default["le"]))),
            SME: value.get("sme", value.get("sme_supplier", value.get(SME, default["sme"]))),
        }
    else:
        values = list(value)
        if len(values) == 2:
            raw = {LE: values[0], SME: values[1]}
        elif len(values) == 3:
            raw = {LE: values[LE], SME: values[SME]}
        else:
            raise ValueError("purchase_discount_tiers 必须包含 LE 和 SME 两个折扣表。")

    return {
        LE: normalize_discount_tiers(raw[LE], label="LE purchase discount tiers"),
        SME: normalize_discount_tiers(raw[SME], label="SME purchase discount tiers"),
    }


def normalize_discount_tiers(tiers, label):
    """把折扣表整理成按订货量升序排列的 [(breakpoint, discount), ...]。"""

    normalized = []
    for item in tiers:
        if isinstance(item, Mapping):
            breakpoint = item.get("breakpoint", item.get("min_qty", item.get("quantity")))
            discount = item.get("discount")
        else:
            breakpoint, discount = item

        breakpoint = float(max(breakpoint, 0.0))
        discount = float(np.clip(discount, 0.0, 1.0))
        normalized.append((breakpoint, discount))

    if not normalized:
        raise ValueError(f"{label} 不能为空。")

    normalized.sort(key=lambda item: item[0])
    if normalized[0][0] > 0.0:
        normalized.insert(0, (0.0, 0.0))

    return normalized


def lead_time_values(lead_time, platform_to_customer_lead_time):
    """整理 LE 支路、SME 支路和 P->C 的提前期。"""

    branch_lead_times = {LE: LE_BRANCH_LEAD_TIME, SME: SME_BRANCH_LEAD_TIME}
    p_to_c = PLATFORM_TO_CUSTOMER_LEAD_TIME

    if isinstance(lead_time, Mapping):
        branch_lead_times[LE] = branch_param_value(
            lead_time,
            names=("le", "m_to_le", "m_to_le_supplier", "le_to_platform", "le_supplier_to_platform"),
            index=LE,
            default=LE_BRANCH_LEAD_TIME,
            cast=int,
            label="LE lead time",
        )
        branch_lead_times[SME] = branch_param_value(
            lead_time,
            names=("sme", "m_to_sme", "m_to_sme_supplier", "sme_to_platform", "sme_supplier_to_platform"),
            index=SME,
            default=SME_BRANCH_LEAD_TIME,
            cast=int,
            label="SME lead time",
        )
        p_to_c = int(
            lead_time.get(
                "platform_to_customer",
                lead_time.get("p_to_c", PLATFORM_TO_CUSTOMER_LEAD_TIME),
            )
        )
    elif lead_time is not None:
        values = list(lead_time)
        if len(values) == 2:
            branch_lead_times = {LE: int(values[0]), SME: int(values[1])}
        elif len(values) == 3:
            branch_lead_times = {LE: int(values[0]), SME: int(values[1])}
            p_to_c = int(values[2])
        else:
            raise ValueError("lead_time 必须包含 2 或 3 个值：LE支路、SME支路、可选P->C。")

    if platform_to_customer_lead_time is not None:
        p_to_c = int(platform_to_customer_lead_time)

    return branch_lead_times, p_to_c


def branch_param_value(mapping, names, index, default, cast, label):
    """读取支路参数；若同一支路两段被写成不同值，则报错。"""

    values = []
    for name in names:
        if name in mapping:
            values.append(mapping[name])
    if index in mapping:
        values.append(mapping[index])

    if not values:
        return cast(default)

    normalized = [cast(v) for v in values]
    if any(v != normalized[0] for v in normalized[1:]):
        raise ValueError(f"{label} 在同一支路上必须保持一致。")
    return normalized[0]


def transport_loss_rates(transport_lost_rate, branch_lost_rate, lost_rate):
    """整理 LE 支路、SME 支路和 P->C 的损耗率。"""

    if lost_rate is not None:
        rate = float(np.clip(lost_rate, 0.0, 1.0))
        branch_rates = {LE: rate, SME: rate}
    else:
        branch_rates = {LE: LE_BRANCH_LOST_RATE, SME: SME_BRANCH_LOST_RATE}

    if branch_lost_rate is not None:
        if isinstance(branch_lost_rate, Mapping):
            le_rate = float(
                np.clip(
                    branch_lost_rate.get("le", branch_lost_rate.get(LE, branch_rates[LE])),
                    0.0,
                    1.0,
                )
            )
            sme_rate = float(
                np.clip(
                    branch_lost_rate.get("sme", branch_lost_rate.get(SME, branch_rates[SME])),
                    0.0,
                    1.0,
                )
            )
        else:
            values = list(branch_lost_rate)
            if len(values) != 2:
                raise ValueError("支路运输损耗率必须包含 LE 和 SME 两个值。")
            le_rate = float(np.clip(values[0], 0.0, 1.0))
            sme_rate = float(np.clip(values[1], 0.0, 1.0))
        branch_rates = {LE: le_rate, SME: sme_rate}

    if transport_lost_rate is not None:
        branch_rates[LE] = branch_param_value(
            transport_lost_rate,
            names=("le", "m_to_le", "m_to_le_supplier", "le_to_platform", "le_supplier_to_platform"),
            index=LE,
            default=branch_rates[LE],
            cast=lambda v: float(np.clip(v, 0.0, 1.0)),
            label="LE loss rate",
        )
        branch_rates[SME] = branch_param_value(
            transport_lost_rate,
            names=("sme", "m_to_sme", "m_to_sme_supplier", "sme_to_platform", "sme_supplier_to_platform"),
            index=SME,
            default=branch_rates[SME],
            cast=lambda v: float(np.clip(v, 0.0, 1.0)),
            label="SME loss rate",
        )

    return {
        "le_branch": float(branch_rates[LE]),
        "sme_branch": float(branch_rates[SME]),
        "platform_to_customer": PLATFORM_TO_CUSTOMER_LOST_RATE,
    }
