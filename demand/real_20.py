"""Load the public 20 x 366 demand dataset without an Excel dependency."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / 'data'


def load_real20(scale_divisor: float = 1000.0) -> tuple[np.ndarray, list[str]]:
    if not np.isfinite(scale_divisor) or scale_divisor <= 0:
        raise ValueError('scale_divisor must be positive and finite')
    frame = pd.read_csv(DATA / 'real_demand_wide.csv', float_precision='round_trip')
    columns = [f'path_{i:02d}' for i in range(1, 21)]
    if list(frame.columns) != ['day_index', *columns] or len(frame) != 366:
        raise ValueError('Expected day_index and 20 paths, each with 366 observations')
    if not np.array_equal(frame.day_index.to_numpy(), np.arange(366)):
        raise ValueError('day_index must be 0 through 365 in order')
    raw = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError('Demand must be finite and nonnegative')
    # Preserve the original loader: cast to float32 before division.
    return (raw.astype(np.float32).T / float(scale_divisor)).astype(np.float32), columns


if __name__ == '__main__':
    bank, names = load_real20()
    print(f'{len(names)} paths x {bank.shape[1]} observations; raw / 1000')
