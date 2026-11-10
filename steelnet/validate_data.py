import hashlib
import json
import numpy as np
import pandas as pd
from demand.real_20 import DATA, load_real20


def validate():
    meta = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name, record in meta["files"].items():
        actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}")
    wide = pd.read_csv(DATA / "real_demand_wide.csv", float_precision="round_trip")
    long = pd.read_csv(DATA / "real_demand_long.csv", float_precision="round_trip")
    if len(long) != 7320 or long.duplicated(["day_index", "path"]).any():
        raise ValueError("Expected 7,320 unique path/time pairs")
    restored = long.pivot(index="day_index", columns="path", values="demand_raw")
    np.testing.assert_array_equal(restored.to_numpy(), wide.iloc[:, 1:].to_numpy())
    bank, names = load_real20()
    assert bank.shape == (20, 366) and len(names) == 20
    print("PASS: 20 x 366 = 7,320 observations; all hashes and wide/long values match.")
    return meta
