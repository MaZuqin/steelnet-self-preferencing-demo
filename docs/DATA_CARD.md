# Public demand data

The release contains all 20 supplied paths, each with 366 observations: 7,320 numeric demand values. Path identifiers are `path_01` through `path_20`; `day_index` runs from 0 to 365. No company names, customer identifiers or transaction IDs are included in these tables.

| File | Contents |
| --- | --- |
| `real_20.xlsx` | Unchanged supplied workbook; `paths_wide` and `paths_long` sheets |
| `real_demand_wide.csv` | `day_index` and 20 numeric path columns |
| `real_demand_long.csv` | `day_index`, `path`, `demand_raw`; 7,320 records |
| `manifest.json` | Dataset metadata and SHA-256 checksums |

The manuscript attributes the underlying material to 2025 steel-order information from China Steel Network. The supplied workbook contains processed numeric paths. Acquisition records, physical units and preprocessing scripts were not included with the supplied dataset. Observation indices are therefore retained without assigning calendar dates or claiming that these are raw transactions.

CSV files preserve the workbook values with round-trip numeric precision. Simulation loading converts values to float32 before division by 1,000, retaining the supplied code's scaling. Public files retain their original values.

Run `python demo.py validate-data` to verify all file checksums, shape and wide/long consistency. The test suite also compares the CSV values against the workbook.

The supplied models were fitted and selected using synthetic demand. All 20 public paths are used for evaluation. The standard synthetic evaluation seeds are 3026 (Poisson) and 3027 (stochastic jump). Each method receives the same demand bank. Quick mode uses shorter episodes and is intended for a runtime check.

See [RIGHTS.md](../RIGHTS.md) for reuse status.
