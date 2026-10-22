"""Generate example/data.csv (synthetic wage data, 200 rows)."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261023)
n = 200

educ = rng.integers(8, 21, n)
exper = rng.integers(0, 31, n)
female = rng.random(n) < 0.5
region = rng.integers(1, 5, n)
sector = rng.choice(["manufacturing", "service", "public"], n)
urban = (rng.random(n) < 0.7).astype(int)

lwage = (
    0.6
    + 0.08 * educ
    + 0.03 * exper
    - 0.0006 * exper**2
    - 0.12 * female
    + 0.10 * urban
    + 0.05 * (region - 2.5)
    + rng.normal(0, 0.3, n)
)

df = pd.DataFrame(
    {
        "id": [f"p{i:03d}" for i in range(1, n + 1)],
        "wage": np.exp(lwage).round(2),
        "lwage": lwage.round(4),
        "educ": educ,
        "exper": exper,
        "female": female,
        "region": region,
        "sector": sector,
        "urban": urban,
    }
)
df.to_csv(Path(__file__).with_name("data.csv"), index=False)
