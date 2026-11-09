"""
Generate a SMALL SYNTHETIC practice dataset with the same column layout as the
Kaggle "Twitter Bot Detection Dataset" (goyaladi). Use it only to test the code
when you have not downloaded the real file yet.

Run:  python src/generate_sample_data.py
Out:  data/sample_accounts.csv
"""
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
random.seed(42)

N = 3000
CITIES = ["Hyderabad", "Mumbai", "Delhi", "London", "New York", "Austin",
          "Sydney", "Toronto", "Berlin", "Singapore"]
TAGS = ["#ai", "#sale", "#news", "#win", "#crypto", "#love", "#tech", "#free", "#job", "#sports"]


def rand_name(bot: bool) -> str:
    base = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 9)))
    if bot:  # bots tend to have longer names with many digits
        base += "".join(random.choices(string.digits, k=random.randint(3, 8)))
    elif random.random() < 0.2:
        base += str(random.randint(1, 99))
    return base


rows = []
for i in range(N):
    bot = int(rng.random() < 0.45)
    rows.append({
        "User ID": 100000 + i,
        "Username": rand_name(bot),
        "Tweet": " ".join(random.choices(["buy", "now", "great", "day", "today", "love", "win", "click", "new", "post"],
                                         k=random.randint(4, 14))),
        "Retweet Count": int(rng.poisson(40 if bot else 12)),
        "Mention Count": int(rng.poisson(4 if bot else 1.5)),
        "Follower Count": int(rng.lognormal(5.2 if bot else 6.5, 1.2)),
        "Verified": int((not bot) and rng.random() < 0.15),
        "Bot Label": bot,
        "Location": random.choice(CITIES),
        "Created At": (pd.Timestamp("2024-01-01") +
                       pd.Timedelta(days=int(rng.integers(0, 1000)))).strftime("%Y-%m-%d %H:%M:%S"),
        "Hashtags": " ".join(random.choices(TAGS, k=random.randint(0, 4 if bot else 2))),
    })

df = pd.DataFrame(rows)

# Inject ~120 CLONE accounts: copy a real account's profile, tweak username slightly
clones = df[df["Bot Label"] == 0].sample(120, random_state=1).copy()
clones["User ID"] = range(900000, 900000 + len(clones))
clones["Username"] = clones["Username"] + "_"
clones["Follower Count"] = (clones["Follower Count"] * rng.uniform(0.9, 1.1, len(clones))).astype(int)
clones["Bot Label"] = 1
df = pd.concat([df, clones], ignore_index=True)

# Add a little missing data so the cleaning step has something to do
df.loc[df.sample(60, random_state=3).index, "Location"] = np.nan
df.loc[df.sample(40, random_state=4).index, "Hashtags"] = np.nan

out = Path(__file__).resolve().parent.parent / "data" / "sample_accounts.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows -> {out}")
