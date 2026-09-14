"""
check_result_feature.py

One-off diagnostic: is `result` a deterministic (or near-deterministic)
function of A1_Score..A10_Score in this dataset, or is it independent
information?

Run this once against your real train.csv:

    python check_result_feature.py

Read the printed correlation and max/min difference to decide how `result`
should be handled in the deployed app (see README.md, "About the `result`
feature").
"""

import pandas as pd

df = pd.read_csv("train.csv")

a_cols = [f"A{i}_Score" for i in range(1, 11)]
df["aq10_sum"] = df[a_cols].sum(axis=1)

corr = df["aq10_sum"].corr(df["result"])
diff = (df["result"] - df["aq10_sum"])

print(f"Correlation between sum(A1..A10) and result: {corr:.4f}")
print(f"result range: [{df['result'].min():.3f}, {df['result'].max():.3f}]")
print(f"aq10_sum range: [{df['aq10_sum'].min()}, {df['aq10_sum'].max()}]")
print(f"mean(result - aq10_sum): {diff.mean():.3f}")
print(f"std(result - aq10_sum): {diff.std():.3f}")
print(f"exact matches (result == aq10_sum): {(diff == 0).sum()} / {len(df)}")

print("""
How to read this:
- If correlation is ~1.0 and 'exact matches' is close to 100% of rows:
  result IS (basically) the AQ-10 sum. Auto-compute it in app.py from the
  AQ-10 answers and remove the manual number input.
- If correlation is high (e.g. > 0.8) but matches are rare and std is not
  near 0: result correlates with the AQ-10 answers but carries extra
  independent information (noise, a different scoring formula, or a
  separate instrument). Keep it as a required manual input, but consider
  pre-filling it with aq10_sum as a starting guess the user can adjust.
- If correlation is low: result is essentially independent of A1-A10.
  Keep it as a required manual input with no auto-fill.
""")
