import pandas as pd
from matminer.featurizers.composition import ElementProperty

df = pd.read_csv("../data/features_large.csv")

raw = pd.read_pickle("../data/raw_materials_large.pkl")
raw = raw[["material_id", "structure"]]

merged = df.merge(raw, on="material_id", how="left")
merged["composition"] = merged["structure"].apply(lambda s: s.composition)

featurizer = ElementProperty.from_preset("magpie")

print("Computing Magpie composition features (may take a minute)...")
out = featurizer.featurize_dataframe(merged, col_id="composition", ignore_errors=True)

out = out.drop(columns=["structure", "composition"])
out.to_csv("../data/features_with_magpie.csv", index=False)

print(f"Saved {len(out)} rows, {out.shape[1]} columns")
print(f"New composition columns added: {out.shape[1] - df.shape[1]}")
