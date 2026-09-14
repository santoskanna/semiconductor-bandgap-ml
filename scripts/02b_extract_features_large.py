import pandas as pd
import numpy as np
from pymatgen.core.periodic_table import Element

df = pd.read_pickle("raw_materials_large.pkl")

def extract_features(structure):
    lattice = structure.lattice
    unique_elements = list(set([str(el) for el in structure.species]))

    atomic_numbers = [Element(el).Z for el in unique_elements]
    electronegativities = [Element(el).X for el in unique_elements]
    atomic_radii = [
        Element(el).atomic_radius or 1.5
        for el in unique_elements
    ]

    return {
        "lattice_a": lattice.a,
        "lattice_b": lattice.b,
        "lattice_c": lattice.c,
        "cell_volume": lattice.volume,
        "mean_atomic_number": np.mean(atomic_numbers),
        "electronegativity_diff": max(electronegativities) - min(electronegativities)
            if len(electronegativities) > 1 else 0.0,
        "mean_electronegativity": np.mean(electronegativities),
        "mean_atomic_radius": np.mean(atomic_radii),
        "num_elements": len(unique_elements),
    }

feature_rows = []
skipped = 0
for _, row in df.iterrows():
    try:
        feats = extract_features(row["structure"])
    except Exception as e:
        skipped += 1
        continue
    feats["material_id"] = row["material_id"]
    feats["formula"] = row["formula"]
    feats["band_gap"] = row["band_gap"]
    feats["density"] = row["density"]
    feats["nsites"] = row["nsites"]
    feature_rows.append(feats)

features_df = pd.DataFrame(feature_rows)
features_df.to_csv("features_large.csv", index=False)
print(f"Saved {len(features_df)} rows to features_large.csv ({skipped} skipped due to errors)")
