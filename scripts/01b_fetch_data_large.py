import os
import pandas as pd
from mp_api.client import MPRester

API_KEY = os.environ["MP_API_KEY"]

rows = []

with MPRester(API_KEY) as mpr:
    # Broad query: binary compounds that are semiconductors or insulators,
    # with a reasonable band gap range, restricted to stable phases only
    docs = mpr.materials.summary.search(
        band_gap=(0.01, 6.0),          # exclude pure metals (band_gap = 0)
        num_elements=(1, 3),           # simple/binary/ternary compounds
        energy_above_hull=(0, 0.05),   # only stable or near-stable phases
        fields=["material_id", "formula_pretty", "band_gap",
                "density", "volume", "nsites",
                "structure", "symmetry", "energy_above_hull"],
        num_chunks=10, chunk_size=100  # fetch up to ~1000 candidates
    )

print(f"Retrieved {len(docs)} candidate materials")

# Deduplicate by formula, keeping the most stable phase per formula
best_by_formula = {}
for doc in docs:
    f = doc.formula_pretty
    if f not in best_by_formula or doc.energy_above_hull < best_by_formula[f].energy_above_hull:
        best_by_formula[f] = doc

# Cap dataset size for reasonable runtime (adjust as needed)
selected_docs = list(best_by_formula.values())[:150]

for doc in selected_docs:
    rows.append({
        "material_id": doc.material_id,
        "formula": doc.formula_pretty,
        "band_gap": doc.band_gap,
        "density": doc.density,
        "volume": doc.volume,
        "nsites": doc.nsites,
        "crystal_system": doc.symmetry.crystal_system.value,
        "energy_above_hull": doc.energy_above_hull,
        "structure": doc.structure,
    })

df = pd.DataFrame(rows)
df.to_pickle("raw_materials_large.pkl")
print(f"Saved {len(df)} materials to raw_materials_large.pkl")
