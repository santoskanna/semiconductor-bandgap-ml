import os
import pandas as pd
from mp_api.client import MPRester

API_KEY = os.environ["MP_API_KEY"]

materials = [
    "Si", "Ge", "GaAs", "GaN", "GaP", "InP", "InAs", "InSb",
    "ZnO", "ZnS", "ZnSe", "CdS", "CdSe", "CdTe", "AlAs", "AlP",
    "SiC", "BN", "C"
]

rows = []

with MPRester(API_KEY) as mpr:
    for formula in materials:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=["material_id", "formula_pretty", "band_gap",
                    "density", "volume", "nsites",
                    "structure", "symmetry", "energy_above_hull"]
        )
        if not docs:
            print(f"No data found for {formula}")
            continue

        # Pick the most stable polymorph (lowest energy above hull),
        # not just the first result — avoids grabbing metastable/metallic phases
        doc = min(docs, key=lambda d: d.energy_above_hull if d.energy_above_hull is not None else float("inf"))

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
        print(f"Fetched {formula}: band_gap = {doc.band_gap:.3f} eV (E_above_hull = {doc.energy_above_hull:.4f})")

df = pd.DataFrame(rows)
df.to_pickle("raw_materials.pkl")
print(f"\nSaved {len(df)} materials to raw_materials.pkl")
