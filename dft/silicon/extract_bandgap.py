import re

with open("si.nscf.out") as f:
    content = f.read()

blocks = re.findall(r"k =.*?bands \(ev\):\s*\n\n(.*?)\n\n", content, re.DOTALL)

all_occupied = []
all_unoccupied = []

n_occupied_bands = 4  # Silicon: 8 valence electrons / 2 atoms = 4 occupied bands per k-point

for block in blocks:
    energies = [float(x) for x in block.split()]
    occupied = energies[:n_occupied_bands]
    unoccupied = energies[n_occupied_bands:]
    all_occupied.extend(occupied)
    all_unoccupied.extend(unoccupied)

vbm = max(all_occupied)
cbm = min(all_unoccupied)
band_gap = cbm - vbm

print(f"Number of k-points processed: {len(blocks)}")
print(f"Valence Band Maximum (VBM): {vbm:.4f} eV")
print(f"Conduction Band Minimum (CBM): {cbm:.4f} eV")
print(f"Band Gap: {band_gap:.4f} eV")
