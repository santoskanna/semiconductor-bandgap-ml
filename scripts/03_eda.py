import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("features.csv")

# Distribution of band gaps
plt.figure(figsize=(6,4))
sns.histplot(df["band_gap"], bins=10, kde=True)
plt.title("Distribution of Band Gaps in Dataset")
plt.xlabel("Band Gap (eV)")
plt.tight_layout()
plt.savefig("bandgap_distribution.png", dpi=150)
plt.close()

# Correlation heatmap
plt.figure(figsize=(8,6))
corr = df.drop(columns=["material_id", "formula"]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

# Band gap vs electronegativity difference
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="electronegativity_diff", y="band_gap", hue="formula", legend=False)
plt.title("Band Gap vs Electronegativity Difference")
plt.tight_layout()
plt.savefig("bandgap_vs_electroneg.png", dpi=150)
plt.close()

print("Saved 3 plots: bandgap_distribution.png, correlation_heatmap.png, bandgap_vs_electroneg.png")
