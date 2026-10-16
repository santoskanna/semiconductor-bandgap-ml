import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("../data/features_with_magpie.csv")

structural = [
    "lattice_a", "lattice_b", "lattice_c", "cell_volume",
    "mean_atomic_number", "electronegativity_diff",
    "mean_electronegativity", "mean_atomic_radius", "num_elements",
]
magpie = [c for c in df.columns if c.startswith("MagpieData")]

# Some elements may lack a property, leaving NaN. Drop those rows (and report it).
before = len(df)
df = df.dropna(subset=magpie).reset_index(drop=True)
print(f"Rows kept: {len(df)} of {before} (dropped {before - len(df)} with missing Magpie values)")
print(f"Structural features: {len(structural)} | Magpie features: {len(magpie)}\n")

y = df["band_gap"]
feature_sets = {
    "Structural only": structural,
    "Magpie only": magpie,
    "Structural + Magpie": structural + magpie,
}

def make_model():
    return RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)

# Repeated 5-fold CV (3 repeats) over all data gives a steadier estimate than one split
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

results = []
for name, cols in feature_sets.items():
    X = df[cols]
    scores = cross_validate(make_model(), X, y, cv=cv,
                            scoring=("r2", "neg_mean_absolute_error"))
    r2_mean, r2_std = scores["test_r2"].mean(), scores["test_r2"].std()
    mae_mean = -scores["test_neg_mean_absolute_error"].mean()

    # Held-out test with the same split as before (random_state=42, 20%)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    m = make_model().fit(X_tr, y_tr)
    pred = m.predict(X_te)

    results.append({
        "Feature set": name,
        "CV R2": f"{r2_mean:.3f} +/- {r2_std:.3f}",
        "CV MAE (eV)": f"{mae_mean:.3f}",
        "Test R2": f"{r2_score(y_te, pred):.3f}",
        "Test MAE (eV)": f"{mean_absolute_error(y_te, pred):.3f}",
    })

print(pd.DataFrame(results).to_string(index=False))

# Top features for the combined model, trained on all data
X_all = df[structural + magpie]
final = make_model().fit(X_all, y)
imp = pd.Series(final.feature_importances_, index=X_all.columns).sort_values(ascending=False)
print("\nTop 10 features (combined model):")
print(imp.head(10))

# Predicted vs actual for the combined model (held-out test set)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, random_state=42)
pred = make_model().fit(X_tr, y_tr).predict(X_te)
plt.figure(figsize=(6, 6))
plt.scatter(y_te, pred, s=60, alpha=0.7)
lims = [0, max(y_te.max(), pred.max()) + 0.3]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Band Gap (eV)")
plt.ylabel("Predicted Band Gap (eV)")
plt.title("Structural + Magpie: Predicted vs Actual")
plt.legend()
plt.tight_layout()
plt.savefig("../results/predicted_vs_actual_magpie.png", dpi=150)
plt.close()
print("\nSaved ../results/predicted_vs_actual_magpie.png")
