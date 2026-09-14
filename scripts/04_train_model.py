import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("features.csv")

feature_cols = [
    "lattice_a", "lattice_b", "lattice_c", "cell_volume",
    "mean_atomic_number", "electronegativity_diff",
    "mean_electronegativity", "mean_atomic_radius", "num_elements"
]

X = df[feature_cols]
y = df["band_gap"]

# With only ~19 samples, Leave-One-Out CV is used instead of a train/test split —
# a single split would leave only 3-4 test points, which isn't statistically meaningful.
loo = LeaveOneOut()
model = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)

y_true, y_pred = [], []
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    y_true.append(y_test.values[0])
    y_pred.append(pred[0])

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("Leave-One-Out Cross-Validation Results:")
print(f"MAE:  {mae:.3f} eV")
print(f"RMSE: {rmse:.3f} eV")
print(f"R2:   {r2:.3f}")

# Fit final model on all data for feature importance
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

# Predicted vs Actual plot
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, s=80)
lims = [min(y_true+y_pred)-0.2, max(y_true+y_pred)+0.2]
plt.plot(lims, lims, 'r--', label="Perfect prediction")
plt.xlabel("Actual Band Gap (eV)")
plt.ylabel("Predicted Band Gap (eV)")
plt.title(f"Predicted vs Actual Band Gap (MAE = {mae:.3f} eV)")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=150)
plt.close()

print("\nSaved predicted_vs_actual.png")
