import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("features_large.csv")

feature_cols = [
    "lattice_a", "lattice_b", "lattice_c", "cell_volume",
    "mean_atomic_number", "electronegativity_diff",
    "mean_electronegativity", "mean_atomic_radius", "num_elements"
]

X = df[feature_cols]
y = df["band_gap"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)

# 5-fold cross-validation on the training set, to get a stable performance estimate
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")
print(f"5-Fold CV MAE: {-cv_scores.mean():.3f} +/- {cv_scores.std():.3f} eV")

# Fit on full training set, evaluate on held-out test set
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nHeld-out Test Set Results:")
print(f"MAE:  {mae:.3f} eV")
print(f"RMSE: {rmse:.3f} eV")
print(f"R2:   {r2:.3f}")

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, s=60, alpha=0.7)
lims = [0, max(y_test.max(), y_pred.max()) + 0.3]
plt.plot(lims, lims, 'r--', label="Perfect prediction")
plt.xlabel("Actual Band Gap (eV)")
plt.ylabel("Predicted Band Gap (eV)")
plt.title(f"Predicted vs Actual Band Gap (Test MAE = {mae:.3f} eV)")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual_large.png", dpi=150)
plt.close()

print("\nSaved predicted_vs_actual_large.png")
