import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load("cantilever_ann_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Unseen test data
# -----------------------------
test_data = pd.DataFrame({
    'L': [6, 8, 10, 12],
    'P': [20, 40, 60, 80],
    'a': [2, 3, 4, 5]
})

X_test = scaler.transform(test_data)

# -----------------------------
# ANN Prediction
# -----------------------------
ann_pred = model.predict(X_test)

# -----------------------------
# Theory calculation
# -----------------------------
EI = 1.0
L = test_data['L'].values
P = test_data['P'].values
a = test_data['a'].values

theory = (P * a**2 * (3*L - a)) / (6 * EI)

# -----------------------------
# Metrics
# -----------------------------
print("MAE:", mean_absolute_error(theory, ann_pred))
print("MSE:", mean_squared_error(theory, ann_pred))
print("R2 :", r2_score(theory, ann_pred))

# -----------------------------
# Comparison table
# -----------------------------
comparison = pd.DataFrame({
    'L': L,
    'P': P,
    'a': a,
    'Theory_Deflection': theory,
    'ANN_Deflection': ann_pred,
    'Error': abs(theory - ann_pred)
})

print(comparison)

# -----------------------------
# Plot
# -----------------------------
plt.figure()
plt.scatter(theory, ann_pred)
plt.plot(theory, theory)
plt.xlabel("Theoretical Deflection")
plt.ylabel("ANN Predicted Deflection")
plt.title("Problem 1: ANN vs Theory")
plt.show()
