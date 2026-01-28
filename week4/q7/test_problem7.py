import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

model = joblib.load("problem7_model.pkl")
scaler = joblib.load("problem7_scaler.pkl")

df_test = pd.DataFrame({
    'L': [6, 8, 10, 12],
    'P': [20, 40, 60, 80]
})

X_test = scaler.transform(df_test)
ann_pred = model.predict(X_test)

EI = 1.0
L = df_test['L'].values
P = df_test['P'].values

theory = (P * L**3) / (192 * EI)

print("MAE:", mean_absolute_error(theory, ann_pred))
print("MSE:", mean_squared_error(theory, ann_pred))
print("R2 :", r2_score(theory, ann_pred))

plt.figure()
plt.scatter(theory, ann_pred)
plt.plot(theory, theory)
plt.title("Problem 7: ANN vs Theory")
plt.show()
