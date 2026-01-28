import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

model = joblib.load("problem5_model.pkl")
scaler = joblib.load("problem5_scaler.pkl")

df_test = pd.DataFrame({
    'L': [6, 8, 10, 12],
    'P': [20, 40, 60, 80],
    'a': [2, 3, 4, 5]
})

X_test = scaler.transform(df_test)
ann_pred = model.predict(X_test)

EI = 1.0
L = df_test['L'].values
P = df_test['P'].values
a = df_test['a'].values

theory = (P * a**2 * (L - a) * (3*L - a)) / (6 * EI * (3*L**2 - 4*L*a + a**2))

print("MAE:", mean_absolute_error(theory, ann_pred))
print("MSE:", mean_squared_error(theory, ann_pred))
print("R2 :", r2_score(theory, ann_pred))

plt.figure()
plt.scatter(theory, ann_pred)
plt.plot(theory, theory)
plt.title("Problem 5: ANN vs Theory")
plt.show()
