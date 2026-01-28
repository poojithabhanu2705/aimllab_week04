import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error

model = joblib.load("problem8_model.pkl")
scaler = joblib.load("problem8_scaler.pkl")

df_test = pd.DataFrame({
    'L': [6, 8, 10, 12],
    'w': [5, 10, 20, 30]
})

X_test = scaler.transform(df_test)
ann_pred = model.predict(X_test)

EI = 1.0
L = df_test['L'].values
w = df_test['w'].values

theory = (w * L**4) / (384 * EI)

print("MAE:", mean_absolute_error(theory, ann_pred))
print("MSE:", mean_squared_error(theory, ann_pred))
print("R2 :", r2_score(theory, ann_pred))

plt.figure()
plt.scatter(theory, ann_pred)
plt.plot(theory, theory)
plt.title("Problem 8: ANN vs Theory")
plt.show()
