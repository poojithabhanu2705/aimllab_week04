import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

L = np.random.uniform(3.0, 15.0, n)
P = np.random.uniform(0.0, 100.0, n)
a = np.random.uniform(0.05*L, 0.95*L)
EI = 1.0

delta = (P * a**2 * (L - a)**2) / (3 * EI * L)

df = pd.DataFrame({
    'L': L,
    'P': P,
    'a': a,
    'Deflection': delta
})

df.to_csv("problem3_ss_pointload.csv", index=False)
print("Problem 3 dataset generated.")
