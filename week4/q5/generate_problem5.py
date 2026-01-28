import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

L = np.random.uniform(3.0, 15.0, n)
P = np.random.uniform(0.0, 100.0, n)
a = np.random.uniform(0.05*L, 0.95*L)
EI = 1.0

delta = (P * a**2 * (L - a) * (3*L - a)) / (6 * EI * (3*L**2 - 4*L*a + a**2))

df = pd.DataFrame({
    'L': L,
    'P': P,
    'a': a,
    'Deflection': delta
})

df.to_csv("problem5_propped_pointload.csv", index=False)
print("Problem 5 dataset generated.")
