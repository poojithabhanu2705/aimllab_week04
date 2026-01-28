import numpy as np
import pandas as pd

np.random.seed(42)

n = 5000

L = np.random.uniform(3.0, 15.0, n)
P = np.random.uniform(0.0, 100.0, n)
a = np.random.uniform(0.05 * L, 0.95 * L)
EI = 1.0

delta = (P * a**2 * (3*L - a)) / (6 * EI)

df = pd.DataFrame({
    'L': L,
    'P': P,
    'a': a,
    'Deflection': delta
})

df.to_csv("cantilever_point_load.csv", index=False)

print("Dataset generated and saved.")
