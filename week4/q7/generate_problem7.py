import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

L = np.random.uniform(3.0, 15.0, n)
P = np.random.uniform(0.0, 100.0, n)
EI = 1.0

delta = (P * L**3) / (192 * EI)

df = pd.DataFrame({
    'L': L,
    'P': P,
    'Deflection': delta
})

df.to_csv("problem7_fixed_pointload.csv", index=False)
print("Problem 7 dataset generated.")
