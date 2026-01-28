import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

L = np.random.uniform(3.0, 15.0, n)
w = np.random.uniform(0.0, 50.0, n)
EI = 1.0

delta = (5 * w * L**4) / (384 * EI)

df = pd.DataFrame({
    'L': L,
    'w': w,
    'Deflection': delta
})

df.to_csv("problem4_ss_udl.csv", index=False)
print("Problem 4 dataset generated.")
