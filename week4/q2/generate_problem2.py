import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

L = np.random.uniform(3.0, 15.0, n)
w = np.random.uniform(0.0, 50.0, n)
EI = 1.0

delta = (w * L**4) / (8 * EI)

df = pd.DataFrame({
    'L': L,
    'w': w,
    'Deflection': delta
})

df.to_csv("problem2_cantilever_udl.csv", index=False)
print("Problem 2 dataset generated.")
