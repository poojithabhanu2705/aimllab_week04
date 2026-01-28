import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

df = pd.read_csv("problem6_propped_udl.csv")

X = df[['L', 'w']]
y = df['Deflection']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = MLPRegressor(
    hidden_layer_sizes=(32, 32),
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "problem6_model.pkl")
joblib.dump(scaler, "problem6_scaler.pkl")

print("Problem 6 model trained and saved.")
