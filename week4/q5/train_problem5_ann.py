import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

df = pd.read_csv("problem5_propped_pointload.csv")

X = df[['L', 'P', 'a']]
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

joblib.dump(model, "problem5_model.pkl")
joblib.dump(scaler, "problem5_scaler.pkl")

print("Problem 5 model trained and saved.")
