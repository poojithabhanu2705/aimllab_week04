import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Load data
df = pd.read_csv("cantilever_point_load.csv")

X = df[['L', 'P', 'a']]
y = df['Deflection']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ANN model (MLP)
model = MLPRegressor(
    hidden_layer_sizes=(32, 32),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "cantilever_ann_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("ANN trained and saved.")
