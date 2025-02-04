import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Paths
data_folder = "data/"
batting_csv = os.path.join(data_folder, "processed_ipl_batting.csv")
bowling_csv = os.path.join(data_folder, "processed_ipl_bowling.csv")
model_folder = "models/"
os.makedirs(model_folder, exist_ok=True)

# Load Data
batting_df = pd.read_csv(batting_csv, dtype={'Season': str})
bowling_df = pd.read_csv(bowling_csv, dtype={'Season': str})

# Features & Targets
batting_features = ["Strike Rate", "Boundaries %", "Balls Faced"]
bowling_features = ["Economy Rate", "Bowling Average"]

X_batting = batting_df[batting_features]
y_batting = batting_df.groupby(["Season", "Batter"])["Total Runs"].transform("sum")

X_bowling = bowling_df[bowling_features]
y_bowling = bowling_df["Wickets"]

# Convert to numeric, replacing errors with NaN
X_batting = X_batting.apply(pd.to_numeric, errors='coerce')
X_bowling = X_bowling.apply(pd.to_numeric, errors='coerce')

# Replace infinite values with NaN
X_batting.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
X_bowling.replace([float("inf"), float("-inf")], float("nan"), inplace=True)

# Fill NaN values with zero
X_batting.fillna(0, inplace=True)
X_bowling.fillna(0, inplace=True)

# Train-Test Split
X_bat_train, X_bat_test, y_bat_train, y_bat_test = train_test_split(X_batting, y_batting, test_size=0.2, random_state=42)
X_bowl_train, X_bowl_test, y_bowl_train, y_bowl_test = train_test_split(X_bowling, y_bowling, test_size=0.2, random_state=42)

# ✅ Handle Class Imbalance with SMOTE for Bowling Model
smote = SMOTE(random_state=42)
X_bowl_train_resampled, y_bowl_train_resampled = smote.fit_resample(X_bowl_train, y_bowl_train)

# Train Batting Model (Regression)
batting_model = RandomForestRegressor(n_estimators=100, random_state=42)
batting_model.fit(X_bat_train, y_bat_train)

# Train Bowling Model (Classification) with Class Weights
bowling_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
bowling_model.fit(X_bowl_train_resampled, y_bowl_train_resampled)

# ✅ Evaluate Batting Model
y_bat_pred = batting_model.predict(X_bat_test)
bat_rmse = mean_squared_error(y_bat_test, y_bat_pred) ** 0.5  # Compute RMSE manually
bat_r2 = r2_score(y_bat_test, y_bat_pred)

print(f"Batting Model - RMSE: {bat_rmse:.4f}, R² Score: {bat_r2:.4f}")

# ✅ Evaluate Improved Bowling Model
y_bowl_pred = bowling_model.predict(X_bowl_test)
bowl_acc = accuracy_score(y_bowl_test, y_bowl_pred)

print(f"Improved Bowling Model - Accuracy: {bowl_acc:.4f}")
print("Improved Bowling Classification Report:\n", classification_report(y_bowl_test, y_bowl_pred))

# ✅ Save Models
joblib.dump(batting_model, os.path.join(model_folder, "batting_model.pkl"))
joblib.dump(bowling_model, os.path.join(model_folder, "bowling_model.pkl"))

print(f"Models saved in {model_folder}")
