import shap
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

# Paths
data_folder = "data/"
model_folder = "models/"
batting_model_path = os.path.join(model_folder, "batting_model.pkl")
bowling_model_path = os.path.join(model_folder, "bowling_model.pkl")
batting_csv = os.path.join(data_folder, "processed_ipl_batting.csv")
bowling_csv = os.path.join(data_folder, "processed_ipl_bowling.csv")

# Load Models
batting_model = joblib.load(batting_model_path)
bowling_model = joblib.load(bowling_model_path)

# Load Data
batting_df = pd.read_csv(batting_csv)
bowling_df = pd.read_csv(bowling_csv)

# Select Features
batting_features = ["Balls Faced", "Strike Rate", "Boundaries %"]
bowling_features = ["Economy Rate", "Bowling Average"]

X_batting = batting_df[batting_features]
X_bowling = bowling_df[bowling_features]

# Create SHAP Explainers
batting_explainer = shap.TreeExplainer(batting_model)    
bowling_explainer = shap.TreeExplainer(bowling_model)

# Compute SHAP Values

# batting_shap_values = batting_explainer.shap_values(X_batting, check_additivity=False)
# bowling_shap_values = bowling_explainer.shap_values(X_bowling, check_additivity=False)

X_batting_sample = X_batting.sample(n=1000, random_state=42)  # Select 1000 random rows
batting_shap_values = batting_explainer.shap_values(X_batting_sample, check_additivity=False)
X_bowling_sample = X_bowling.sample(n=1000, random_state=42)
bowling_shap_values = bowling_explainer.shap_values(X_bowling_sample, check_additivity=False)


# Plot SHAP Summary for Batting Model
shap.summary_plot(batting_shap_values, X_batting_sample, show=False)
plt.savefig("data/batting_shap_summary.png")
plt.clf()

# Plot SHAP Summary for Bowling Model
X_bowling = X_bowling.to_numpy()  # Convert to NumPy array
shap.summary_plot(bowling_shap_values, X_bowling_sample, show=False)
plt.savefig("data/bowling_shap_summary.png")
plt.clf()

print("SHAP analysis completed! Plots saved in 'data/' folder.")
