import os
import joblib
import pandas as pd

# Paths
model_folder = "models/"
batting_model_path = os.path.join(model_folder, "batting_model.pkl")
bowling_model_path = os.path.join(model_folder, "bowling_model.pkl")

# Load Models
batting_model = joblib.load(batting_model_path)
bowling_model = joblib.load(bowling_model_path)

def predict_performance(batting_stats, bowling_stats):
    """Predict Runs for Batters & Wickets for Bowlers."""
    
    # Convert input to DataFrame format
    batting_df = pd.DataFrame([batting_stats], columns=batting_model.feature_names_in_)
    bowling_df = pd.DataFrame([bowling_stats], columns=bowling_model.feature_names_in_)


    # Make Predictions
    predicted_runs = batting_model.predict(batting_df)[0]
    predicted_wickets = bowling_model.predict(bowling_df)[0]

    return predicted_runs, predicted_wickets

if __name__ == "__main__":
    # Sample Input Data
    sample_batting_stats = {"Strike Rate": 140.0, "Boundaries %": 0.35, "Balls Faced": 30}  # Example value  # Example batter stats
    sample_bowling_stats = {"Economy Rate": 7.5, "Bowling Average": 25.0}  # Example bowler stats

    runs, wickets = predict_performance(sample_batting_stats, sample_bowling_stats)

    print(f"🏏 Predicted Runs: {runs}")
    print(f"⚡ Predicted Wickets: {wickets}")
