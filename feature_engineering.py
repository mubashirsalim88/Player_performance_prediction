import os
import pandas as pd

# Paths
data_folder = "data/"
batting_csv = os.path.join(data_folder, "ipl_batting_stats.csv")
bowling_csv = os.path.join(data_folder, "ipl_bowling_stats.csv")
processed_batting_csv = os.path.join(data_folder, "processed_ipl_batting.csv")
processed_bowling_csv = os.path.join(data_folder, "processed_ipl_bowling.csv")

def load_data():
    """Load batting and bowling datasets with correct dtype."""
    batting_df = pd.read_csv(batting_csv, dtype={'Season': str})
    bowling_df = pd.read_csv(bowling_csv, dtype={'Season': str})
    return batting_df.copy(), bowling_df.copy()  # Ensure independent copies

def add_batting_features(df):
    df["Balls Faced"] = df.groupby(["Season", "Batter"])["Runs"].transform("count")  # Count balls faced
    df["Strike Rate"] = (df["Runs"] / df["Balls Faced"]) * 100  # Fix calculation
    df["Strike Rate"] = df["Strike Rate"].fillna(0)
    df["Boundaries %"] = ((df["Runs"] >= 4).astype(int)) / df["Balls Faced"]  # Fix dependency
    df["Boundaries %"] = df["Boundaries %"].fillna(0)
    return df

def add_bowling_features(df):
    """Add bowling-related derived features."""
    df["Economy Rate"] = df["Runs Conceded"] / 6  # Assuming over = 6 balls

    df["Bowling Average"] = df["Runs Conceded"] / df["Wickets"]
    df["Bowling Average"] = df["Bowling Average"].fillna(0)  # Fix for Pandas 3.0

    return df

if __name__ == "__main__":
    # Load data
    batting_df, bowling_df = load_data()

    # Feature engineering
    batting_df = add_batting_features(batting_df)
    bowling_df = add_bowling_features(bowling_df)

    # Save processed data
    batting_df.to_csv(processed_batting_csv, index=False)
    bowling_df.to_csv(processed_bowling_csv, index=False)

    print(f"Feature Engineering Completed! Processed data saved in {data_folder}")
