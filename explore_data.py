import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
data_folder = "data/"
batting_csv = os.path.join(data_folder, "ipl_batting_stats.csv")
bowling_csv = os.path.join(data_folder, "ipl_bowling_stats.csv")

def load_data():
    """Load batting and bowling datasets."""
    batting_df = pd.read_csv(batting_csv)
    bowling_df = pd.read_csv(bowling_csv)
    return batting_df, bowling_df

def check_missing_values(df, name):
    """Print missing values for a given DataFrame."""
    print(f"\nMissing values in {name} Data:")
    print(df.isnull().sum())

def summarize_data(df, name):
    """Print summary statistics for a given DataFrame."""
    print(f"\n{name} Stats Summary:")
    print(df.describe())

def plot_distribution(df, column, title, xlabel, filename):
    """Plot and save distribution of a numerical column."""
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], bins=20, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(data_folder, filename))
    plt.show()

if __name__ == "__main__":
    # Load data
    batting_df, bowling_df = load_data()

    # Check for missing values
    check_missing_values(batting_df, "Batting")
    check_missing_values(bowling_df, "Bowling")

    # Summarize data
    summarize_data(batting_df, "Batting")
    summarize_data(bowling_df, "Bowling")

    # Visualizations
    plot_distribution(batting_df, "Runs", "Distribution of Runs Scored by Batters", "Runs", "batting_runs_distribution.png")
    plot_distribution(bowling_df, "Wickets", "Distribution of Wickets Taken by Bowlers", "Wickets", "bowling_wickets_distribution.png")

    print("Data exploration completed!")
