import os
import json
import pandas as pd

# Path to JSON files folder
json_folder = "ipl_json/"  # Update this if needed

# Lists to store extracted data
batting_data = []
bowling_data = []

# Process each JSON file in the folder
for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(json_folder, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            match_data = json.load(f)
            
            # Extract match info
            match_info = match_data.get("info", {})
            venue = match_info.get("venue", "Unknown")
            season = match_info.get("season", "Unknown")
            teams = match_info.get("teams", [])
            toss_winner = match_info.get("toss", {}).get("winner", "Unknown")
            toss_decision = match_info.get("toss", {}).get("decision", "Unknown")
            winner = match_info.get("outcome", {}).get("winner", "Unknown")
            
            # Process innings
            for inning in match_data.get("innings", []):
                batting_team = inning.get("team", "Unknown")
                for over in inning.get("overs", []):
                    for delivery in over.get("deliveries", []):
                        batter = delivery.get("batter", "Unknown")
                        bowler = delivery.get("bowler", "Unknown")
                        runs = delivery.get("runs", {}).get("batter", 0)
                        extras = delivery.get("runs", {}).get("extras", 0)
                        total_runs = delivery.get("runs", {}).get("total", 0)
                        
                        # Track batting stats
                        batting_data.append([
                            season, venue, batting_team, batter, runs, extras, total_runs,
                            toss_winner, toss_decision, winner
                        ])
                        
                        # Track bowling stats
                        wickets = 1 if "wickets" in delivery else 0
                        bowling_data.append([
                            season, venue, batting_team, bowler, total_runs, wickets,
                            toss_winner, toss_decision, winner
                        ])

# Create DataFrames
batting_df = pd.DataFrame(batting_data, columns=[
    "Season", "Venue", "Batting Team", "Batter", "Runs", "Extras", "Total Runs",
    "Toss Winner", "Toss Decision", "Match Winner"
])

bowling_df = pd.DataFrame(bowling_data, columns=[
    "Season", "Venue", "Batting Team", "Bowler", "Runs Conceded", "Wickets",
    "Toss Winner", "Toss Decision", "Match Winner"
])

# Save to CSV
batting_df.to_csv("data/ipl_batting_stats.csv", index=False)
bowling_df.to_csv("data/ipl_bowling_stats.csv", index=False)

print("Data extraction completed! CSV files saved.")