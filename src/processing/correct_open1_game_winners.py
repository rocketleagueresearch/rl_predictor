import pandas as pd
import numpy as np

INPUT = "data/processed/ballchasing/open1_games.csv"
OUTPUT = "data/processed/ballchasing/open1_games_corrected.csv"

df = pd.read_csv(INPUT)

def calculate_winner(row):
    blue_goals = row["blue_goals"]
    orange_goals = row["orange_goals"]

    if pd.isna(blue_goals) or pd.isna(orange_goals):
        return np.nan

    if blue_goals > orange_goals:
        return row["blue_team"]

    if orange_goals > blue_goals:
        return row["orange_team"]

    return np.nan


# Preserve the original field for auditability.
df["game_winner_original"] = df["game_winner"]

# Recalculate winner from actual match goals.
df["game_winner"] = df.apply(calculate_winner, axis=1)

# Check how many values changed.
changed = (
    df["game_winner_original"].fillna("__MISSING__")
    != df["game_winner"].fillna("__MISSING__")
)

print("Correcting Open 1 game winners")
print("===============================")
print()
print("Games:", len(df))
print("Original winners:", df["game_winner_original"].notna().sum())
print("Corrected winners:", df["game_winner"].notna().sum())
print("Winners changed:", changed.sum())

print()
print("Example corrections")
print("--------------------")

corrections = df[changed]

for _, row in corrections.head(10).iterrows():
    print(
        row["replay_id"],
        "|",
        row["blue_team"],
        f"{int(row['blue_goals'])}-{int(row['orange_goals'])}",
        row["orange_team"],
        "|",
        "old:",
        row["game_winner_original"],
        "| new:",
        row["game_winner"]
    )

df.to_csv(OUTPUT, index=False)

print()
print("Saved to:")
print(OUTPUT)
print()
print("Finished.")