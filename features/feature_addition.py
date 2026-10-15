from pathlib import Path
import pandas as pd
import numpy as np


def generate_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """Takes a raw IPL ball-by-ball dataframe and generates standardized

    cricket analytics features (phases, RAA, false shots, partnerships, etc.).
    """
    df = master_df.copy()
    features = {}

    # Base temporary tracking dataframe
    temp_df = pd.DataFrame(
        {
            "season": df.get("year"),
            "match_id": df.get("p_match"),
            "innings": df.get("inns"),
            "batsman_on_strike": df.get("bat"),
            "bowler": df.get("bowl"),
            "runs": df.get("score", 0),
        },
        index=df.index,
    )

    # Balls faced counter
    temp_df["balls_faced_counter"] = (
        df.groupby(["p_match", "inns", "bat"])["ballfaced"]
        .cumsum()
        .reset_index(drop=True)
    )

    batter_map = (
        df[["p_bat", "bat"]]
        .drop_duplicates("p_bat")
        .set_index("p_bat")["bat"]
    )

    dismissals = [
        "bowled",
        "caught",
        "leg before wicket",
        "stumped",
        "retired not out (hurt)",
        "run out",
        "hit wicket",
        "retired out",
        "obstructing the field",
    ]

    features["balls_faced_per_innings"] = temp_df.groupby(
        ["match_id", "innings", "batsman_on_strike"]
    )["balls_faced_counter"].transform("max")

    features["batsman_dismissed"] = np.where(
        df["dismissal"].isin(dismissals), df["p_out"].map(batter_map), np.nan
    )

    features["wicket_counter"] = np.where(
        df["dismissal"].isin(
            ["bowled", "caught", "leg before wicket", "stumped", "hit wicket"]
        ),
        1,
        0,
    )

    # False shot metrics (with safeguard if 'control' column is missing)
    if "control" in df.columns:
        features["false_shot"] = 1 - df["control"]
        features["false_shot_wkts"] = np.where(
            (df["control"] == 0) & (df["out"] == 1), 1, 0
        )
    else:
        features["false_shot"] = np.nan
        features["false_shot_wkts"] = 0

    features["4s"] = np.where(df["batruns"] == 4, 1, 0)
    features["6s"] = np.where(df["batruns"] == 6, 1, 0)
    features["boundaries"] = np.where(df["batruns"].isin([4, 6]), 1, 0)

    # Specific dismissal counters
    for wicket in dismissals:
        col_name = f'batsman_{wicket.lower().replace(" ", "_")}'
        features[f"{col_name}_counter"] = np.where(
            df["dismissal"] == wicket, 1, 0
        )
        temp_df[f"{col_name}_counter"] = features[f"{col_name}_counter"]
        features[f"{col_name}_total"] = temp_df.groupby(
            ["match_id", "innings", "batsman_on_strike"]
        )[f"{col_name}_counter"].cumsum()

    # Match overs handling
    max_balls = df.get("max_balls", pd.Series([120] * len(df)))
    ball_id = df.get("ball_id", pd.Series([0] * len(df)))

    features["powerplay_overs"] = np.where(
        (
            (~max_balls.isin([90, 96]) & ball_id.between(0, 6))
            | ((max_balls == 90) & ball_id.between(0, 4))
            | ((max_balls == 96) & ball_id.between(0, 5))
        ),
        1,
        0,
    )

    features["middle_overs"] = np.where(
        (
            (~max_balls.isin([90, 96]) & ball_id.between(6, 15))
            | ((max_balls == 90) & ball_id.between(4, 10))
            | ((max_balls == 96) & ball_id.between(5, 11))
        ),
        1,
        0,
    )

    features["death_overs"] = np.where(
        (
            (~max_balls.isin([90, 96]) & ball_id.between(15, 20))
            | ((max_balls == 90) & ball_id.between(10, 15))
            | ((max_balls == 96) & ball_id.between(11, 16))
        ),
        1,
        0,
    )

    features["phase"] = np.where(
        features["powerplay_overs"] == 1,
        "Powerplay phase",
        np.where(features["middle_overs"] == 1, "Middle phase", "Death phase"),
    )

    features["is_legal"] = np.where(
        (df.get("wide", 0) == 0) & (df.get("noball", 0) == 0), 1, 0
    )

    features["partnership_id"] = (
        df.groupby(["p_match", "inns"])["out"].cumsum() + 1
    )
    temp_df["partnership_id"] = features["partnership_id"]
    features["partnership_runs"] = temp_df.groupby(
        ["match_id", "innings", "partnership_id"]
    )["runs"].cumsum()

    # Runs Above Average (RAA) per ball
    if (
        "score" in master_df.columns
        and "predscore" in master_df.columns
        and "max_balls" in master_df.columns
    ):
        features["raa_per_ball"] = master_df["score"] - (
            master_df["predscore"] / master_df["max_balls"]
        )
    else:
        features["raa_per_ball"] = 0.0

    temp_df["team_runs"] = (
        temp_df.groupby(["match_id", "innings"])["runs"]
        .cumsum()
        .reset_index(drop=True)
    )
    features["team_runs"] = temp_df["team_runs"]

    features["batter_entry_runs"] = temp_df.groupby(
        ["match_id", "innings", "batsman_on_strike"]
    )["team_runs"].transform("first")
    features["batter_exit_runs"] = temp_df.groupby(
        ["match_id", "innings", "batsman_on_strike"]
    )["team_runs"].transform("last")

    if "inns_rr" in master_df.columns:
        features["batter_entry_rr"] = master_df.groupby(
            ["p_match", "inns", "bat"]
        )["inns_rr"].transform("first")
        features["batter_exit_rr"] = master_df.groupby(
            ["p_match", "inns", "bat"]
        )["inns_rr"].transform("last")

    conditions = [
        master_df['out'] == 1,
        master_df['batruns'] == 6,
        master_df['batruns'] == 4,
        master_df['batruns'].isin([1, 2, 3]),
        master_df['batruns'] >= 5,
        master_df['batruns'] == 0
    ]

    # Corresponding clean names for each condition
    choices = [
        'Wicket',
        'Boundary (6)',
        'Boundary (4)',
        'Running Runs (1-3)',
        'Unconventional Runs (5+)',
        'Dot Ball'
    ]

    # Apply using np.select with a fallback default value for any edge cases missed
    features['outcomes'] = np.select(
        conditions, 
        choices, 
        default='Other / Unclassified'
    )
    features_df = pd.DataFrame(features, index=df.index)
    
    
    return pd.concat([df, features_df], axis=1)


if __name__ == "__main__":
    # Robust relative path execution
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "raw" / "master_df.csv"
    OUTPUT_PATH = BASE_DIR / "data" / "processed" / "master_features.csv"

    print(f"Loading data from {DATA_PATH}...")
    master_df = pd.read_csv(DATA_PATH)

    print("Generating features...")
    master_data = generate_features(master_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    master_data.to_csv(OUTPUT_PATH, index=False)
    print(f"Features successfully saved to {OUTPUT_PATH}")