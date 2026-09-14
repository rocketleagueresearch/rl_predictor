import pandas as pd
from pathlib import Path


OUTPUT_FILE = "data/processed/metadata/replay_metadata.csv"


def create_test_metadata():

    series_id = "100-vs-sng-ott01kdne5"

    rows = [
        {
            "replay_id": "b0bd4045-9f1a-404e-801a-dbde07155c75",
            "series_id": series_id,
            "game_number": 1,
            "tournament": "RLCS 2025 EU Open 1",
            "stage": "Group Stage",
            "date": "2025-01-10 16:11:04",
            "team_a": "100%",
            "team_b": "SYNERGY",
            "game_team_a_score": 2,
            "game_team_b_score": 1,
            "series_team_a_score": 3,
            "series_team_b_score": 2,
            "series_winner": "100%",
            "source": "Ballchasing",
        },
        {
            "replay_id": "A7B0FA4211EFCF6D0B51D48C76F184E1",
            "series_id": series_id,
            "game_number": 2,
            "tournament": "RLCS 2025 EU Open 1",
            "stage": "Group Stage",
            "date": "2025-01-10 16:18:55",
            "team_a": "100%",
            "team_b": "SYNERGY",
            "game_team_a_score": 1,
            "game_team_b_score": 3,
            "series_team_a_score": 3,
            "series_team_b_score": 2,
            "series_winner": "100%",
            "source": "Ballchasing",
        },
        {
            "replay_id": "A6BFE7FA11EFCF6E7551D48C5FFA4FCB",
            "series_id": series_id,
            "game_number": 3,
            "tournament": "RLCS 2025 EU Open 1",
            "stage": "Group Stage",
            "date": "2025-01-10 16:25:42",
            "team_a": "100%",
            "team_b": "SYNERGY",
            "game_team_a_score": 1,
            "game_team_b_score": 2,
            "series_team_a_score": 3,
            "series_team_b_score": 2,
            "series_winner": "100%",
            "source": "Ballchasing",
        },
        {
            "replay_id": "B329A2DC11EFCF6F87EFD48C777CAF14",
            "series_id": series_id,
            "game_number": 4,
            "tournament": "RLCS 2025 EU Open 1",
            "stage": "Group Stage",
            "date": "2025-01-10 16:33:15",
            "team_a": "100%",
            "team_b": "SYNERGY",
            "game_team_a_score": 3,
            "game_team_b_score": 0,
            "series_team_a_score": 3,
            "series_team_b_score": 2,
            "series_winner": "100%",
            "source": "Ballchasing",
        },
        {
            "replay_id": "0B56D5A011EFCF717590D58D9CC09E51",
            "series_id": series_id,
            "game_number": 5,
            "tournament": "RLCS 2025 EU Open 1",
            "stage": "Group Stage",
            "date": "2025-01-10 16:44:32",
            "team_a": "100%",
            "team_b": "SYNERGY",
            "game_team_a_score": 1,
            "game_team_b_score": 3,
            "series_team_a_score": 3,
            "series_team_b_score": 2,
            "series_winner": "100%",
            "source": "Ballchasing",
        },
    ]

    return pd.DataFrame(rows)


if __name__ == "__main__":

    print("=" * 60)
    print("POPULATING TEST REPLAY METADATA")
    print("=" * 60)

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = create_test_metadata()

    df.to_csv(output_path, index=False)

    print("\nMetadata:")
    print(df.to_string(index=False))

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("TEST METADATA COMPLETE")
    print("=" * 60)
