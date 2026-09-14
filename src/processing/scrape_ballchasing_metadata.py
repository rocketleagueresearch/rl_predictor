import os
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup


INPUT_FILE = r"data\processed\ballchasing\open1_games.csv"
OUTPUT_FILE = r"data\processed\ballchasing\open1_replay_metadata.csv"

BASE_URL = "https://ballchasing.com/replay/{}"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

REQUEST_DELAY = 5


def extract_replay_datetime(soup):
    """
    Extract the actual replay/game datetime from Ballchasing.

    Ballchasing uses two known formats:

    1. A local time without timezone:
       2025-01-10 15:30

    2. A UTC time followed by the local replay time:
       2025-01-10 18:22 UTC (2025-01-10 12:22 -06)

    We want the local replay time.
    """

    # Format 1
    element = soup.find(
        "span",
        attrs={
            "title": "When the game took place "
                     "(this date does not have a timezone)"
        }
    )

    if element:
        text = element.get_text(" ", strip=True)

        match = re.search(
            r"\b(20\d{2}-\d{2}-\d{2} \d{2}:\d{2})\b",
            text
        )

        if match:
            return match.group(1)

    # Format 2
    element = soup.find(
        "span",
        attrs={"title": "When the game took place"}
    )

    if element:
        text = element.get_text(" ", strip=True)

        # Prefer the local time inside parentheses.
        match = re.search(
            r"\(\s*(20\d{2}-\d{2}-\d{2} \d{2}:\d{2})\s*[-+]\d{2}\s*\)",
            text
        )

        if match:
            return match.group(1)

        # Fallback: first datetime in the element.
        match = re.search(
            r"\b(20\d{2}-\d{2}-\d{2} \d{2}:\d{2})\b",
            text
        )

        if match:
            return match.group(1)

    return None


def extract_matchguid(soup):
    """
    Extract Ballchasing MatchGUID.
    """

    text = soup.get_text(" ", strip=True)

    match = re.search(
        r"MatchGUID:\s*([A-Fa-f0-9]{32})",
        text
    )

    if match:
        return match.group(1)

    return None


def scrape_replay(replay_id):
    """
    Scrape metadata for one replay.
    """

    url = BASE_URL.format(replay_id)

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        status_code = response.status_code

        if status_code == 429:
            return {
                "replay_id": replay_id,
                "replay_datetime": None,
                "matchguid": None,
                "http_status": status_code,
                "status": "rate_limited"
            }

        if status_code != 200:
            return {
                "replay_id": replay_id,
                "replay_datetime": None,
                "matchguid": None,
                "http_status": status_code,
                "status": "http_error"
            }

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        replay_datetime = extract_replay_datetime(soup)
        matchguid = extract_matchguid(soup)

        if replay_datetime:
            status = "success"
        else:
            status = "date_not_found"

        return {
            "replay_id": replay_id,
            "replay_datetime": replay_datetime,
            "matchguid": matchguid,
            "http_status": status_code,
            "status": status
        }

    except Exception as e:
        return {
            "replay_id": replay_id,
            "replay_datetime": None,
            "matchguid": None,
            "http_status": None,
            "status": f"error: {e}"
        }


def main():

    print("Loading Open 1 game data...")

    games = pd.read_csv(INPUT_FILE)

    replay_ids = (
        games["replay_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(f"Replay IDs found: {len(replay_ids)}")
    print()

    # Load existing results if available.
    if os.path.exists(OUTPUT_FILE):

        existing = pd.read_csv(OUTPUT_FILE)

        print(
            f"Existing metadata rows: {len(existing)}"
        )

    else:

        existing = pd.DataFrame(
            columns=[
                "replay_id",
                "replay_datetime",
                "matchguid",
                "http_status",
                "status"
            ]
        )

    # Only skip rows where a replay date was successfully found.
    successful_ids = set(
        existing.loc[
            existing["replay_datetime"].notna(),
            "replay_id"
        ].astype(str)
    )

    print(
        f"Already successful: {len(successful_ids)}"
    )

    remaining_ids = [
        replay_id
        for replay_id in replay_ids
        if replay_id not in successful_ids
    ]

    print(
        f"Remaining to scrape: {len(remaining_ids)}"
    )
    print()

    results = existing.to_dict("records")

    for index, replay_id in enumerate(
        remaining_ids,
        start=1
    ):

        print(
            f"[{index}/{len(remaining_ids)}] "
            f"{replay_id}"
        )

        result = scrape_replay(replay_id)

        print(
            f"  status={result['status']} "
            f"date={result['replay_datetime']}"
        )

        results.append(result)

        # Save after every replay.
        pd.DataFrame(results).drop_duplicates(
            subset=["replay_id"],
            keep="last"
        ).to_csv(
            OUTPUT_FILE,
            index=False
        )

        # Handle rate limiting.
        if result["status"] == "rate_limited":

            print(
                "  Rate limited. Waiting 60 seconds..."
            )

            time.sleep(60)

        else:

            time.sleep(REQUEST_DELAY)

    final = pd.DataFrame(results).drop_duplicates(
        subset=["replay_id"],
        keep="last"
    )

    final.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Finished.")
    print()
    print(
        "Successful:",
        final["replay_datetime"].notna().sum()
    )
    print(
        "Date not found:",
        (
            final["status"] == "date_not_found"
        ).sum()
    )
    print(
        "Other failures:",
        (
            ~final["status"].isin(
                ["success", "date_not_found"]
            )
        ).sum()
    )
    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()