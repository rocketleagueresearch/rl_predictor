from pathlib import Path
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ballchasing"
)

OPEN_CONFIG = {
    "Open 2": {
        "games_file": PROCESSED_DIR / "open2_games.csv",
        "output_file": PROCESSED_DIR / "open2_replay_metadata.csv",
    },
    "Open 3": {
        "games_file": PROCESSED_DIR / "open3_games.csv",
        "output_file": PROCESSED_DIR / "open3_replay_metadata.csv",
    },
}


# ============================================================
# SCRAPER SETTINGS
# ============================================================

BASE_URL = "https://ballchasing.com/replay/{}"

REQUEST_DELAY_SECONDS = 1.25

MAX_RETRIES = 5

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-GB,en;q=0.9",
}

# For the Ballchasing pages that explicitly say:
# "this date does not have a timezone"
#
# In our Open 1 work, those displayed replay times corresponded
# to the -06 local clock shown by Ballchasing on other replay pages.
# January/February 2025 is UTC-6, so we convert that displayed
# replay clock to UTC using a fixed -06 offset.
#
# This does NOT use Liquipedia or any external schedule.
NAIVE_DISPLAY_OFFSET_HOURS = -6


# ============================================================
# REGEX
# ============================================================

UTC_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s+UTC",
    re.IGNORECASE,
)

LOCAL_WITH_OFFSET_PATTERN = re.compile(
    r"\((\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s+([+-]\d{2})\)"
)

PLAIN_DATETIME_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})"
)

MATCH_GUID_PATTERN = re.compile(
    r"MatchGUID\s*[:\s]\s*([A-Fa-f0-9]{16,64})"
)

SERVER_PATTERN = re.compile(
    r"Server\s*[:\s]\s*([A-Za-z0-9._-]+)"
)


# ============================================================
# HELPERS
# ============================================================

def divider(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_existing(output_file):
    """
    Resume safely from an existing output file.
    """
    if not output_file.exists():
        return pd.DataFrame()

    try:
        existing = pd.read_csv(output_file)
    except Exception:
        return pd.DataFrame()

    if "replay_id" not in existing.columns:
        return pd.DataFrame()

    return existing


def save_metadata(rows, output_file):
    """
    Save all current rows in a stable column order.
    """
    df = pd.DataFrame(rows)

    preferred_columns = [
        "replay_id",
        "url",
        "http_status",
        "scrape_success",
        "replay_date_raw",
        "replay_timestamp_utc",
        "timestamp_method",
        "match_guid",
        "server",
        "page_title",
        "error",
    ]

    for col in preferred_columns:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[preferred_columns]

    df = (
        df
        .drop_duplicates(
            subset=["replay_id"],
            keep="last",
        )
        .sort_values("replay_id")
        .reset_index(drop=True)
    )

    df.to_csv(
        output_file,
        index=False,
    )

    return df


def parse_timestamp(raw_text, title_text):
    """
    Return:
      replay_timestamp_utc
      timestamp_method

    Priority:
      1. Explicit UTC text on Ballchasing page.
      2. Parenthetical local clock + explicit numeric offset.
      3. Plain Ballchasing replay clock with the known -06
         display offset used in Jan/Feb 2025.

    The replay page itself remains the timestamp source.
    """

    raw_text = str(raw_text).strip()
    title_text = str(title_text or "").strip()

    # --------------------------------------------------------
    # 1. Explicit UTC
    # --------------------------------------------------------

    match = UTC_PATTERN.search(raw_text)

    if match:
        dt = datetime.strptime(
            match.group(1),
            "%Y-%m-%d %H:%M",
        ).replace(
            tzinfo=timezone.utc
        )

        return (
            dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "explicit_utc",
        )

    # --------------------------------------------------------
    # 2. Explicit local clock + numeric offset
    # --------------------------------------------------------

    match = LOCAL_WITH_OFFSET_PATTERN.search(raw_text)

    if match:
        local_dt = datetime.strptime(
            match.group(1),
            "%Y-%m-%d %H:%M",
        )

        offset_hours = int(
            match.group(2)
        )

        local_tz = timezone(
            timedelta(hours=offset_hours)
        )

        local_dt = local_dt.replace(
            tzinfo=local_tz
        )

        utc_dt = local_dt.astimezone(
            timezone.utc
        )

        return (
            utc_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            f"explicit_offset_{offset_hours:+03d}",
        )

    # --------------------------------------------------------
    # 3. Ballchasing clock with no timezone shown
    # --------------------------------------------------------

    match = PLAIN_DATETIME_PATTERN.search(raw_text)

    if match:
        local_dt = datetime.strptime(
            match.group(1),
            "%Y-%m-%d %H:%M",
        )

        local_tz = timezone(
            timedelta(
                hours=NAIVE_DISPLAY_OFFSET_HOURS
            )
        )

        local_dt = local_dt.replace(
            tzinfo=local_tz
        )

        utc_dt = local_dt.astimezone(
            timezone.utc
        )

        return (
            utc_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "ballchasing_display_clock_assumed_-06",
        )

    return None, None


def find_replay_date_tag(soup):
    """
    Locate the Ballchasing span whose title says
    "When the game took place...".
    """

    for tag in soup.find_all(
        ["span", "div", "td"]
    ):
        title = tag.get("title")

        if not title:
            continue

        if (
            "when the game took place"
            in str(title).lower()
        ):
            return tag

    return None


def extract_match_guid(page_text):
    match = MATCH_GUID_PATTERN.search(
        page_text
    )

    if match:
        return match.group(1)

    return None


def extract_server(page_text):
    match = SERVER_PATTERN.search(
        page_text
    )

    if match:
        return match.group(1)

    return None


def scrape_replay(session, replay_id):
    """
    Scrape one replay page.
    """

    url = BASE_URL.format(replay_id)

    last_error = None
    last_status = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:
            response = session.get(
                url,
                timeout=30,
            )

            last_status = response.status_code

            if response.status_code == 429:
                wait_seconds = min(
                    20 * attempt,
                    120,
                )

                print(
                    f"  HTTP 429 for {replay_id}; "
                    f"waiting {wait_seconds}s before retry"
                )

                time.sleep(wait_seconds)
                continue

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser",
            )

            page_title = (
                soup.title.get_text(
                    " ",
                    strip=True,
                )
                if soup.title
                else ""
            )

            page_text = soup.get_text(
                " ",
                strip=True,
            )

            date_tag = find_replay_date_tag(
                soup
            )

            if date_tag is None:
                return {
                    "replay_id": replay_id,
                    "url": url,
                    "http_status": response.status_code,
                    "scrape_success": False,
                    "replay_date_raw": pd.NA,
                    "replay_timestamp_utc": pd.NA,
                    "timestamp_method": pd.NA,
                    "match_guid": extract_match_guid(
                        page_text
                    ),
                    "server": extract_server(
                        page_text
                    ),
                    "page_title": page_title,
                    "error": "replay_date_tag_not_found",
                }

            raw_date = date_tag.get_text(
                " ",
                strip=True,
            )

            tag_title = date_tag.get(
                "title",
                "",
            )

            timestamp_utc, method = parse_timestamp(
                raw_date,
                tag_title,
            )

            success = (
                timestamp_utc is not None
            )

            return {
                "replay_id": replay_id,
                "url": url,
                "http_status": response.status_code,
                "scrape_success": success,
                "replay_date_raw": raw_date,
                "replay_timestamp_utc": (
                    timestamp_utc
                    if timestamp_utc is not None
                    else pd.NA
                ),
                "timestamp_method": (
                    method
                    if method is not None
                    else pd.NA
                ),
                "match_guid": extract_match_guid(
                    page_text
                ),
                "server": extract_server(
                    page_text
                ),
                "page_title": page_title,
                "error": (
                    ""
                    if success
                    else "timestamp_parse_failed"
                ),
            }

        except Exception as exc:
            last_error = repr(exc)

            if attempt < MAX_RETRIES:
                wait_seconds = min(
                    5 * attempt,
                    30,
                )

                print(
                    f"  Request error for {replay_id}; "
                    f"retrying after {wait_seconds}s"
                )

                time.sleep(wait_seconds)

    return {
        "replay_id": replay_id,
        "url": url,
        "http_status": last_status,
        "scrape_success": False,
        "replay_date_raw": pd.NA,
        "replay_timestamp_utc": pd.NA,
        "timestamp_method": pd.NA,
        "match_guid": pd.NA,
        "server": pd.NA,
        "page_title": pd.NA,
        "error": last_error or "request_failed",
    }


# ============================================================
# PROCESS ONE OPEN
# ============================================================

def process_open(open_name, games_file, output_file):

    divider(f"SCRAPING {open_name.upper()} REPLAY METADATA")

    if not games_file.exists():
        raise FileNotFoundError(
            f"Could not find games file:\n{games_file}"
        )

    games = pd.read_csv(
        games_file
    )

    if "replay_id" not in games.columns:
        raise ValueError(
            f"{games_file.name} has no replay_id column."
        )

    replay_ids = (
        games["replay_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print(
        f"\nReplay IDs found: "
        f"{len(replay_ids)}"
    )

    existing = load_existing(
        output_file
    )

    existing_rows = {}

    if len(existing) > 0:

        for _, row in existing.iterrows():
            existing_rows[
                str(row["replay_id"])
            ] = row.to_dict()

    successful_existing = {
        replay_id
        for replay_id, row
        in existing_rows.items()
        if str(
            row.get(
                "scrape_success",
                "",
            )
        ).lower()
        in {"true", "1"}
        and pd.notna(
            row.get(
                "replay_timestamp_utc"
            )
        )
    }

    print(
        f"Existing metadata rows: "
        f"{len(existing_rows)}"
    )

    print(
        f"Already successful: "
        f"{len(successful_existing)}"
    )

    remaining = [
        replay_id
        for replay_id in replay_ids
        if replay_id
        not in successful_existing
    ]

    print(
        f"Remaining to scrape: "
        f"{len(remaining)}"
    )

    all_rows = {
        replay_id: row
        for replay_id, row
        in existing_rows.items()
        if replay_id in replay_ids
    }

    session = requests.Session()
    session.headers.update(
        HEADERS
    )

    for index, replay_id in enumerate(
        remaining,
        start=1,
    ):

        result = scrape_replay(
            session,
            replay_id,
        )

        all_rows[replay_id] = result

        status_text = (
            "OK"
            if result["scrape_success"]
            else "FAILED"
        )

        print(
            f"[{index}/{len(remaining)}] "
            f"{replay_id} -> {status_text}"
        )

        save_metadata(
            list(all_rows.values()),
            output_file,
        )

        if index < len(remaining):
            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    final_df = save_metadata(
        list(all_rows.values()),
        output_file,
    )

    final_df = final_df[
        final_df["replay_id"]
        .astype(str)
        .isin(replay_ids)
    ].copy()

    success_mask = (
        final_df["scrape_success"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
        &
        final_df[
            "replay_timestamp_utc"
        ].notna()
    )

    successful = int(
        success_mask.sum()
    )

    failed = int(
        len(replay_ids) - successful
    )

    print(
        f"\nSuccessful: {successful}"
    )

    print(
        f"Failures: {failed}"
    )

    if "timestamp_method" in final_df.columns:
        print(
            "\nTimestamp methods:"
        )

        print(
            final_df[
                "timestamp_method"
            ]
            .value_counts(
                dropna=False
            )
        )

    print("\nSaved to:")
    print(output_file)

    return final_df


# ============================================================
# MAIN
# ============================================================

print("=" * 72)
print("BALLCHASING REPLAY METADATA - OPEN 2 AND OPEN 3")
print("=" * 72)

results = {}

for open_name, config in OPEN_CONFIG.items():

    results[open_name] = process_open(
        open_name,
        config["games_file"],
        config["output_file"],
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

divider("FINAL SUMMARY")

expected_counts = {
    "Open 2": 146,
    "Open 3": 140,
}

all_good = True

for open_name in [
    "Open 2",
    "Open 3",
]:

    df = results[open_name]

    success_mask = (
        df["scrape_success"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
        &
        df[
            "replay_timestamp_utc"
        ].notna()
    )

    success_count = int(
        success_mask.sum()
    )

    expected = expected_counts[
        open_name
    ]

    print(
        f"{open_name}: "
        f"{success_count}/{expected} "
        f"replay timestamps successful"
    )

    if success_count != expected:
        all_good = False

if all_good:
    print(
        "\nPASS: all 286 Open 2/Open 3 replay "
        "timestamps were collected successfully."
    )
else:
    print(
        "\nWARNING: one or more replay timestamps "
        "still need attention."
    )

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
