"""Download missing all-seasons CSVs for Football-Data's extra leagues.

Existing local files are never overwritten. The raw files are retained as
provider downloads until their country-specific season schema is reviewed.
"""

from __future__ import annotations

import argparse
import csv
from io import StringIO
from pathlib import Path
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_prediction.football_data import COMPETITIONS


EXTRA_LEAGUE_CODES = (
    "ARG", "AUT", "BRA", "CHN", "DNK", "FIN", "IRL", "JPN",
    "MEX", "NOR", "POL", "ROU", "RUS", "SWE", "SWZ", "USA",
)
SOURCE_URL = "https://football-data.co.uk/new/{code}.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "extra"


def validate_csv(content: bytes, code: str) -> None:
    """Check the separate extra-league schema, not the main-league schema.

    The country exports use the shorter Home/Away/HG/AG names and carry a
    Season column; they do not use the main files' Div/HomeTeam/AwayTeam/
    FTHG/FTAG headers.
    """
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("cp1252")
    reader = csv.DictReader(StringIO(text, newline=""))
    headers = {
        header.strip().casefold()
        for header in (reader.fieldnames or [])
        if header
    }
    required = {"season", "date", "home", "away", "hg", "ag"}
    missing = required - headers
    if missing:
        raise ValueError(
            f"{code} extra-league CSV is missing columns: {', '.join(sorted(missing))}; "
            f"found: {', '.join(sorted(headers))}"
        )


def download_missing(output_dir: Path, *, pause_seconds: float = 0.5) -> tuple[int, int]:
    """Download missing official CSVs atomically, preserving existing files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    for code in EXTRA_LEAGUE_CODES:
        target = output_dir / f"{code}.csv"
        if target.is_file():
            print(f"KEEP existing {code}: {target}")
            skipped += 1
            continue

        url = SOURCE_URL.format(code=code)
        request = Request(url, headers={"User-Agent": "FootballPredictionApp/1.0"})
        try:
            with urlopen(request, timeout=30) as response:
                content = response.read()
        except (OSError, URLError) as exc:
            raise RuntimeError(f"Could not download {url}: {exc}") from exc

        validate_csv(content, code)
        temporary = target.with_suffix(".csv.part")
        temporary.write_bytes(content)
        temporary.replace(target)
        print(f"SAVED {COMPETITIONS[code]} ({code}): {target} ({len(content):,} bytes)")
        downloaded += 1
        time.sleep(pause_seconds)

    return downloaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download only missing Football-Data extra-league CSVs."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Destination for the raw country CSVs (default: data/extra)",
    )
    args = parser.parse_args()
    downloaded, skipped = download_missing(args.output_dir)
    print(f"\nDownloaded {downloaded}; preserved {skipped} existing files.")
    print("These are raw all-seasons files; inspect season labels before modeling.")


if __name__ == "__main__":
    main()
