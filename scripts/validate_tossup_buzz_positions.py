#!/usr/bin/env python3
"""Compare each tossup's latest recorded buzz with its end-marker position.

MODAQ stores zero-indexed word positions and adds a buzzable END marker after
the last word. The dataset builder subtracts the leading moderator-instruction
offset, while the published question text is sanitized. A distance of zero
means the latest buzz is at the sanitized question's END index. These token
counts use whitespace splitting as a diagnostic approximation; MODAQ skips
some formatted text, such as pronunciation guides and reader directives.
Clue spans use a separate sentence splitter and are character offsets.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.acf_sanitization import get_buzz_offset, sanitize_question

DEFAULT_DB = ROOT / "data/dbs/acf-co-23-25.db"


def validate(db_path: Path) -> tuple[list[dict], Counter]:
    results = []
    totals = Counter()
    with sqlite3.connect(db_path) as connection:
        buzz_positions: dict[int, Counter[int]] = {}
        for tossup_id, position, count in connection.execute(
            """SELECT tossup_id, buzz_position, COUNT(*)
               FROM buzz
               WHERE typeof(buzz_position) = 'integer'
               GROUP BY tossup_id, buzz_position"""
        ):
            buzz_positions.setdefault(tossup_id, Counter())[position] = count
        for (
            tossup_id,
            question,
            answer_line,
            question_slug,
            question_set_slug,
            last_buzz,
            buzz_count,
            invalid_count,
        ) in connection.execute(
            """
            SELECT t.id, t.question, t.answer, q.slug,
                   (SELECT qs.slug
                    FROM packet_question AS pq
                    JOIN packet AS p ON p.id = pq.packet_id
                    JOIN question_set_edition AS qse ON qse.id = p.question_set_edition_id
                    JOIN question_set AS qs ON qs.id = qse.question_set_id
                    WHERE pq.question_id = q.id
                    ORDER BY pq.id LIMIT 1),
                   MAX(CASE WHEN typeof(b.buzz_position) = 'integer'
                            THEN b.buzz_position END),
                   COUNT(b.id),
                   SUM(CASE WHEN b.id IS NOT NULL
                                 AND typeof(b.buzz_position) != 'integer'
                            THEN 1 ELSE 0 END)
            FROM tossup AS t
            JOIN question AS q ON q.id = t.question_id
            LEFT JOIN buzz AS b ON b.tossup_id = t.id
            GROUP BY t.id
            ORDER BY t.id
            """
        ):
            totals["tossups"] += 1
            totals["buzzes"] += buzz_count
            totals["invalid_buzz_positions"] += invalid_count
            if last_buzz is None:
                totals["without_numeric_buzz"] += 1
                continue
            raw_token_count = len(question.split())
            token_count = len(sanitize_question(question).split())
            offset = get_buzz_offset(question)
            adjusted_buzz = last_buzz - offset
            distance = adjusted_buzz - token_count
            raw_distance = last_buzz - raw_token_count
            beyond_end_positions = {
                position: count
                for position, count in buzz_positions[tossup_id].items()
                if position - offset > token_count
            }
            beyond_end_buzz_count = sum(beyond_end_positions.values())
            beyond_end_distinct_position_count = len(beyond_end_positions)
            cause = (
                "within_sanitized_count"
                if distance <= 0
                else (
                    "past_raw_whitespace_count"
                    if raw_distance > 0
                    else "within_raw_whitespace_count"
                )
            )
            question_url = (
                "https://quizbowlstats.com/buzzpoints/set/"
                f"{quote(question_set_slug, safe='')}/tossup/{quote(question_slug, safe='')}"
                if question_set_slug and question_slug
                else ""
            )
            tournaments = []
            if distance > 0:
                tournaments = [
                    (date[:4] if date else "", name or "")
                    for date, name in connection.execute(
                        """
                        SELECT DISTINCT tr.start_date, tr.name
                        FROM buzz AS b
                        JOIN game AS g ON g.id = b.game_id
                        JOIN round AS r ON r.id = g.round_id
                        JOIN tournament AS tr ON tr.id = r.tournament_id
                        WHERE b.tossup_id = ? AND b.buzz_position = ?
                        ORDER BY tr.start_date, tr.name
                        """,
                        (tossup_id, last_buzz),
                    )
                ]
            results.append(
                {
                    "tossup_id": tossup_id,
                    "answer_line": answer_line,
                    "question_url": question_url,
                    "tournaments": tournaments,
                    "raw_token_count": raw_token_count,
                    "token_count": token_count,
                    "instruction_offset": offset,
                    "last_buzz_raw": last_buzz,
                    "last_buzz_adjusted": adjusted_buzz,
                    "distance_from_end": distance,
                    "distance_from_raw_end": raw_distance,
                    "cause": cause,
                    "buzz_count": buzz_count,
                    "beyond_end_buzz_count": beyond_end_buzz_count,
                    "beyond_end_distinct_position_count": beyond_end_distinct_position_count,
                }
            )
            totals["with_numeric_buzz"] += 1
            totals["beyond_end_buzzes"] += beyond_end_buzz_count
            totals[
                "beyond_end_distinct_positions"
            ] += beyond_end_distinct_position_count
            totals[
                (
                    "beyond_end"
                    if distance > 0
                    else "at_end" if distance == 0 else "before_end"
                )
            ] += 1
            if distance > 0:
                totals[cause] += 1
    return results, totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help="Merged ACF SQLite database"
    )
    parser.add_argument(
        "--csv", type=Path, help="Write one row per tossup with a numeric buzz"
    )
    parser.add_argument(
        "--overshoots-csv",
        type=Path,
        help="Write overshoots with one row per tournament at the latest buzz",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Examples to show at each extreme"
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if not args.db.is_file():
        parser.error(f"database not found: {args.db}")

    results, totals = validate(args.db)
    print(f"Database: {args.db}")
    print(
        tabulate(
            [
                ("Tossups", f"{totals['tossups']:,}"),
                ("With numeric buzz", f"{totals['with_numeric_buzz']:,}"),
                ("Without numeric buzz", f"{totals['without_numeric_buzz']:,}"),
                ("Buzzes", f"{totals['buzzes']:,}"),
                ("Nonnumeric positions", f"{totals['invalid_buzz_positions']:,}"),
            ],
            headers=("Metric", "Count"),
            tablefmt="simple",
        )
    )
    print(
        "Distance = (max source word_index - leading instruction offset) "
        "- sanitized whitespace token count (END index)"
    )
    print(
        tabulate(
            [
                ("Before END", f"{totals['before_end']:,}"),
                ("At END", f"{totals['at_end']:,}"),
                ("Past END", f"{totals['beyond_end']:,}"),
                ("Buzz records past END", f"{totals['beyond_end_buzzes']:,}"),
                (
                    "Out-of-range positions",
                    f"{totals['beyond_end_distinct_positions']:,}",
                ),
                (
                    "Past raw whitespace count",
                    f"{totals['past_raw_whitespace_count']:,}",
                ),
                (
                    "Past sanitized END (within raw count)",
                    f"{totals['within_raw_whitespace_count']:,}",
                ),
            ],
            headers=("Result", "Count"),
            tablefmt="simple",
        )
    )
    if results:
        distances = Counter(row["distance_from_end"] for row in results)
        print("Most common distances:")
        print(
            tabulate(
                [
                    (distance, f"{count:,}")
                    for distance, count in distances.most_common(12)
                ],
                headers=("Distance from END", "Tossups"),
                tablefmt="simple",
            )
        )
        overshoots = sorted(
            (row for row in results if row["distance_from_end"] > 0),
            key=lambda row: (-row["distance_from_end"], row["tossup_id"]),
        )
        for title, rows in (
            ("Largest overshoots", overshoots),
            (
                "Earliest last buzzes",
                sorted(
                    results,
                    key=lambda row: (row["distance_from_end"], row["tossup_id"]),
                ),
            ),
        ):
            print(f"{title}:")
            shown_rows = rows[: args.limit]
            print(
                tabulate(
                    [
                        (
                            row["tossup_id"],
                            f"{row['distance_from_end']:+d}",
                            f"{row['distance_from_raw_end']:+d}",
                            row["last_buzz_adjusted"],
                            row["token_count"],
                            row["buzz_count"],
                            row["beyond_end_buzz_count"],
                            row["beyond_end_distinct_position_count"],
                        )
                        for row in shown_rows
                    ],
                    headers=(
                        "ID",
                        "Sanitized distance",
                        "Raw distance",
                        "Adjusted buzz",
                        "Tokens",
                        "Buzzes",
                        "Past-END buzzes",
                        "Past-END positions",
                    ),
                    tablefmt="simple",
                )
            )
            for row in shown_rows:
                if row["distance_from_end"] > 0:
                    tournament_text = (
                        "; ".join(
                            f"{year} | {name}" for year, name in row["tournaments"]
                        )
                        or "unknown tournament"
                    )
                    print(f"         {tournament_text} | Answer: {row['answer_line']}")
                    if row["question_url"]:
                        print(
                            f"         [{row['question_url']}]({row['question_url']})"
                        )
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "tossup_id",
                    "raw_token_count",
                    "token_count",
                    "instruction_offset",
                    "last_buzz_raw",
                    "last_buzz_adjusted",
                    "distance_from_end",
                    "distance_from_raw_end",
                    "cause",
                    "buzz_count",
                    "answer_line",
                    "question_url",
                    "tournaments",
                    "beyond_end_buzz_count",
                    "beyond_end_distinct_position_count",
                ),
            )
            writer.writeheader()
            writer.writerows(
                {
                    **row,
                    "tournaments": "; ".join(
                        f"{year} {name}" for year, name in row["tournaments"]
                    ),
                }
                for row in results
            )
        print(f"Wrote {len(results):,} rows to {args.csv}")
    if args.overshoots_csv:
        args.overshoots_csv.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with args.overshoots_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "tossup_id",
                    "year",
                    "tournament_name",
                    "answer_line",
                    "question_url",
                    "distance_from_end",
                    "distance_from_raw_end",
                    "cause",
                    "last_buzz_raw",
                    "token_count",
                    "raw_token_count",
                    "beyond_end_buzz_count",
                    "beyond_end_distinct_position_count",
                ),
            )
            writer.writeheader()
            for row in results:
                if row["distance_from_end"] <= 0:
                    continue
                for year, name in row["tournaments"] or [("", "")]:
                    writer.writerow(
                        {
                            "tossup_id": row["tossup_id"],
                            "year": year,
                            "tournament_name": name,
                            "answer_line": row["answer_line"],
                            "question_url": row["question_url"],
                            "distance_from_end": row["distance_from_end"],
                            "distance_from_raw_end": row["distance_from_raw_end"],
                            "cause": row["cause"],
                            "last_buzz_raw": row["last_buzz_raw"],
                            "token_count": row["token_count"],
                            "raw_token_count": row["raw_token_count"],
                            "beyond_end_buzz_count": row["beyond_end_buzz_count"],
                            "beyond_end_distinct_position_count": row[
                                "beyond_end_distinct_position_count"
                            ],
                        }
                    )
                    count += 1
        print(f"Wrote {count:,} overshoot tournament rows to {args.overshoots_csv}")


if __name__ == "__main__":
    main()
