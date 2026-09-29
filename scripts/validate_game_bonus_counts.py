#!/usr/bin/env python3
"""Compare correct tossups with bonuses played in each ACF game.

A correct tossup is a distinct (game, tossup) with a positive buzz value.
A played bonus is one set of three bonus-part response rows. The same bonus
ID can recur within a game, so counting distinct bonus IDs undercounts plays.
The script checks that every source bonus has three parts and each game's
response count is divisible by three before calculating the comparison.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data/dbs/acf-co-23-25.db"


def validate(db_path: Path) -> list[dict]:
    with sqlite3.connect(db_path) as connection:
        corrects = dict(connection.execute(
            "SELECT game_id, COUNT(DISTINCT tossup_id) FROM buzz "
            "WHERE typeof(value) = 'integer' AND value > 0 GROUP BY game_id"
        ))
        wrong_part_counts = list(connection.execute(
            "SELECT bonus_id, COUNT(*) FROM bonus_part GROUP BY bonus_id HAVING COUNT(*) != 3"
        ))
        if wrong_part_counts:
            raise ValueError(f"bonuses without three parts: {wrong_part_counts[:10]}")
        bonus_part_rows = dict(connection.execute(
            "SELECT game_id, COUNT(*) FROM bonus_part_direct GROUP BY game_id"
        ))
        incomplete_games = [
            game_id for game_id, count in bonus_part_rows.items() if count % 3
        ]
        if incomplete_games:
            raise ValueError(f"games with incomplete bonus-part triplets: {incomplete_games[:10]}")
        rows = []
        for game_id, date, tournament, round_number, packet, tossups_read in connection.execute(
            """
            SELECT g.id, tr.start_date, tr.name, r.number, p.name, g.tossups_read
            FROM game AS g
            JOIN round AS r ON r.id = g.round_id
            JOIN tournament AS tr ON tr.id = r.tournament_id
            JOIN packet AS p ON p.id = r.packet_id
            ORDER BY g.id
            """
        ):
            correct = corrects.get(game_id, 0)
            part_rows = bonus_part_rows.get(game_id, 0)
            bonus = part_rows // 3
            rows.append({
                "game_id": game_id,
                "year": date[:4] if date else "",
                "tournament": tournament,
                "round": round_number,
                "packet": packet,
                "tossups_read": tossups_read,
                "correct_tossups": correct,
                "bonus_part_rows": part_rows,
                "bonuses_played": bonus,
                "difference": correct - bonus,
            })
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--csv", type=Path, help="Write the mismatched games")
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error(f"database not found: {args.db}")

    rows = validate(args.db)
    mismatches = [row for row in rows if row["difference"]]
    differences = Counter(row["difference"] for row in rows)
    print(f"Games: {len(rows):,}; matching: {differences[0]:,}; mismatched: {len(mismatches):,}")
    print("Difference (correct tossups - bonuses played): " + ", ".join(
        f"{difference:+d}: {count:,}" for difference, count in sorted(differences.items())
    ))
    print("game  year  tournament  round  correct  bonuses  difference")
    for row in mismatches:
        print(f"{row['game_id']:4d}  {row['year']}  {row['tournament']}  "
              f"{row['round']}  {row['correct_tossups']}  {row['bonuses_played']}  "
              f"{row['difference']:+d}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "game_id", "year", "tournament", "round", "packet",
                "tossups_read", "correct_tossups", "bonus_part_rows",
                "bonuses_played", "difference",
            ))
            writer.writeheader()
            writer.writerows(mismatches)
        print(f"Wrote {len(mismatches):,} mismatches to {args.csv}")


if __name__ == "__main__":
    main()
