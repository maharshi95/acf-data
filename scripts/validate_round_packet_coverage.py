#!/usr/bin/env python3
"""Audit game coverage by tournament round and packet.

Every recorded round has one packet. A registered team absent from a round is
reported for investigation, not treated as proof of a missing game: byes and
playoff formats can also explain the absence. Without a schedule, missing
opponents cannot be reconstructed reliably.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data/dbs/acf-co-23-25.db"


def audit(db_path: Path) -> tuple[list[dict], list[dict], Counter]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        tournaments = connection.execute("""
            SELECT tr.id, tr.name, tr.start_date, qs.slug AS question_set
            FROM tournament tr
            LEFT JOIN question_set_edition e ON e.id = tr.question_set_edition_id
            LEFT JOIN question_set qs ON qs.id = e.question_set_id
            ORDER BY tr.id
        """).fetchall()
        teams = defaultdict(dict)
        for team in connection.execute("SELECT id, tournament_id, name FROM team"):
            teams[team["tournament_id"]][team["id"]] = team["name"]
        games = defaultdict(list)
        for game in connection.execute("""
            SELECT id, round_id, team_one_id, team_two_id FROM game ORDER BY id
        """):
            games[game["round_id"]].append(game)

        round_rows = []
        rounds_by_packet = Counter()
        games_by_packet = Counter()
        totals = Counter()
        for tournament in tournaments:
            tournament_id = tournament["id"]
            roster = teams[tournament_id]
            rounds = connection.execute("""
                SELECT r.id, r.number, r.packet_id, p.name AS packet,
                       qs.slug AS packet_question_set
                FROM round r
                LEFT JOIN packet p ON p.id = r.packet_id
                LEFT JOIN question_set_edition e ON e.id = p.question_set_edition_id
                LEFT JOIN question_set qs ON qs.id = e.question_set_id
                WHERE r.tournament_id = ?
                ORDER BY r.number, r.id
            """, (tournament_id,)).fetchall()
            totals["tournaments"] += 1
            if not rounds:
                totals["tournaments_without_rounds"] += 1
            for round_row in rounds:
                round_games = games[round_row["id"]]
                appearances = Counter(
                    team_id for game in round_games
                    for team_id in (game["team_one_id"], game["team_two_id"])
                    if team_id is not None
                )
                absent = sorted(set(roster) - appearances.keys())
                repeated = sorted(team_id for team_id, count in appearances.items() if count > 1)
                foreign = sorted(set(appearances) - set(roster))
                self_games = [game["id"] for game in round_games
                              if game["team_one_id"] == game["team_two_id"]]
                null_teams = [game["id"] for game in round_games
                              if game["team_one_id"] is None or game["team_two_id"] is None]
                possible_pair = (
                    " vs ".join(f"{team_id}:{roster[team_id]}" for team_id in absent)
                    if len(roster) % 2 == 0 and len(absent) == 2
                    and len(round_games) == len(roster) // 2 - 1
                    and not (repeated or foreign or self_games or null_teams)
                    else ""
                )
                packet_id = round_row["packet_id"]
                if packet_id is not None:
                    rounds_by_packet[packet_id] += 1
                    games_by_packet[packet_id] += len(round_games)
                totals["rounds"] += 1
                totals["games"] += len(round_games)
                if not round_games:
                    totals["rounds_without_games"] += 1
                if absent:
                    totals["rounds_with_absent_registered_teams"] += 1
                if repeated or foreign or self_games or null_teams:
                    totals["rounds_with_team_integrity_issues"] += 1
                if round_row["packet_id"] is None:
                    totals["rounds_without_packet"] += 1
                if (round_row["packet_question_set"] is not None
                        and tournament["question_set"] != round_row["packet_question_set"]):
                    totals["rounds_with_cross_set_packet"] += 1
                round_rows.append({
                    "tournament_id": tournament_id,
                    "tournament": tournament["name"],
                    "year": (tournament["start_date"] or "")[:4],
                    "question_set": tournament["question_set"] or "",
                    "round_id": round_row["id"],
                    "round_number": round_row["number"],
                    "packet_id": packet_id,
                    "packet": round_row["packet"] or "",
                    "game_count": len(round_games),
                    "registered_team_count": len(roster),
                    "full_field_game_target": len(roster) // 2,
                    "shortfall_if_full_field": max(0, len(roster) // 2 - len(round_games)),
                    "teams_in_games": len(appearances),
                    "absent_registered_teams": "; ".join(
                        f"{team_id}:{roster[team_id]}" for team_id in absent
                    ),
                    "possible_pair_if_full_field": possible_pair,
                    "teams_in_multiple_games": "; ".join(
                        f"{team_id}:{roster.get(team_id, '?')} ({appearances[team_id]})"
                        for team_id in repeated
                    ),
                    "foreign_team_ids": "; ".join(map(str, foreign)),
                    "self_game_ids": "; ".join(map(str, self_games)),
                    "null_team_game_ids": "; ".join(map(str, null_teams)),
                    "game_ids": "; ".join(str(game["id"]) for game in round_games),
                })
            if rounds:
                peak_games = max(len(games[round_row["id"]]) for round_row in rounds)
                for row in round_rows[-len(rounds):]:
                    row["tournament_peak_game_count"] = peak_games
                    row["shortfall_vs_tournament_peak"] = max(
                        0, peak_games - row["game_count"]
                    )

        packet_rows = []
        question_ids_by_packet = defaultdict(set)
        for packet_id, question_id in connection.execute(
            "SELECT packet_id, question_id FROM packet_question"
        ):
            question_ids_by_packet[packet_id].add(question_id)
        used_packet_ids_by_set = defaultdict(list)
        for packet_id, question_set in connection.execute("""
            SELECT p.id, qs.slug FROM packet p
            JOIN question_set_edition e ON e.id = p.question_set_edition_id
            JOIN question_set qs ON qs.id = e.question_set_id
        """):
            if rounds_by_packet[packet_id]:
                used_packet_ids_by_set[question_set].append(packet_id)
        for packet in connection.execute("""
            SELECT p.id, p.name, p.number, e.slug AS edition, qs.slug AS question_set,
                   COUNT(DISTINCT pq.question_id) AS question_count
            FROM packet p
            JOIN question_set_edition e ON e.id = p.question_set_edition_id
            JOIN question_set qs ON qs.id = e.question_set_id
            LEFT JOIN packet_question pq ON pq.packet_id = p.id
            GROUP BY p.id ORDER BY qs.slug, e.slug, p.number, p.id
        """):
            packet_id = packet["id"]
            round_count = rounds_by_packet[packet_id]
            game_count = games_by_packet[packet_id]
            question_ids = question_ids_by_packet[packet_id]
            best_used_packet_id = ""
            shared_questions = 0
            if not round_count and question_ids:
                for used_packet_id in used_packet_ids_by_set[packet["question_set"]]:
                    overlap = len(question_ids & question_ids_by_packet[used_packet_id])
                    if overlap > shared_questions:
                        shared_questions = overlap
                        best_used_packet_id = used_packet_id
            exact_shadow = bool(
                not round_count and question_ids and best_used_packet_id
                and question_ids == question_ids_by_packet[best_used_packet_id]
            )
            totals["packets"] += 1
            if not round_count:
                totals["packets_without_rounds"] += 1
            if exact_shadow:
                totals["unassigned_exact_shadows"] += 1
            packet_rows.append({
                "question_set": packet["question_set"],
                "edition": packet["edition"],
                "packet_id": packet_id,
                "packet_number": packet["number"],
                "packet": packet["name"],
                "question_count": packet["question_count"],
                "round_count": round_count,
                "game_count": game_count,
                "best_used_packet_id": best_used_packet_id,
                "shared_questions_with_used_packet": shared_questions,
                "exact_shadow_of_used_packet": exact_shadow,
            })
    return round_rows, packet_rows, totals


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--rounds-csv", type=Path, help="Write coverage for every recorded round")
    parser.add_argument("--packets-csv", type=Path, help="Write usage for every packet")
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error(f"database not found: {args.db}")
    round_rows, packet_rows, totals = audit(args.db)
    print(f"Tournaments: {totals['tournaments']}; rounds: {totals['rounds']}; "
          f"games: {totals['games']}; packets: {totals['packets']}")
    for key in ("tournaments_without_rounds", "rounds_without_packet",
                "rounds_without_games", "rounds_with_team_integrity_issues",
                "rounds_with_cross_set_packet", "rounds_with_absent_registered_teams",
                "packets_without_rounds", "unassigned_exact_shadows"):
        print(f"{key}: {totals[key]}")
    print("Rounds below full-field game target: "
          f"{sum(row['shortfall_if_full_field'] > 0 for row in round_rows)}; "
          "below tournament peak: "
          f"{sum(row['shortfall_vs_tournament_peak'] > 0 for row in round_rows)}")
    print("Rounds with one possible unrecorded pairing under full-field assumption: "
          f"{sum(bool(row['possible_pair_if_full_field']) for row in round_rows)}")
    print("Absent registered teams can reflect byes, playoffs, or missing games; "
          "the database has no schedule to distinguish them.")
    if args.rounds_csv:
        write_csv(args.rounds_csv, round_rows)
        print(f"Wrote {len(round_rows)} rounds to {args.rounds_csv}")
    if args.packets_csv:
        write_csv(args.packets_csv, packet_rows)
        print(f"Wrote {len(packet_rows)} packets to {args.packets_csv}")


if __name__ == "__main__":
    main()
