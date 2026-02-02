#!/usr/bin/env python3
"""
Create a HuggingFace dataset from multiple SQLite databases containing quiz bowl tournament data.

This script produces a DatasetDict with 6 configs (tables):
- players: Player records with team references
- teams: Team records with tournament info
- tossup_questions: Tossup questions with sanitized text and answers
- bonus_questions: Bonus questions with parts
- buzz_points: Buzz records linking players to tossups
- bonus_responses: Bonus part response records
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to allow imports when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))
from collections import defaultdict

from datasets import Dataset, DatasetDict
from huggingface_hub import whoami

from core import models
from utils import acf_sanitization, qb_tokenization


def create_player_entries(session, db_prefix: str) -> list[dict]:
    """
    Query all Players with their Team relationships.

    Args:
        session: SQLAlchemy session
        db_prefix: Prefix for unique IDs

    Returns:
        List of player dictionaries
    """
    players = session.query(models.Player).all()
    entries = []
    for player in players:
        entries.append({
            "player_id": f"{db_prefix}-p-{player.id}",
            "team_id": f"{db_prefix}-tm-{player.team_id}",
            "name": player.name,
            "slug": player.slug,
        })
    return entries


def create_team_entries(session, db_prefix: str) -> list[dict]:
    """
    Query all Teams with their Tournament relationships.

    Args:
        session: SQLAlchemy session
        db_prefix: Prefix for unique IDs

    Returns:
        List of team dictionaries
    """
    teams = session.query(models.Team).all()
    entries = []
    for team in teams:
        tournament = team.tournament
        entries.append({
            "team_id": f"{db_prefix}-tm-{team.id}",
            "tournament_id": f"{db_prefix}-tour-{tournament.id}",
            "tournament_slug": tournament.slug,
            "tournament_name": tournament.name,
            "name": team.name,
            "slug": team.slug,
        })
    return entries


def create_tossup_entries(session, db_prefix: str) -> tuple[list[dict], dict[int, str]]:
    """
    Create tossup question entries from the database.

    Args:
        session: SQLAlchemy session
        db_prefix: Prefix for unique IDs

    Returns:
        Tuple of (list of tossup dictionaries, mapping from tossup_id to qid)
    """
    tossups = session.query(models.Tossup).all()
    entries = []
    tossup_qid_map = {}  # tossup.id -> qid

    for tossup in tossups:
        question_sanitized = acf_sanitization.sanitize_question(tossup.question_text)
        clue_spans = qb_tokenization.get_clue_spans(
            question_sanitized, tokenization_scheme="blingfire"
        )
        answers = acf_sanitization.get_short_clean_answers(tossup.answer)

        question = tossup.question
        pq = question.packet_questions[0]
        qset = question.question_set_edition.question_set

        qid = f"{db_prefix}-t-{pq.packet_id}-{pq.question_number}"
        tossup_qid_map[tossup.id] = qid

        entries.append({
            "qid": qid,
            "question": question_sanitized,
            "answer_line": tossup.answer,
            "answer_primary": tossup.answer_primary,
            "clean_answers": list(answers["clean"]),
            "explanation": answers["explanation"],
            "clue_spans": clue_spans,
            "metadata": {
                "category": question.category_slug,
                "subcategory": [question.subcategory_slug],
                "category_main": question.category_main_slug,
                "category_full": question.category_full,
                "difficulty": qset.difficulty.split()[0],
                "question_set": qset.slug,
                "packet": pq.packet.name,
            },
        })

    return entries, tossup_qid_map


def create_bonus_entries(session, db_prefix: str) -> tuple[list[dict], dict[int, str]]:
    """
    Create bonus question entries from the database.

    Args:
        session: SQLAlchemy session
        db_prefix: Prefix for unique IDs

    Returns:
        Tuple of (list of bonus dictionaries, mapping from bonus_part_id to bonus qid)
    """
    bonuses = session.query(models.Bonus).all()
    entries = []
    bonus_part_qid_map = {}  # bonus_part.id -> bonus qid

    for bonus in bonuses:
        leadin = acf_sanitization.sanitize_question(bonus.leadin)
        q_info = bonus.question
        pq = q_info.packet_questions[0]
        qset = q_info.question_set_edition.question_set

        qid = f"{db_prefix}-b-{pq.packet_id}-{pq.question_number}"

        parts = []
        for part in bonus.bonus_parts:
            part_text = acf_sanitization.sanitize_answer(part.part)
            answers = acf_sanitization.get_short_clean_answers(part.answer)

            # Map this bonus part ID to the bonus qid
            bonus_part_qid_map[part.id] = qid

            parts.append({
                "number": part.part_number,
                "question": part_text,
                "answer_line": part.answer,
                "answer_primary": part.answer_primary,
                "clean_answers": list(answers["clean"]),
                "explanation": answers["explanation"],
                "value": part.value,
                "difficulty_modifier": part.difficulty_modifier,
            })

        entries.append({
            "qid": qid,
            "leadin": leadin,
            "parts": parts,
            "metadata": {
                "category": q_info.category_slug,
                "subcategory": [q_info.subcategory_slug],
                "category_main": q_info.category_main_slug,
                "category_full": q_info.category_full,
                "difficulty": qset.difficulty.split()[0],
                "question_set": qset.slug,
                "packet": pq.packet.name,
            },
        })

    return entries, bonus_part_qid_map


def create_buzz_point_entries(
    session, db_prefix: str, tossup_qid_map: dict[int, str]
) -> list[dict]:
    """
    Create buzz point entries from the database.

    Args:
        session: SQLAlchemy session
        db_prefix: Prefix for unique IDs
        tossup_qid_map: Mapping from tossup_id to qid

    Returns:
        List of buzz point dictionaries
    """
    buzzes = session.query(models.Buzz).all()

    # Group buzzes by (game_id, tossup_id) for buzz_number computation
    game_tossup_buzzes = defaultdict(list)
    for buzz in buzzes:
        key = (buzz.game_id, buzz.tossup_id)
        game_tossup_buzzes[key].append(buzz)

    # Compute buzz numbers for each buzz
    buzz_numbers = {}  # buzz.id -> buzz_number
    for key, group in game_tossup_buzzes.items():
        sorted_group = sorted(group, key=lambda b: b.buzz_position)
        for i, buzz in enumerate(sorted_group, 1):
            buzz_numbers[buzz.id] = i

    entries = []
    for buzz in buzzes:
        # Get player's team_id
        player = buzz.player
        team_id = f"{db_prefix}-tm-{player.team_id}"

        entries.append({
            "buzz_id": f"{db_prefix}-bz-{buzz.id}",
            "qid": tossup_qid_map.get(buzz.tossup_id, f"{db_prefix}-t-unknown-{buzz.tossup_id}"),
            "game_id": f"{db_prefix}-g-{buzz.game_id}",
            "team_id": team_id,
            "player_id": f"{db_prefix}-p-{buzz.player_id}",
            "buzz_number": buzz_numbers[buzz.id],
            "correctness": buzz.value > 0,
            "value": buzz.value,
            "token_position": buzz.buzz_position,
        })

    return entries


def create_bonus_response_entries(
    session, db_prefix: str, bonus_part_qid_map: dict[int, str]
) -> list[dict]:
    """
    Create bonus response entries from the database.

    Args:
        session: SQLAlchemy session
        db_prefix: Prefix for unique IDs
        bonus_part_qid_map: Mapping from bonus_part_id to bonus qid

    Returns:
        List of bonus response dictionaries
    """
    bonus_part_directs = session.query(models.BonusPartDirect).all()
    entries = []

    for bpd in bonus_part_directs:
        entries.append({
            "response_id": f"{db_prefix}-br-{bpd.id}",
            "bonus_part_id": f"{db_prefix}-bp-{bpd.bonus_part_id}",
            "bonus_qid": bonus_part_qid_map.get(
                bpd.bonus_part_id, f"{db_prefix}-b-unknown-{bpd.bonus_part_id}"
            ),
            "team_id": f"{db_prefix}-tm-{bpd.team_id}",
            "game_id": f"{db_prefix}-g-{bpd.game_id}",
            "correctness": bpd.value > 0,
            "value": bpd.value,
        })

    return entries


def load_from_db(db_path: str, prefix: str) -> dict[str, list[dict]]:
    """
    Load all 6 table types from a single database.

    Args:
        db_path: Path to the SQLite database
        prefix: Prefix for unique IDs

    Returns:
        Dictionary mapping table name to list of records
    """
    session = models.create_session(db_path)

    # Create entries with mappings for cross-references
    tossup_entries, tossup_qid_map = create_tossup_entries(session, prefix)
    bonus_entries, bonus_part_qid_map = create_bonus_entries(session, prefix)

    return {
        "players": create_player_entries(session, prefix),
        "teams": create_team_entries(session, prefix),
        "tossup_questions": tossup_entries,
        "bonus_questions": bonus_entries,
        "buzz_points": create_buzz_point_entries(session, prefix, tossup_qid_map),
        "bonus_responses": create_bonus_response_entries(session, prefix, bonus_part_qid_map),
    }


def create_tournament_dataset(
    db_paths: list[str], prefixes: list[str]
) -> DatasetDict:
    """
    Create a DatasetDict with 6 configs from multiple databases.

    Args:
        db_paths: List of paths to SQLite databases
        prefixes: List of prefixes for each database

    Returns:
        DatasetDict with 6 configs (players, teams, tossup_questions,
        bonus_questions, buzz_points, bonus_responses)
    """
    # Initialize merged records
    merged_records = {
        "players": [],
        "teams": [],
        "tossup_questions": [],
        "bonus_questions": [],
        "buzz_points": [],
        "bonus_responses": [],
    }

    # Load and merge records from each database
    for db_path, prefix in zip(db_paths, prefixes):
        print(f"Loading from {db_path} with prefix '{prefix}'...")
        records = load_from_db(db_path, prefix)
        for table_name, table_records in records.items():
            merged_records[table_name].extend(table_records)
            print(f"  {table_name}: {len(table_records)} records")

    # Create DatasetDict
    dataset_dict = DatasetDict()
    for table_name, records in merged_records.items():
        print(f"Creating {table_name} dataset with {len(records)} total records...")
        dataset_dict[table_name] = Dataset.from_list(records)

    return dataset_dict


def main():
    parser = argparse.ArgumentParser(
        description="Create a HuggingFace dataset from quiz bowl tournament databases"
    )
    parser.add_argument(
        "--db-paths", "-d",
        nargs="+",
        required=True,
        help="Paths to SQLite databases"
    )
    parser.add_argument(
        "--prefixes", "-p",
        nargs="+",
        required=True,
        help="Prefixes for each database (must match count of db-paths)"
    )
    parser.add_argument(
        "--repo-id", "-r",
        required=True,
        help="HuggingFace repository ID"
    )
    parser.add_argument(
        "--org", "-o",
        help="HuggingFace organization (optional)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the dataset public"
    )

    args = parser.parse_args()

    # Validate arguments
    if len(args.db_paths) != len(args.prefixes):
        parser.error("Number of --db-paths must match number of --prefixes")

    # Create the dataset
    dataset_dict = create_tournament_dataset(args.db_paths, args.prefixes)

    # Print summary
    print("\nDataset Summary:")
    for config_name, dataset in dataset_dict.items():
        print(f"  {config_name}: {len(dataset)} records")

    # Construct the full repo ID
    if args.org:
        full_repo_id = f"{args.org}/{args.repo_id}"
    else:
        try:
            user_info = whoami()
            full_repo_id = f"{user_info['name']}/{args.repo_id}"
        except Exception:
            full_repo_id = args.repo_id

    # Push to HuggingFace Hub
    print(f"\nPushing to HuggingFace Hub: {full_repo_id}")
    dataset_dict.push_to_hub(
        full_repo_id,
        private=not args.public,
    )
    print(f"Dataset pushed successfully to {full_repo_id}")


if __name__ == "__main__":
    main()
