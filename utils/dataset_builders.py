"""
Shared dataset building utilities for creating HuggingFace datasets from quiz bowl data.

This module provides common functions for:
- Creating tossup question entries with sanitization and tokenization
- Creating bonus question entries with parts
- Creating player and team entries with proper ID generation
- Creating buzz point and bonus response entries

These functions are shared between different dataset creation scripts.
"""

import hashlib
from collections import defaultdict

from core import models
from utils import acf_sanitization, qb_tokenization


def compute_team_player_hash(player_slugs: list[str]) -> str:
    """
    Compute a hash for a team based on the sorted player slugs.

    This allows identifying teams with the same set of players across
    different tournaments.

    Args:
        player_slugs: List of player slugs on the team

    Returns:
        A hex digest hash string based on sorted player slugs
    """
    sorted_slugs = sorted(player_slugs)
    hash_input = "|".join(sorted_slugs)
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def create_tossup_entry_dict(
    tossup: models.Tossup,
    prefix: str,
    include_buzz_positions: bool = False,
    use_type_in_qid: bool = True,
) -> dict:
    """
    Create a tossup question entry dictionary from a Tossup model.

    This is the shared implementation for creating tossup entries that can be
    used by different dataset creation scripts.

    Args:
        tossup: The Tossup model instance
        prefix: Prefix for the QID
        include_buzz_positions: Whether to include human buzz positions in metadata
        use_type_in_qid: Whether to include type marker (-t-) in QID

    Returns:
        Dictionary containing tossup question data
    """
    question_sanitized = acf_sanitization.sanitize_question(tossup.question_text)
    clue_spans = qb_tokenization.get_clue_spans(
        question_sanitized, tokenization_scheme="blingfire"
    )
    answers = acf_sanitization.get_short_clean_answers(tossup.answer)

    question = tossup.question
    pq = question.packet_questions[0]
    qset = question.question_set_edition.question_set

    if use_type_in_qid:
        prefix = f"{prefix}-t"
    qid = f"{prefix}-{pq.packet_id:02d}-{pq.question_number:02d}"

    metadata = {
        "category": question.category_slug,
        "subcategory": [question.subcategory_slug],
        "category_main": question.category_main_slug,
        "category_full": question.category_full,
        "difficulty": qset.difficulty.split()[0],
        "question_set": qset.slug,
        "packet": pq.packet.name,
    }

    if include_buzz_positions:
        buzz_offset = acf_sanitization.get_buzz_offset(tossup.question_text)
        human_buzz_positions = sorted(
            [(b.buzz_position - buzz_offset, b.value) for b in tossup.buzzes]
        )
        metadata["human_buzz_positions"] = human_buzz_positions

    return {
        "qid": qid,
        "question": question_sanitized,
        "answer_line": tossup.answer,
        "answer_sanitized": tossup.answer_sanitized,
        "answer_primary": tossup.answer_primary,
        "clean_answers": list(answers["clean"]),
        "explanation": answers["explanation"],
        "clue_spans": clue_spans,
        "metadata": metadata,
    }


def create_bonus_entry_dict(bonus: models.Bonus, prefix: str) -> dict:
    """
    Create a bonus question entry dictionary from a Bonus model.

    Args:
        bonus: The Bonus model instance
        prefix: Prefix for the QID

    Returns:
        Dictionary containing bonus question data
    """
    leadin = acf_sanitization.sanitize_question(bonus.leadin)
    q_info = bonus.question
    pq = q_info.packet_questions[0]
    qset = q_info.question_set_edition.question_set

    qid = f"{prefix}-b-{pq.packet_id:02d}-{pq.question_number:02d}"

    parts = []
    for part in bonus.bonus_parts:
        part_text = acf_sanitization.sanitize_answer(part.part)
        answers = acf_sanitization.get_short_clean_answers(part.answer)

        parts.append(
            {
                "number": part.part_number,
                "question": part_text,
                "answer_line": part.answer,
                "answer_primary": part.answer_primary,
                "clean_answers": list(answers["clean"]),
                "explanation": answers["explanation"],
                "value": part.value,
                "difficulty_modifier": part.difficulty_modifier,
            }
        )

    return {
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
    }


def create_player_entries(
    session,
    existing_player_slugs: set[str] | None = None,
) -> tuple[list[dict], dict[int, str]]:
    """
    Query all Players and create player entries.

    Players are unique across tournaments - their ID is based on their slug only,
    not on tournament. This means the same player appearing in multiple tournaments
    will have the same player_id.

    Args:
        session: SQLAlchemy session
        existing_player_slugs: Set of player slugs already processed (for deduplication)

    Returns:
        Tuple of:
        - List of player dictionaries (only new players not in existing_player_slugs)
        - Mapping from player.id to player_id string
    """
    if existing_player_slugs is None:
        existing_player_slugs = set()

    players = session.query(models.Player).all()
    entries = []
    player_id_map = {}  # player.id -> player_id

    for player in players:
        # Player ID is based on slug only (tournament-independent)
        player_id = f"p-{player.slug}"
        player_id_map[player.id] = player_id

        # Only add entry if we haven't seen this player before
        if player.slug not in existing_player_slugs:
            entries.append(
                {
                    "player_id": player_id,
                    "name": player.name,
                    "slug": player.slug,
                }
            )
            existing_player_slugs.add(player.slug)

    return entries, player_id_map


def create_team_entries(
    session,
    db_prefix: str,
    player_id_map: dict[int, str] | None = None,
) -> tuple[list[dict], dict[int, str]]:
    """
    Query all Teams and create team entries with player hash.

    Teams are tournament-specific but include a player_hash to identify
    teams with the same set of players across tournaments.

    Args:
        session: SQLAlchemy session
        db_prefix: Prefix for unique IDs
        player_id_map: Mapping from player.id to player_id (optional)

    Returns:
        Tuple of:
        - List of team dictionaries
        - Mapping from team.id to team_id string
    """
    teams = session.query(models.Team).all()
    entries = []
    team_id_map = {}  # team.id -> team_id

    for team in teams:
        tournament = team.tournament
        team_id = f"{db_prefix}-tm-{team.id}"
        team_id_map[team.id] = team_id

        # Get player slugs for this team and compute hash
        player_slugs = [player.slug for player in team.players]
        player_hash = compute_team_player_hash(player_slugs)

        # Get player IDs for this team
        if player_id_map:
            player_ids = [player_id_map.get(p.id, f"p-{p.slug}") for p in team.players]
        else:
            player_ids = [f"p-{p.slug}" for p in team.players]
        player_ids.sort()

        entries.append(
            {
                "team_id": team_id,
                "tournament_id": f"{db_prefix}-tour-{tournament.id}",
                "tournament_slug": tournament.slug,
                "tournament_name": tournament.name,
                "name": team.name,
                "slug": team.slug,
                "hash": player_hash,
                "players": player_ids,
            }
        )

    return entries, team_id_map


def create_tossup_entries(
    session,
    db_prefix: str,
    include_buzz_positions: bool = True,
) -> tuple[list[dict], dict[int, str]]:
    """
    Create tossup question entries from the database.

    Args:
        session: SQLAlchemy session
        db_prefix: Prefix for unique IDs
        include_buzz_positions: Whether to include human buzz positions in metadata

    Returns:
        Tuple of (list of tossup dictionaries, mapping from tossup_id to qid)
    """
    tossups = session.query(models.Tossup).all()
    entries = []
    tossup_qid_map = {}  # tossup.id -> qid

    for tossup in tossups:
        entry = create_tossup_entry_dict(tossup, db_prefix, include_buzz_positions)
        tossup_qid_map[tossup.id] = entry["qid"]
        entries.append(entry)

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
        entry = create_bonus_entry_dict(bonus, db_prefix)

        # Map bonus part IDs to the bonus qid
        for part in bonus.bonus_parts:
            bonus_part_qid_map[part.id] = entry["qid"]

        entries.append(entry)

    return entries, bonus_part_qid_map


def create_buzz_point_entries(
    session,
    db_prefix: str,
    tossup_qid_map: dict[int, str],
    player_id_map: dict[int, str] | None = None,
    team_id_map: dict[int, str] | None = None,
) -> list[dict]:
    """
    Create buzz point entries from the database.

    Args:
        session: SQLAlchemy session
        db_prefix: Prefix for unique IDs
        tossup_qid_map: Mapping from tossup_id to qid
        player_id_map: Mapping from player.id to player_id (optional)
        team_id_map: Mapping from team.id to team_id (optional)

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
        player = buzz.player

        # Use mappings if provided, otherwise construct IDs
        if player_id_map:
            player_id = player_id_map.get(buzz.player_id, f"p-{player.slug}")
        else:
            player_id = f"p-{player.slug}"

        if team_id_map:
            default_team_id = f"{db_prefix}-tm-{player.team_id}"
            team_id = team_id_map.get(player.team_id, default_team_id)
        else:
            team_id = f"{db_prefix}-tm-{player.team_id}"

        entries.append(
            {
                "buzz_id": f"{db_prefix}-bz-{buzz.id}",
                "qid": tossup_qid_map.get(
                    buzz.tossup_id, f"{db_prefix}-t-unknown-{buzz.tossup_id}"
                ),
                "game_id": f"{db_prefix}-g-{buzz.game_id}",
                "team_id": team_id,
                "player_id": player_id,
                "buzz_number": buzz_numbers[buzz.id],
                "correctness": buzz.value > 0,
                "value": buzz.value,
                "token_position": buzz.buzz_position,
            }
        )

    return entries


def create_bonus_response_entries(
    session,
    db_prefix: str,
    bonus_part_qid_map: dict[int, str],
    team_id_map: dict[int, str] | None = None,
) -> list[dict]:
    """
    Create bonus response entries from the database.

    Args:
        session: SQLAlchemy session
        db_prefix: Prefix for unique IDs
        bonus_part_qid_map: Mapping from bonus_part_id to bonus qid
        team_id_map: Mapping from team.id to team_id (optional)

    Returns:
        List of bonus response dictionaries
    """
    bonus_part_directs = session.query(models.BonusPartDirect).all()
    entries = []

    for bpd in bonus_part_directs:
        # Use mapping if provided, otherwise construct ID
        if team_id_map:
            team_id = team_id_map.get(bpd.team_id, f"{db_prefix}-tm-{bpd.team_id}")
        else:
            team_id = f"{db_prefix}-tm-{bpd.team_id}"

        entries.append(
            {
                "response_id": f"{db_prefix}-br-{bpd.id}",
                "bonus_part_id": f"{db_prefix}-bp-{bpd.bonus_part_id}",
                "bonus_qid": bonus_part_qid_map.get(
                    bpd.bonus_part_id, f"{db_prefix}-b-unknown-{bpd.bonus_part_id}"
                ),
                "team_id": team_id,
                "game_id": f"{db_prefix}-g-{bpd.game_id}",
                "correctness": bpd.value > 0,
                "value": bpd.value,
            }
        )

    return entries


def load_all_tables_from_db(
    db_path: str,
    prefix: str,
    existing_player_slugs: set[str] | None = None,
    include_tossups: bool = True,
    include_bonuses: bool = True,
) -> dict[str, list[dict]]:
    """
    Load all 6 table types from a single database.

    Args:
        db_path: Path to the SQLite database
        prefix: Prefix for unique IDs
        existing_player_slugs: Set of player slugs already processed (for deduplication)

    Returns:
        Dictionary mapping table name to list of records
    """
    session = models.create_session(db_path)

    # Create player entries first to get the mapping
    player_entries, player_id_map = create_player_entries(
        session, existing_player_slugs
    )

    # Create team entries with player hash
    team_entries, team_id_map = create_team_entries(session, prefix, player_id_map)

    tables = {"players": player_entries, "teams": team_entries}

    # Create question entries with mappings for cross-references
    if include_tossups:
        tossup_entries, tossup_qid_map = create_tossup_entries(session, prefix)
        tossup_responses = create_buzz_point_entries(
            session, prefix, tossup_qid_map, player_id_map, team_id_map
        )
        tables |= {
            "tossup_questions": tossup_entries,
            "tossup_responses": tossup_responses,
        }
    if include_bonuses:
        bonus_entries, bonus_part_qid_map = create_bonus_entries(session, prefix)
        bonus_responses = create_bonus_response_entries(
            session, prefix, bonus_part_qid_map, team_id_map
        )
        tables |= {"bonus_questions": bonus_entries, "bonus_responses": bonus_responses}

    return tables
