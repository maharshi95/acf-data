#!/usr/bin/env python3
"""Build the 2023--2025 ACF Hugging Face dataset from the merged database.

Player identity is resolved from ``data/acf-23-25-team-lookup.csv`` before any
dataset rows are built.  Consequently, player rows, team rosters and hashes,
and tossup responses all use the same resolved player slugs. Each table is a
Hugging Face config with ``2023``, ``2024``, ``2025``, and ``full`` splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import acf_sanitization, qb_tokenization


DEFAULT_DB = ROOT / "data/dbs/acf-co-23-25.db"
DEFAULT_LOOKUP = ROOT / "data/acf-23-25-team-lookup.csv"
DEFAULT_OUTPUT = ROOT / "data/hf/acf-23-25"
DATASET_DOCUMENTATION = ROOT / "docs/acf-23-25-dataset.md"
TABLES = (
    "players",
    "tournament",
    "teams",
    "games",
    "tossup_questions",
    "bonus_questions",
    "question_placements",
    "tossup_responses",
    "bonus_responses",
)
YEARS = ("2023", "2024", "2025")
SPLITS = (*YEARS, "full")


@dataclass(frozen=True)
class LookupRow:
    id: int
    team_id: int
    name: str
    slug: str
    person_id: str | None

    @property
    def identity_key(self) -> str:
        return f"person:{self.person_id}" if self.person_id else f"unmatched:{self.id}"


@dataclass
class IdentityResolution:
    player_id_by_db_id: dict[int, str]
    slug_by_db_id: dict[int, str]
    players: list[dict[str, Any]]
    mapping_rows: list[dict[str, Any]]
    inconsistencies: dict[str, list[dict[str, Any]]]


def read_lookup(path: Path) -> list[LookupRow]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "team_id", "name", "slug", "person_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Lookup is missing required columns: {missing}")
        rows = []
        for raw in reader:
            person_id = (raw["person_id"] or "").strip()
            rows.append(
                LookupRow(
                    id=int(raw["id"]),
                    team_id=int(raw["team_id"]),
                    name=raw["name"].strip(),
                    slug=raw["slug"].strip(),
                    person_id=None if not person_id or person_id.upper() == "NA" else person_id,
                )
            )
    return rows


def validate_lookup(
    connection: sqlite3.Connection, rows: list[LookupRow]
) -> list[dict[str, Any]]:
    """Return hard lookup/database contract violations."""
    db_rows = {
        row["id"]: row
        for row in connection.execute(
            "SELECT id, team_id, name, slug FROM player ORDER BY id"
        )
    }
    lookup_rows = {row.id: row for row in rows}
    violations: list[dict[str, Any]] = []
    duplicate_ids = sorted(row_id for row_id, n in Counter(r.id for r in rows).items() if n > 1)
    if duplicate_ids:
        violations.append({"code": "duplicate_lookup_ids", "ids": duplicate_ids})
    if missing := sorted(set(db_rows) - set(lookup_rows)):
        violations.append({"code": "players_missing_from_lookup", "ids": missing})
    if extra := sorted(set(lookup_rows) - set(db_rows)):
        violations.append({"code": "lookup_players_missing_from_db", "ids": extra})
    for player_id in sorted(set(db_rows) & set(lookup_rows)):
        db_row = db_rows[player_id]
        lookup_row = lookup_rows[player_id]
        mismatches = {
            field: {"lookup": getattr(lookup_row, field), "database": db_row[field]}
            for field in ("team_id", "name", "slug")
            if getattr(lookup_row, field) != db_row[field]
        }
        if mismatches:
            violations.append(
                {"code": "lookup_row_mismatch", "player_db_id": player_id, "fields": mismatches}
            )
    return violations


NUMERIC_SLUG_SUFFIX_RE = re.compile(r"^(?P<base>.+)-(?P<number>[1-9]\d*)$")


def _alphabetical(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def assign_missing_person_ids(
    rows: list[LookupRow], first_person_id: int = 50_000
) -> tuple[list[LookupRow], list[dict[str, Any]]]:
    """Inject stable person IDs for unmatched rows in alphabetical order."""
    used_ids = {int(row.person_id) for row in rows if row.person_id}
    missing = sorted(
        (row for row in rows if not row.person_id),
        key=lambda row: (
            _alphabetical(row.name),
            row.slug.casefold(),
            row.team_id,
            row.id,
        ),
    )
    assigned_by_row_id = {}
    next_person_id = first_person_id
    for row in missing:
        while next_person_id in used_ids:
            next_person_id += 1
        assigned_by_row_id[row.id] = str(next_person_id)
        used_ids.add(next_person_id)
        next_person_id += 1
    resolved_rows = [
        replace(row, person_id=assigned_by_row_id[row.id])
        if row.id in assigned_by_row_id
        else row
        for row in rows
    ]
    assignments = [
        {
            "lookup_row_id": row.id,
            "name": row.name,
            "source_slug": row.slug,
            "assigned_person_id": assigned_by_row_id[row.id],
        }
        for row in missing
    ]
    return resolved_rows, assignments


def _slug_preferences(rows: list[LookupRow]) -> list[str]:
    """Rank aliases by base-slug convention, then descriptive length."""
    slugs = {row.slug for row in rows}
    numeric_bases = {
        match.group("base")
        for slug in slugs
        if (match := NUMERIC_SLUG_SUFFIX_RE.match(slug))
        and match.group("base") in slugs
    }
    preferred_bases = sorted(numeric_bases, key=lambda slug: (-len(slug), slug))
    remaining = sorted(
        slugs - set(preferred_bases), key=lambda slug: (-len(slug), slug)
    )
    return preferred_bases + remaining


def _identity_sort_key(key: str) -> tuple[int, str]:
    person_id = key.split(":", 1)[1]
    return (int(person_id), person_id)


def _canonical_name(rows: list[LookupRow], canonical_slug: str) -> str:
    preferred = [row.name for row in rows if row.slug == canonical_slug]
    names = Counter(preferred or [row.name for row in rows])
    return min(names, key=lambda name: (-names[name], -len(name), name.casefold()))


def resolve_identities(rows: list[LookupRow]) -> IdentityResolution:
    rows, missing_person_id_assignments = assign_missing_person_ids(rows)
    groups: dict[str, list[LookupRow]] = defaultdict(list)
    for row in rows:
        groups[row.identity_key].append(row)

    # First preserve slugs belonging to single-alias people. Multi-alias people
    # then choose the best available original alias, avoiding needless suffixes.
    candidates = {
        key: _slug_preferences(group)[0]
        for key, group in groups.items()
        if len({row.slug for row in group}) == 1
    }
    taken_candidates = set(candidates.values())
    multi_alias_groups = sorted(
        (
            (key, group)
            for key, group in groups.items()
            if len({row.slug for row in group}) > 1
        ),
        key=lambda item: (
            _alphabetical(_canonical_name(item[1], _slug_preferences(item[1])[0])),
            _identity_sort_key(item[0]),
        ),
    )
    for key, group in multi_alias_groups:
        preferences = _slug_preferences(group)
        candidate = next(
            (slug for slug in preferences if slug not in taken_candidates),
            preferences[0],
        )
        candidates[key] = candidate
        taken_candidates.add(candidate)

    keys_by_candidate: dict[str, list[str]] = defaultdict(list)
    for key, slug in candidates.items():
        keys_by_candidate[slug].append(key)

    resolved_slugs: dict[str, str] = {}
    used: set[str] = set()
    protected_candidates = set(candidates.values())
    for candidate in sorted(keys_by_candidate):
        keys = sorted(
            keys_by_candidate[candidate],
            key=lambda key: (
                len({row.slug for row in groups[key]}) != 1,
                _identity_sort_key(key),
            ),
        )
        for index, key in enumerate(keys):
            if index == 0 and candidate not in used:
                resolved = candidate
            else:
                serial = 2
                resolved = f"{candidate}-{serial}"
                while resolved in used or resolved in protected_candidates:
                    serial += 1
                    resolved = f"{candidate}-{serial}"
            resolved_slugs[key] = resolved
            used.add(resolved)

    for assignment in missing_person_id_assignments:
        key = f"person:{assignment['assigned_person_id']}"
        assignment["orig-acf-slug"] = candidates[key]
        assignment["resolved_slug"] = resolved_slugs[key]

    same_person_multiple_slugs = []
    name_variants = []
    for key, group in sorted(groups.items()):
        slugs = sorted({row.slug for row in group})
        names = sorted({row.name for row in group}, key=str.casefold)
        person_id = group[0].person_id
        if person_id and len(slugs) > 1:
            same_person_multiple_slugs.append(
                {
                    "person_id": person_id,
                    "source_slugs": slugs,
                    "orig-acf-slug": candidates[key],
                    "resolved_slug": resolved_slugs[key],
                    "lookup_row_ids": sorted(row.id for row in group),
                }
            )
        if person_id and len(names) > 1:
            name_variants.append(
                {
                    "person_id": person_id,
                    "names": names,
                    "resolved_slug": resolved_slugs[key],
                }
            )

    source_slug_people: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_slug_people[row.slug].add(row.identity_key)
    same_slug_multiple_people = []
    for slug, keys in sorted(source_slug_people.items()):
        if len(keys) > 1:
            same_slug_multiple_people.append(
                {
                    "source_slug": slug,
                    "identities": sorted(keys),
                    "resolved_slugs": sorted({resolved_slugs[key] for key in keys}),
                    "lookup_row_ids": sorted(row.id for row in rows if row.slug == slug),
                }
            )

    players = []
    mapping_rows = []
    player_id_by_db_id = {}
    slug_by_db_id = {}
    for key, group in sorted(groups.items(), key=lambda item: min(r.id for r in item[1])):
        canonical_slug = candidates[key]
        resolved_slug = resolved_slugs[key]
        person_id = group[0].person_id
        player_id = f"p-{resolved_slug}"
        players.append(
            {
                "player_id": player_id,
                "name": _canonical_name(group, canonical_slug),
                "slug": resolved_slug,
                "person_id": person_id,
                "orig-acf-slug": canonical_slug,
                "source_slugs": sorted({row.slug for row in group}),
                "source_player_ids": sorted(row.id for row in group),
            }
        )
        for row in sorted(group, key=lambda item: item.id):
            player_id_by_db_id[row.id] = player_id
            slug_by_db_id[row.id] = resolved_slug
            mapping_rows.append(
                {
                    "player_db_id": row.id,
                    "team_db_id": row.team_id,
                    "name": row.name,
                    "source_slug": row.slug,
                    "person_id": person_id,
                    "orig-acf-slug": canonical_slug,
                    "resolved_slug": resolved_slug,
                    "player_id": player_id,
                }
            )

    return IdentityResolution(
        player_id_by_db_id=player_id_by_db_id,
        slug_by_db_id=slug_by_db_id,
        players=players,
        mapping_rows=sorted(mapping_rows, key=lambda row: row["player_db_id"]),
        inconsistencies={
            "same_person_multiple_slugs": same_person_multiple_slugs,
            "same_slug_multiple_people": same_slug_multiple_people,
            "missing_person_id": missing_person_id_assignments,
            "name_variants_for_person": name_variants,
        },
    )


def _team_hash(slugs: Iterable[str]) -> str:
    return hashlib.md5("|".join(sorted(set(slugs))).encode()).hexdigest()[:12]


def infer_event_level(question_set_slug: str) -> str:
    slug = (question_set_slug or "").casefold()
    for level in ("nationals", "regionals", "winter", "fall"):
        if f"acf-{level}" in slug:
            return level
    if "open" in slug:
        return "open"
    return "unknown"


def normalized_packet_set(question_set_slug: str) -> str:
    match = re.match(r"^(?P<year>20\d{2})-", question_set_slug or "")
    year = match.group("year") if match else "unknown"
    return f"{year}-{infer_event_level(question_set_slug)}"


def infer_dots(question_set_slug: str, difficulty: str | None = None) -> int | None:
    """Return the conventional ACF difficulty tier from one through four dots."""
    level = infer_event_level(question_set_slug)
    by_level = {"fall": 1, "winter": 2, "regionals": 3, "nationals": 4, "open": 4}
    if level in by_level:
        return by_level[level]
    bullet_count = (difficulty or "").count("●")
    if bullet_count:
        return bullet_count
    match = re.search(r"\b([1-4])\s*dots?\b", difficulty or "", re.IGNORECASE)
    return int(match.group(1)) if match else None


def infer_game_type(packet_name: str | None) -> str | None:
    """Use only an explicit stage label at the start of a packet name."""
    match = re.match(
        r"^\s*(prelims?|playoffs?|finals?|play[ -]?in|emergency)\b",
        packet_name or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    label = match.group(1).casefold().replace(" ", "-")
    if label.startswith("prelim"):
        return "prelim"
    if label.startswith("playoff"):
        return "playoff"
    if label.startswith("final"):
        return "final"
    if label in {"play-in", "playin"}:
        return "play-in"
    return "emergency"


def build_tournaments(
    connection: sqlite3.Connection, prefix: str
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    entries = []
    tournament_id_map = {}
    query = """
        SELECT tr.*, qs.slug AS question_set_slug,
               qs.name AS question_set_name, qs.difficulty,
               qs.format AS question_set_format, qs.bonuses,
               qse.slug AS question_set_edition_slug,
               SUBSTR(tr.start_date, 1, 4) AS year
        FROM tournament tr
        JOIN question_set_edition qse ON qse.id = tr.question_set_edition_id
        JOIN question_set qs ON qs.id = qse.question_set_id
        ORDER BY tr.id
    """
    for row in connection.execute(query):
        tournament_id = f"{prefix}-tour-{row['id']}"
        tournament_id_map[row["id"]] = tournament_id
        entries.append(
            {
                "_year": row["year"],
                "tournament_id": tournament_id,
                "name": row["name"],
                "slug": row["slug"],
                "level": infer_event_level(row["question_set_slug"]),
                "field_level": row["level"],
                "location": row["location"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "packet_set": normalized_packet_set(row["question_set_slug"]),
                "packet_set_name": row["question_set_name"],
                "difficulty": row["difficulty"],
                "dots": infer_dots(row["question_set_slug"], row["difficulty"]),
                "format": row["question_set_format"],
                "has_bonuses": bool(row["bonuses"]),
                "question_set": row["question_set_slug"],
                "question_set_edition": row["question_set_edition_slug"],
            }
        )
    return entries, tournament_id_map


def build_teams(
    connection: sqlite3.Connection, prefix: str, identities: IdentityResolution
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    player_rows_by_team: dict[int, list[int]] = defaultdict(list)
    for row in connection.execute("SELECT id, team_id FROM player ORDER BY id"):
        player_rows_by_team[row["team_id"]].append(row["id"])
    teams = []
    team_id_map = {}
    years_by_player_id: dict[str, set[str]] = defaultdict(set)
    query = """
        SELECT tm.id, tm.name, tm.slug, tr.id AS tournament_db_id,
               tr.name AS tournament_name, tr.slug AS tournament_slug,
               SUBSTR(tr.start_date, 1, 4) AS year
        FROM team tm JOIN tournament tr ON tr.id = tm.tournament_id
        ORDER BY tm.id
    """
    for row in connection.execute(query):
        team_id = f"{prefix}-tm-{row['id']}"
        team_id_map[row["id"]] = team_id
        db_player_ids = player_rows_by_team[row["id"]]
        player_ids = sorted({identities.player_id_by_db_id[player_id] for player_id in db_player_ids})
        slugs = {identities.slug_by_db_id[player_id] for player_id in db_player_ids}
        for player_id in player_ids:
            years_by_player_id[player_id].add(row["year"])
        teams.append(
            {
                "_year": row["year"],
                "team_id": team_id,
                "tournament_id": f"{prefix}-tour-{row['tournament_db_id']}",
                "tournament_slug": row["tournament_slug"],
                "tournament_name": row["tournament_name"],
                "name": row["name"],
                "slug": row["slug"],
                "hash": _team_hash(slugs),
                "players": player_ids,
            }
        )
    for player in identities.players:
        player["_years"] = sorted(years_by_player_id[player["player_id"]])
    return teams, team_id_map


def build_games(
    connection: sqlite3.Connection,
    prefix: str,
    team_id_map: dict[int, str],
    tournament_id_map: dict[int, str],
) -> list[dict[str, Any]]:
    entries = []
    query = """
        SELECT g.id, g.tossups_read, g.team_one_id, g.team_two_id,
               r.id AS round_db_id, r.number AS round_number,
               r.exclude_from_individual, p.id AS packet_db_id,
               p.name AS packet_name, p.descriptor AS packet_descriptor,
               tr.id AS tournament_db_id, tr.name AS tournament_name,
               tr.slug AS tournament_slug, tr.level AS field_level,
               qs.slug AS question_set_slug,
               SUBSTR(tr.start_date, 1, 4) AS year
        FROM game g
        JOIN round r ON r.id = g.round_id
        JOIN tournament tr ON tr.id = r.tournament_id
        JOIN packet p ON p.id = r.packet_id
        JOIN question_set_edition qse ON qse.id = tr.question_set_edition_id
        JOIN question_set qs ON qs.id = qse.question_set_id
        ORDER BY g.id
    """
    for row in connection.execute(query):
        entries.append(
            {
                "_year": row["year"],
                "game_id": f"{prefix}-g-{row['id']}",
                "tournament_id": tournament_id_map[row["tournament_db_id"]],
                "tournament_name": row["tournament_name"],
                "tournament_slug": row["tournament_slug"],
                "level": infer_event_level(row["question_set_slug"]),
                "field_level": row["field_level"],
                "team_a": team_id_map[row["team_one_id"]],
                "team_b": team_id_map[row["team_two_id"]],
                "round": row["round_number"],
                "round_id": f"{prefix}-r-{row['round_db_id']}",
                "type": infer_game_type(row["packet_name"]),
                "packet_id": f"{prefix}-p-{row['packet_db_id']}",
                "packet_name": row["packet_name"],
                "packet_descriptor": row["packet_descriptor"],
                "tossups_read": row["tossups_read"],
                "exclude_from_individual": bool(row["exclude_from_individual"]),
            }
        )
    return entries


QUESTION_QUERY = """
    SELECT body.id, q.id AS question_id, body.question AS question_text,
           body.answer, body.answer_sanitized, body.answer_primary,
           q.category_slug, q.subcategory_slug, q.category_main_slug,
           q.category_full, pq.packet_id, pq.question_number,
           p.name AS packet_name, qs.difficulty, qs.slug AS question_set_slug,
           SUBSTR(qse.date, 1, 4) AS year
    FROM tossup body
    JOIN question q ON q.id = body.question_id
    JOIN packet_question pq ON pq.id = (
        SELECT MIN(pq2.id) FROM packet_question pq2 WHERE pq2.question_id = q.id
    )
    JOIN packet p ON p.id = pq.packet_id
    JOIN question_set_edition qse ON qse.id = p.question_set_edition_id
    JOIN question_set qs ON qs.id = qse.question_set_id
    ORDER BY body.id
"""


def _metadata(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "category": row["category_slug"],
        "subcategory": [row["subcategory_slug"]],
        "category_main": row["category_main_slug"],
        "category_full": row["category_full"],
        "difficulty": (row["difficulty"] or "").split()[0],
        "question_set": row["question_set_slug"],
        "packet": row["packet_name"],
    }


def build_tossups(
    connection: sqlite3.Connection, prefix: str, include_buzz_positions: bool
) -> tuple[
    list[dict[str, Any]],
    dict[int, str],
    dict[str, list[dict[str, Any]]],
]:
    buzz_positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    if include_buzz_positions:
        for row in connection.execute(
            "SELECT tossup_id, buzz_position, value FROM buzz ORDER BY tossup_id, buzz_position"
        ):
            buzz_positions[row["tossup_id"]].append((row["buzz_position"], row["value"]))
    entries = []
    qid_map = {}
    tokenization_fallbacks = []
    invalid_buzz_positions = []
    for row in connection.execute(QUESTION_QUERY):
        qid = f"{prefix}-t-{row['packet_id']:02d}-{row['question_number']:02d}"
        question = acf_sanitization.sanitize_question(row["question_text"])
        answers = acf_sanitization.get_short_clean_answers(row["answer"])
        metadata = _metadata(row)
        if include_buzz_positions:
            offset = acf_sanitization.get_buzz_offset(row["question_text"])
            normalized_positions = []
            for position, value in buzz_positions[row["id"]]:
                if not isinstance(position, int):
                    invalid_buzz_positions.append(
                        {
                            "tossup_db_id": row["id"],
                            "qid": qid,
                            "buzz_position": position,
                        }
                    )
                    continue
                normalized_positions.append((position - offset, value))
            metadata["human_buzz_positions"] = sorted(normalized_positions)
        qid_map[row["id"]] = qid
        try:
            clue_spans = qb_tokenization.get_clue_spans(
                question, tokenization_scheme="blingfire"
            )
        except (RuntimeError, ValueError) as error:
            # Preserve the question and make it usable as one clue while making
            # the malformed source punctuation visible in the build report.
            clue_spans = [(0, len(question))]
            tokenization_fallbacks.append(
                {"tossup_db_id": row["id"], "qid": qid, "error": str(error)}
            )
        entries.append(
            {
                "qid": qid,
                "question": question,
                "answer_line": row["answer"],
                "answer_sanitized": row["answer_sanitized"],
                "answer_primary": row["answer_primary"],
                "clean_answers": list(answers["clean"]),
                "explanation": answers["explanation"],
                "clue_spans": clue_spans,
                "metadata": metadata,
            }
        )
    return entries, qid_map, {
        "question_tokenization_fallbacks": tokenization_fallbacks,
        "invalid_human_buzz_positions": invalid_buzz_positions,
    }


BONUS_QUERY = QUESTION_QUERY.replace(
    "body.question AS question_text,\n           body.answer, body.answer_sanitized, body.answer_primary,",
    "body.leadin,",
).replace("FROM tossup body", "FROM bonus body")


def build_bonuses(
    connection: sqlite3.Connection, prefix: str
) -> tuple[list[dict[str, Any]], dict[int, str], dict[int, str]]:
    parts_by_bonus: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for part in connection.execute("SELECT * FROM bonus_part ORDER BY bonus_id, part_number, id"):
        parts_by_bonus[part["bonus_id"]].append(part)
    entries = []
    part_qid_map = {}
    bonus_qid_map = {}
    for row in connection.execute(BONUS_QUERY):
        qid = f"{prefix}-b-{row['packet_id']:02d}-{row['question_number']:02d}"
        bonus_qid_map[row["id"]] = qid
        parts = []
        for part in parts_by_bonus[row["id"]]:
            answers = acf_sanitization.get_short_clean_answers(part["answer"])
            parts.append(
                {
                    "number": part["part_number"],
                    "question": acf_sanitization.sanitize_answer(part["part"]),
                    "answer_line": part["answer"],
                    "answer_primary": part["answer_primary"],
                    "clean_answers": list(answers["clean"]),
                    "explanation": answers["explanation"],
                    "value": part["value"],
                    "difficulty_modifier": part["difficulty_modifier"],
                }
            )
            part_qid_map[part["id"]] = qid
        entries.append(
            {
                "qid": qid,
                "leadin": acf_sanitization.sanitize_question(row["leadin"]),
                "parts": parts,
                "metadata": _metadata(row),
            }
        )
    return entries, part_qid_map, bonus_qid_map


def build_question_placements(
    connection: sqlite3.Connection,
    prefix: str,
    tossup_qid_map: dict[int, str],
    bonus_qid_map: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    """Build the edition-specific packet placements for canonical questions."""
    entries = []
    years_by_qid: dict[str, set[str]] = defaultdict(set)
    query = """
        SELECT pq.id, pq.question_id, pq.question_number,
               p.id AS packet_db_id, p.name AS packet_name,
               p.descriptor AS packet_descriptor,
               qse.id AS edition_db_id, qse.name AS edition_name,
               qse.slug AS edition_slug, qse.date AS edition_date,
               qs.slug AS question_set_slug,
               t.id AS tossup_id, b.id AS bonus_id,
               SUBSTR(qse.date, 1, 4) AS year,
               pq.id = (
                   SELECT MIN(pq2.id) FROM packet_question pq2
                   WHERE pq2.question_id = pq.question_id
               ) AS is_canonical
        FROM packet_question pq
        JOIN packet p ON p.id = pq.packet_id
        JOIN question_set_edition qse ON qse.id = p.question_set_edition_id
        JOIN question_set qs ON qs.id = qse.question_set_id
        LEFT JOIN tossup t ON t.question_id = pq.question_id
        LEFT JOIN bonus b ON b.question_id = pq.question_id
        ORDER BY pq.id
    """
    for row in connection.execute(query):
        if row["tossup_id"] is not None:
            question_type = "tossup"
            qid = tossup_qid_map[row["tossup_id"]]
        else:
            question_type = "bonus"
            qid = bonus_qid_map[row["bonus_id"]]
        years_by_qid[qid].add(row["year"])
        entries.append(
            {
                "_year": row["year"],
                "placement_id": f"{prefix}-qp-{row['id']}",
                "qid": qid,
                "question_type": question_type,
                "question_set": row["question_set_slug"],
                "question_set_edition": row["edition_slug"],
                "question_set_edition_name": row["edition_name"],
                "question_set_edition_date": row["edition_date"],
                "packet_id": f"{prefix}-p-{row['packet_db_id']}",
                "packet_name": row["packet_name"],
                "packet_descriptor": row["packet_descriptor"],
                "question_number": row["question_number"],
                "is_canonical_qid_placement": bool(row["is_canonical"]),
            }
        )
    return entries, years_by_qid


def build_tossup_responses(
    connection: sqlite3.Connection,
    prefix: str,
    identities: IdentityResolution,
    team_id_map: dict[int, str],
    tossup_qid_map: dict[int, str],
) -> list[dict[str, Any]]:
    entries = []
    last_key = None
    buzz_number = 0
    query = """
        SELECT b.*, p.team_id, SUBSTR(tr.start_date, 1, 4) AS year
        FROM buzz b
        JOIN player p ON p.id = b.player_id
        JOIN game g ON g.id = b.game_id
        JOIN round r ON r.id = g.round_id
        JOIN tournament tr ON tr.id = r.tournament_id
        ORDER BY b.game_id, b.tossup_id, b.buzz_position, b.id
    """
    for row in connection.execute(query):
        key = (row["game_id"], row["tossup_id"])
        buzz_number = buzz_number + 1 if key == last_key else 1
        last_key = key
        value = row["value"]
        numeric_value = value if isinstance(value, int) else None
        token_position = row["buzz_position"]
        numeric_position = token_position if isinstance(token_position, int) else None
        entries.append(
            {
                "_year": row["year"],
                "buzz_id": f"{prefix}-bz-{row['id']}",
                "qid": tossup_qid_map[row["tossup_id"]],
                "game_id": f"{prefix}-g-{row['game_id']}",
                "team_id": team_id_map[row["team_id"]],
                "player_id": identities.player_id_by_db_id[row["player_id"]],
                "buzz_number": buzz_number,
                "correctness": numeric_value is not None and numeric_value > 0,
                "value": numeric_value,
                "token_position": numeric_position,
            }
        )
    return entries


def build_bonus_responses(
    connection: sqlite3.Connection,
    prefix: str,
    team_id_map: dict[int, str],
    part_qid_map: dict[int, str],
) -> list[dict[str, Any]]:
    entries = []
    query = """
        SELECT d.*, SUBSTR(tr.start_date, 1, 4) AS year
        FROM bonus_part_direct d
        JOIN game g ON g.id = d.game_id
        JOIN round r ON r.id = g.round_id
        JOIN tournament tr ON tr.id = r.tournament_id
        ORDER BY d.id
    """
    for row in connection.execute(query):
        numeric_value, correctness, is_scored = normalize_bonus_value(row["value"])
        entries.append({
            "_year": row["year"],
            "response_id": f"{prefix}-br-{row['id']}",
            "bonus_part_id": f"{prefix}-bp-{row['bonus_part_id']}",
            "bonus_qid": part_qid_map[row["bonus_part_id"]],
            "team_id": team_id_map[row["team_id"]],
            "game_id": f"{prefix}-g-{row['game_id']}",
            "correctness": correctness,
            "value": numeric_value,
            "is_scored": is_scored,
        })
    return entries


def normalize_bonus_value(value: Any) -> tuple[int | None, bool | None, bool]:
    """Normalize a scored 0/10 response or an unscored source placeholder."""
    if isinstance(value, int):
        return value, value > 0, True
    return None, None, False


def database_checks(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    checks = []
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    checks.append({"name": "sqlite_integrity", "status": "pass" if integrity == "ok" else "fail", "detail": integrity})
    foreign_keys = [dict(row) for row in connection.execute("PRAGMA foreign_key_check")]
    checks.append({"name": "foreign_keys", "status": "pass" if not foreign_keys else "fail", "count": len(foreign_keys), "examples": foreign_keys[:20]})
    cross_edition_sharing = [dict(row) for row in connection.execute("""
        SELECT pq.question_id, COUNT(*) AS placements,
               COUNT(DISTINCT p.question_set_edition_id) AS editions
        FROM packet_question pq JOIN packet p ON p.id = pq.packet_id
        GROUP BY pq.question_id
        HAVING COUNT(DISTINCT p.question_set_edition_id) > 1
        ORDER BY pq.question_id
    """)]
    checks.append({
        "name": "shared_question_across_editions",
        "status": "info" if cross_edition_sharing else "pass",
        "count": len(cross_edition_sharing),
        "examples": cross_edition_sharing[:20],
    })
    cross_edition_mismatches = [dict(row) for row in connection.execute("""
        SELECT pq.question_id, COUNT(DISTINCT p.descriptor) AS packet_descriptors,
               COUNT(DISTINCT pq.question_number) AS question_numbers
        FROM packet_question pq JOIN packet p ON p.id = pq.packet_id
        GROUP BY pq.question_id
        HAVING COUNT(DISTINCT p.question_set_edition_id) > 1
           AND (COUNT(DISTINCT p.descriptor) > 1
                OR COUNT(DISTINCT pq.question_number) > 1)
        ORDER BY pq.question_id
    """)]
    checks.append({
        "name": "cross_edition_question_position_mismatch",
        "status": "warning" if cross_edition_mismatches else "pass",
        "count": len(cross_edition_mismatches),
        "examples": cross_edition_mismatches[:20],
    })
    within_edition_reuse = [dict(row) for row in connection.execute("""
        SELECT pq.question_id, p.question_set_edition_id,
               COUNT(*) AS placements,
               GROUP_CONCAT(p.id || ':' || pq.question_number) AS packet_slots
        FROM packet_question pq JOIN packet p ON p.id = pq.packet_id
        GROUP BY pq.question_id, p.question_set_edition_id
        HAVING COUNT(*) > 1
        ORDER BY pq.question_id, p.question_set_edition_id
    """)]
    checks.append({
        "name": "question_reused_within_edition",
        "status": "warning" if within_edition_reuse else "pass",
        "count": len(within_edition_reuse),
        "examples": within_edition_reuse[:20],
    })
    unscored_bonus_parts = [dict(row) for row in connection.execute("""
        SELECT d.id AS response_id, d.game_id, d.team_id,
               d.bonus_part_id, d.value
        FROM bonus_part_direct d
        WHERE d.value IS NULL OR TYPEOF(d.value) != 'integer'
        ORDER BY d.id
    """)]
    checks.append({
        "name": "unscored_bonus_part_values",
        "status": "warning" if unscored_bonus_parts else "pass",
        "count": len(unscored_bonus_parts),
        "examples": unscored_bonus_parts[:20],
    })
    queries = {
        "buzz_player_team_is_in_game": """
            SELECT b.id FROM buzz b JOIN player p ON p.id=b.player_id JOIN game g ON g.id=b.game_id
            WHERE p.team_id NOT IN (g.team_one_id, g.team_two_id)
        """,
        "bonus_response_team_is_in_game": """
            SELECT d.id FROM bonus_part_direct d JOIN game g ON g.id=d.game_id
            WHERE d.team_id NOT IN (g.team_one_id, g.team_two_id)
        """,
        "tossup_values_are_expected": "SELECT id FROM buzz WHERE value NOT IN (-5, 0, 10, 15)",
        "numeric_bonus_values_are_expected": """
            SELECT id FROM bonus_part_direct
            WHERE TYPEOF(value) = 'integer' AND value NOT IN (0, 10)
        """,
    }
    for name, query in queries.items():
        bad_ids = [row[0] for row in connection.execute(query)]
        # Unexpected score values are source-data violations worth surfacing, but
        # they do not break referential integrity. Text placeholders such as NA
        # are normalized to null in the dataset.
        status = "pass" if not bad_ids else (
            "warning" if name.endswith("values_are_expected") else "fail"
        )
        checks.append({"name": name, "status": status, "count": len(bad_ids), "examples": bad_ids[:20]})
    return checks


def dataset_checks(records: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    checks = []
    id_fields = {
        "players": "player_id", "tournament": "tournament_id",
        "teams": "team_id", "games": "game_id", "tossup_questions": "qid",
        "bonus_questions": "qid", "tossup_responses": "buzz_id",
        "question_placements": "placement_id", "bonus_responses": "response_id",
    }
    for table, field in id_fields.items():
        values = [row[field] for row in records[table]]
        duplicate_count = len(values) - len(set(values))
        checks.append({"name": f"unique_{table}_{field}", "status": "pass" if not duplicate_count else "fail", "count": duplicate_count})

    player_ids = {row["player_id"] for row in records["players"]}
    tournament_ids = {row["tournament_id"] for row in records["tournament"]}
    team_ids = {row["team_id"] for row in records["teams"]}
    game_ids = {row["game_id"] for row in records["games"]}
    tossup_qids = {row["qid"] for row in records["tossup_questions"]}
    bonus_qids = {row["qid"] for row in records["bonus_questions"]}
    all_question_qids = tossup_qids | bonus_qids
    references = {
        "team_tournament_references": ({row["tournament_id"] for row in records["teams"]}, tournament_ids),
        "team_player_references": (set(p for row in records["teams"] for p in row["players"]), player_ids),
        "game_tournament_references": ({row["tournament_id"] for row in records["games"]}, tournament_ids),
        "game_team_references": ({team_id for row in records["games"] for team_id in (row["team_a"], row["team_b"])}, team_ids),
        "buzz_player_references": ({row["player_id"] for row in records["tossup_responses"]}, player_ids),
        "buzz_team_references": ({row["team_id"] for row in records["tossup_responses"]}, team_ids),
        "buzz_game_references": ({row["game_id"] for row in records["tossup_responses"]}, game_ids),
        "buzz_question_references": ({row["qid"] for row in records["tossup_responses"]}, tossup_qids),
        "bonus_team_references": ({row["team_id"] for row in records["bonus_responses"]}, team_ids),
        "bonus_game_references": ({row["game_id"] for row in records["bonus_responses"]}, game_ids),
        "bonus_question_references": ({row["bonus_qid"] for row in records["bonus_responses"]}, bonus_qids),
        "placement_question_references": ({row["qid"] for row in records["question_placements"]}, all_question_qids),
    }
    for name, (actual, expected) in references.items():
        missing = sorted(actual - expected)
        checks.append({"name": name, "status": "pass" if not missing else "fail", "count": len(missing), "examples": missing[:20]})
    invalid_bonus_semantics = [
        row["response_id"]
        for row in records["bonus_responses"]
        if (
            row["is_scored"]
            and (row["value"] is None or row["correctness"] is None)
        )
        or (
            not row["is_scored"]
            and (row["value"] is not None or row["correctness"] is not None)
        )
    ]
    checks.append({
        "name": "bonus_response_scoring_semantics",
        "status": "pass" if not invalid_bonus_semantics else "fail",
        "count": len(invalid_bonus_semantics),
        "examples": invalid_bonus_semantics[:20],
    })
    return checks


def partition_records(
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Turn heterogeneous tables into configs with homogeneous year splits."""
    configs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for config_name, rows in records.items():
        split_rows = {split: [] for split in SPLITS}
        for row in rows:
            years = row.get("_years") or [row.get("_year")]
            clean_row = {
                key: value for key, value in row.items() if key not in {"_year", "_years"}
            }
            split_rows["full"].append(clean_row)
            for year in years:
                if year not in YEARS:
                    raise ValueError(
                        f"{config_name} record has unsupported or missing year {year!r}: "
                        f"{next(iter(clean_row.values()), '<empty>')}"
                    )
                split_rows[year].append(clean_row)
        configs[config_name] = split_rows
    return configs


def build_records(
    db_path: Path,
    lookup_path: Path,
    prefix: str,
    include_buzz_positions: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], IdentityResolution, dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        lookup_rows = read_lookup(lookup_path)
        lookup_violations = validate_lookup(connection, lookup_rows)
        if lookup_violations:
            raise ValueError("Lookup/database validation failed:\n" + json.dumps(lookup_violations, indent=2))
        identities = resolve_identities(lookup_rows)
        tournaments, tournament_id_map = build_tournaments(connection, prefix)
        teams, team_id_map = build_teams(connection, prefix, identities)
        games = build_games(
            connection, prefix, team_id_map, tournament_id_map
        )
        tossups, tossup_qid_map, preprocessing_inconsistencies = build_tossups(
            connection, prefix, include_buzz_positions
        )
        bonuses, part_qid_map, bonus_qid_map = build_bonuses(connection, prefix)
        question_placements, years_by_qid = build_question_placements(
            connection, prefix, tossup_qid_map, bonus_qid_map
        )
        for question in tossups + bonuses:
            question["_years"] = sorted(years_by_qid[question["qid"]])
        records = {
            "players": identities.players,
            "tournament": tournaments,
            "teams": teams,
            "games": games,
            "tossup_questions": tossups,
            "bonus_questions": bonuses,
            "question_placements": question_placements,
            "tossup_responses": build_tossup_responses(connection, prefix, identities, team_id_map, tossup_qid_map),
            "bonus_responses": build_bonus_responses(connection, prefix, team_id_map, part_qid_map),
        }
        configs = partition_records(records)
        report = {
            "database": str(db_path),
            "lookup": str(lookup_path),
            "prefix": prefix,
            "table_counts": {table: len(records[table]) for table in TABLES},
            "split_counts": {
                config_name: {
                    split: len(split_rows[split]) for split in SPLITS
                }
                for config_name, split_rows in configs.items()
            },
            "identity_counts": {
                "lookup_rows": len(lookup_rows),
                "resolved_people": len(identities.players),
                **{name: len(items) for name, items in identities.inconsistencies.items()},
            },
            "preprocessing_inconsistencies": preprocessing_inconsistencies,
            "lookup_violations": lookup_violations,
            "checks": database_checks(connection) + dataset_checks(records),
        }
        return records, identities, report
    finally:
        connection.close()


def write_reports(output_dir: Path, identities: IdentityResolution, report: dict[str, Any]) -> None:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "player_identity_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(identities.mapping_rows[0]))
        writer.writeheader()
        writer.writerows(identities.mapping_rows)
    (report_dir / "identity_inconsistencies.json").write_text(
        json.dumps(identities.inconsistencies, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (report_dir / "sanity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = ["# ACF dataset build report", "", "## Table counts", ""]
    lines += [f"- `{name}`: {count:,}" for name, count in report["table_counts"].items()]
    lines += ["", "## Config and split counts", ""]
    lines += [
        f"- `{config_name}`: "
        + ", ".join(f"`{split}` {count:,}" for split, count in split_counts.items())
        for config_name, split_counts in report["split_counts"].items()
    ]
    lines += ["", "## Identity preprocessing", ""]
    lines += [f"- `{name}`: {count:,}" for name, count in report["identity_counts"].items()]
    lines += ["", "## Other preprocessing inconsistencies", ""]
    lines += [
        f"- `{name}`: {len(items):,}"
        for name, items in report["preprocessing_inconsistencies"].items()
    ]
    lines += ["", "## Sanity checks", ""]
    lines += [f"- **{check['status'].upper()}** `{check['name']}`: {check.get('count', check.get('detail', ''))}" for check in report["checks"]]
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if DATASET_DOCUMENTATION.exists():
        (output_dir / "README.md").write_text(
            DATASET_DOCUMENTATION.read_text(encoding="utf-8"), encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--lookup-path", type=Path, default=DEFAULT_LOOKUP)
    parser.add_argument("--prefix", default="acf-23-25")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", help="Optional Hugging Face repository ID to push")
    parser.add_argument("--public", action="store_true", help="Make a pushed dataset public")
    parser.add_argument("--no-save", action="store_true", help="Validate/build without saving config DatasetDicts")
    parser.add_argument("--no-buzz-positions", action="store_true", help="Omit human buzz positions from tossup metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Building from {args.db_path}")
    records, identities, report = build_records(
        args.db_path, args.lookup_path, args.prefix, not args.no_buzz_positions
    )
    write_reports(args.output_dir, identities, report)
    failures = [check for check in report["checks"] if check["status"] == "fail"]
    for table, rows in records.items():
        print(f"  {table}: {len(rows):,}")
    print(f"Reports written to {args.output_dir / 'reports'}")
    if failures:
        names = ", ".join(check["name"] for check in failures)
        raise SystemExit(f"Sanity checks failed: {names}")

    if not args.no_save or args.repo_id:
        from datasets import Dataset, DatasetDict

        configs = partition_records(records)
        for config_name, split_rows in configs.items():
            full_dataset = Dataset.from_list(split_rows["full"])
            dataset = DatasetDict(
                {
                    split: (
                        full_dataset
                        if split == "full"
                        else Dataset.from_list(
                            split_rows[split], features=full_dataset.features
                        )
                    )
                    for split in SPLITS
                }
            )
            if not args.no_save:
                config_dir = args.output_dir / config_name
                dataset.save_to_disk(str(config_dir))
                print(f"Config {config_name} saved to {config_dir}")
            if args.repo_id:
                dataset.push_to_hub(
                    args.repo_id,
                    config_name=config_name,
                    set_default=config_name == TABLES[0],
                    private=not args.public,
                )
                print(f"Config {config_name} pushed to {args.repo_id}")
        write_reports(args.output_dir, identities, report)


if __name__ == "__main__":
    main()
