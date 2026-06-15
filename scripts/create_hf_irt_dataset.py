#!/usr/bin/env python3
"""Create a HuggingFace IRT-style dataset from an ACF tournament database."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

Dataset = None
DatasetDict = None
models = None
acf_sanitization = None
qb_tokenization = None


DEFAULT_DB_PATH = "data/dbs/acf-24-25.db"
DEFAULT_OUTPUT_DIR = "data/hf/quizbowl-irt"
DEFAULT_IDENTITY_OVERRIDES = "data/player_identity_overrides.yml"
DATASET_README_TEMPLATE = Path(__file__).parent.parent / "docs" / "quizbowl-irt-dataset.md"
SUFFIX_RE = re.compile(r"^(?P<base>.+)-(?P<n>[2-9][0-9]*)$")
TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class PlayerOccurrence:
    player_db_id: int
    player_slug: str
    player_name: str
    team_slug: str
    team_name: str
    tournament_slug: str
    tournament_name: str
    start_date: Any
    end_date: Any

    @property
    def ref(self) -> str:
        return f"{self.tournament_slug}::{self.team_slug}::{self.player_slug}"


def normalize_name(name: str) -> str:
    name = (name or "").translate(str.maketrans({"Ł": "L", "ł": "l"}))
    stripped = unicodedata.normalize("NFKD", name or "")
    ascii_name = "".join(c for c in stripped if not unicodedata.combining(c))
    return " ".join(ascii_name.casefold().split())


def strip_numeric_suffix(slug: str) -> str:
    match = SUFFIX_RE.match(slug or "")
    return match.group("base") if match else slug


def is_suffixed_slug(slug: str) -> bool:
    return SUFFIX_RE.match(slug or "") is not None


def slugify(value: str, fallback: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.casefold()).strip("-")
    return value or fallback


def dates_overlap(a: PlayerOccurrence, b: PlayerOccurrence) -> bool:
    a_start = a.start_date or a.end_date
    a_end = a.end_date or a.start_date
    b_start = b.start_date or b.end_date
    b_end = b.end_date or b.start_date
    if not a_start or not a_end or not b_start or not b_end:
        return True
    return not (a_end < b_start or b_end < a_start)


def load_identity_overrides(path: str | Path) -> dict[tuple[str, str, str], str]:
    """Load scoped player identity overrides.

    Preferred YAML shape:

    players:
      - identity_id: p-example
        sources:
          - tournament_slug: 2024-acf-winter-at-oxford
            team_slug: southampton-b
            player_slug: example-2
    """
    path = Path(path)
    if not path.exists():
        return {}

    text = path.read_text()
    if not text.strip():
        return {}

    if path.suffix == ".json":
        data = json.loads(text)
    else:
        data = _load_yamlish(text)

    overrides: dict[tuple[str, str, str], str] = {}
    for player in data.get("players", []) or []:
        identity_id = player.get("identity_id")
        if not identity_id:
            continue
        for source in player.get("sources", []) or []:
            key = (
                source.get("tournament_slug", ""),
                source.get("team_slug", ""),
                source.get("player_slug", ""),
            )
            if all(key):
                overrides[key] = identity_id
    return overrides


def _load_yamlish(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        return _parse_identity_override_yaml_subset(text)


def _parse_identity_override_yaml_subset(text: str) -> dict[str, Any]:
    """Parse the small override YAML shape without requiring PyYAML."""
    players: list[dict[str, Any]] = []
    current_player: dict[str, Any] | None = None
    current_source: dict[str, str] | None = None
    in_sources = False

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or line.strip() == "players:":
            continue
        stripped = line.strip()

        if stripped.startswith("- identity_id:"):
            current_player = {
                "identity_id": _unquote(stripped.split(":", 1)[1].strip()),
                "sources": [],
            }
            players.append(current_player)
            current_source = None
            in_sources = False
            continue
        if stripped == "sources:":
            in_sources = True
            continue
        if in_sources and stripped.startswith("- "):
            if current_player is None:
                continue
            current_source = {}
            current_player["sources"].append(current_source)
            stripped = stripped[2:].strip()
            if stripped:
                key, value = stripped.split(":", 1)
                current_source[key.strip()] = _unquote(value.strip())
            continue
        if current_source is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_source[key.strip()] = _unquote(value.strip())

    return {"players": players}


def _unquote(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def require_datasets() -> None:
    global Dataset, DatasetDict
    if Dataset is not None and DatasetDict is not None:
        return
    try:
        from datasets import Dataset as _Dataset, DatasetDict as _DatasetDict
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise ModuleNotFoundError(
            "The 'datasets' package is required to create HuggingFace datasets. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc
    Dataset = _Dataset
    DatasetDict = _DatasetDict


def require_project_modules() -> None:
    global models, acf_sanitization, qb_tokenization
    if models is not None and acf_sanitization is not None and qb_tokenization is not None:
        return
    try:
        from core import models as _models
        from utils import acf_sanitization as _acf_sanitization
        from utils import qb_tokenization as _qb_tokenization
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise ModuleNotFoundError(
            "Project database dependencies are unavailable. Install them with "
            "`pip install -r requirements.txt` before building the dataset."
        ) from exc
    models = _models
    acf_sanitization = _acf_sanitization
    qb_tokenization = _qb_tokenization


def load_player_occurrences(session) -> list[PlayerOccurrence]:
    occurrences = []
    for player in session.query(models.Player).all():
        team = player.team
        tournament = team.tournament
        occurrences.append(
            PlayerOccurrence(
                player_db_id=player.id,
                player_slug=player.slug or slugify(player.name, f"player-{player.id}"),
                player_name=player.name or "",
                team_slug=team.slug or slugify(team.name, f"team-{team.id}"),
                team_name=team.name or "",
                tournament_slug=tournament.slug
                or slugify(tournament.name, f"tournament-{tournament.id}"),
                tournament_name=tournament.name or "",
                start_date=tournament.start_date,
                end_date=tournament.end_date,
            )
        )
    return occurrences


def build_player_identity_maps(
    occurrences: list[PlayerOccurrence],
    overrides: dict[tuple[str, str, str], str] | None = None,
) -> tuple[dict[int, str], list[dict], list[dict]]:
    overrides = overrides or {}
    by_raw_slug: dict[str, list[PlayerOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        by_raw_slug[occurrence.player_slug].append(occurrence)

    player_id_by_db_id: dict[int, str] = {}
    warnings: list[dict] = []

    for slug, group in by_raw_slug.items():
        grouped_names = {normalize_name(o.player_name) for o in group}
        has_overlap = any(
            dates_overlap(a, b)
            for i, a in enumerate(group)
            for b in group[i + 1 :]
        )
        if len(grouped_names) > 1:
            warnings.append(
                {
                    "type": "raw_slug_name_conflict",
                    "slug": slug,
                    "names": sorted(o.player_name for o in group),
                }
            )
        if has_overlap:
            warnings.append(
                {
                    "type": "raw_slug_date_overlap",
                    "slug": slug,
                    "source_player_refs": sorted(o.ref for o in group),
                }
            )

        for occurrence in group:
            key = (
                occurrence.tournament_slug,
                occurrence.team_slug,
                occurrence.player_slug,
            )
            if key in overrides:
                player_id_by_db_id[occurrence.player_db_id] = overrides[key]
            elif len(grouped_names) == 1 and not has_overlap:
                player_id_by_db_id[occurrence.player_db_id] = f"p-{slug}"
            else:
                player_id_by_db_id[occurrence.player_db_id] = (
                    f"p-{slug}--{occurrence.tournament_slug}--{occurrence.team_slug}"
                )

    by_base_slug: dict[str, list[PlayerOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        by_base_slug[strip_numeric_suffix(occurrence.player_slug)].append(occurrence)

    for base_slug, group in by_base_slug.items():
        variants = sorted({o.player_slug for o in group})
        if len(variants) <= 1:
            continue
        if any(is_suffixed_slug(v) for v in variants):
            warnings.append(
                {
                    "type": "ambiguous_suffixed_slug_group",
                    "base_slug": base_slug,
                    "slugs": variants,
                    "names": sorted({o.player_name for o in group}),
                    "source_player_refs": sorted(o.ref for o in group),
                }
            )

    player_rows_by_id: dict[str, dict] = {}
    occurrences_by_id: dict[str, list[PlayerOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        occurrences_by_id[player_id_by_db_id[occurrence.player_db_id]].append(occurrence)

    for player_id, group in occurrences_by_id.items():
        sorted_group = sorted(
            group,
            key=lambda o: ((o.start_date or o.end_date).isoformat(), o.ref)
            if (o.start_date or o.end_date)
            else ("", o.ref),
        )
        first = sorted_group[0]
        tournaments = sorted({o.tournament_slug for o in sorted_group})
        years = sorted({o.start_date.year for o in sorted_group if o.start_date})
        player_rows_by_id[player_id] = with_years(
            {
                "player_id": player_id,
                "name": first.player_name,
                "first_seen_tournament": first.tournament_slug,
                "tournaments_played": tournaments,
                "source_slugs": sorted({o.player_slug for o in sorted_group}),
                "source_player_refs": sorted(o.ref for o in sorted_group),
            },
            years,
        )

    return player_id_by_db_id, sorted(player_rows_by_id.values(), key=lambda r: r["player_id"]), warnings


def with_years(row: dict, years: Iterable[int | None]) -> dict:
    row["_split_years"] = sorted({int(y) for y in years if y})
    return row


def clean_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "_split_years"}


def dataset_dict_from_rows(rows: list[dict]) -> DatasetDict:
    require_datasets()
    dataset_dict = DatasetDict()
    dataset_dict["all"] = Dataset.from_list([clean_row(row) for row in rows])
    years = sorted({year for row in rows for year in row.get("_split_years", [])})
    for year in years:
        dataset_dict[f"year_{year}"] = Dataset.from_list(
            [clean_row(row) for row in rows if year in row.get("_split_years", [])]
        )
    return dataset_dict


def event_id_for_question(question: models.Question) -> str:
    return question.question_set_edition.question_set.slug


def packet_question_for(question: models.Question):
    return sorted(
        question.packet_questions,
        key=lambda pq: (pq.packet_id or 0, pq.question_number or 0, pq.id or 0),
    )[0]


def tossup_id_for(tossup: models.Tossup) -> str:
    pq = packet_question_for(tossup.question)
    return f"{event_id_for_question(tossup.question)}::p{pq.packet_id:03d}::q{pq.question_number:02d}"


def bonus_id_for(bonus: models.Bonus) -> str:
    pq = packet_question_for(bonus.question)
    return f"{event_id_for_question(bonus.question)}::p{pq.packet_id:03d}::b{pq.question_number:02d}"


def team_id_for(team: models.Team) -> str:
    tournament_id = team.tournament.slug or slugify(team.tournament.name, f"tournament-{team.tournament_id}")
    team_slug = team.slug or slugify(team.name, f"team-{team.id}")
    return f"{tournament_id}::{team_slug}"


def game_id_for(game: models.Game) -> str:
    tournament = game.round.tournament
    tournament_id = tournament.slug or slugify(tournament.name, f"tournament-{tournament.id}")
    return f"{tournament_id}::game-{game.id}"


def token_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def sentence_boundaries(text: str, spans: list[tuple[int, int]]) -> list[int]:
    if not text:
        return []
    try:
        clue_spans = qb_tokenization.get_clue_spans(text, tokenization_scheme="blingfire")
    except Exception:
        clue_spans = [(0, len(text))]
    boundaries = []
    for _, char_end in clue_spans:
        token_index = sum(1 for _, token_end in spans if token_end <= char_end)
        if token_index:
            boundaries.append(token_index)
    return sorted(set(boundaries))


def question_category(question: models.Question) -> tuple[str, str]:
    return question.category or "", question.subcategory or ""


def create_tournaments(session) -> list[dict]:
    rows = []
    for tournament in session.query(models.Tournament).all():
        qset = tournament.question_set_edition.question_set
        year = tournament.start_date.year if tournament.start_date else None
        packet_ids = {round_.packet_id for round_ in tournament.rounds if round_.packet_id}
        rows.append(
            with_years(
                {
                    "tournament_id": tournament.slug
                    or slugify(tournament.name, f"tournament-{tournament.id}"),
                    "name": tournament.name or "",
                    "tier": infer_tier(qset.slug, qset.name),
                    "year": year or 0,
                    "num_packets": len(packet_ids),
                    "question_set_event_id": qset.slug,
                    "level": tournament.level or "",
                    "location": tournament.location or "",
                    "start_date": tournament.start_date.isoformat()
                    if tournament.start_date
                    else "",
                    "end_date": tournament.end_date.isoformat()
                    if tournament.end_date
                    else "",
                },
                [year],
            )
        )
    return sorted(rows, key=lambda r: (r["year"], r["tournament_id"]))


def infer_tier(qset_slug: str, qset_name: str) -> str:
    slug = qset_slug or slugify(qset_name, "unknown")
    slug = re.sub(r"^\d{4}-acf-", "", slug)
    return slug


def create_teams(session, player_id_by_db_id: dict[int, str]) -> list[dict]:
    rows = []
    for team in session.query(models.Team).all():
        tournament = team.tournament
        year = tournament.start_date.year if tournament.start_date else None
        player_ids = [player_id_by_db_id[player.id] for player in team.players]
        rows.append(
            with_years(
                {
                    "team_id": team_id_for(team),
                    "tournament_id": tournament.slug
                    or slugify(tournament.name, f"tournament-{tournament.id}"),
                    "team_name": team.name or "",
                    "player_ids": player_ids,
                },
                [year],
            )
        )
    return sorted(rows, key=lambda r: r["team_id"])


def create_tossups(session) -> tuple[list[dict], dict[int, str], dict[int, int]]:
    rows = []
    tossup_id_by_db_id = {}
    tossup_qnum_by_db_id = {}
    for tossup in session.query(models.Tossup).all():
        question = tossup.question
        pq = packet_question_for(question)
        qset = question.question_set_edition.question_set
        year = question.question_set_edition.date.year if question.question_set_edition.date else None
        text = acf_sanitization.sanitize_question(tossup.question_text or "")
        spans = token_spans(text)
        tokens = [text[start:end] for start, end in spans]
        tossup_id = tossup_id_for(tossup)
        tossup_id_by_db_id[tossup.id] = tossup_id
        tossup_qnum_by_db_id[tossup.id] = pq.question_number
        category, subcategory = question_category(question)
        rows.append(
            with_years(
                {
                    "tossup_id": tossup_id,
                    "question_set_event_id": qset.slug,
                    "packet": pq.packet_id,
                    "question_num": pq.question_number,
                    "category": category,
                    "subcategory": subcategory,
                    "answer": tossup.answer or "",
                    "tokens": tokens,
                    "sentence_boundaries": sentence_boundaries(text, spans),
                    "question": text,
                },
                [year],
            )
        )
    return sorted(rows, key=lambda r: r["tossup_id"]), tossup_id_by_db_id, tossup_qnum_by_db_id


def create_bonuses(session) -> tuple[list[dict], dict[int, str], dict[int, int], dict[int, int]]:
    rows = []
    bonus_id_by_bonus_db_id = {}
    bonus_id_by_part_db_id = {}
    bonus_qnum_by_bonus_db_id = {}
    for bonus in session.query(models.Bonus).all():
        question = bonus.question
        pq = packet_question_for(question)
        qset = question.question_set_edition.question_set
        year = question.question_set_edition.date.year if question.question_set_edition.date else None
        bonus_id = bonus_id_for(bonus)
        bonus_id_by_bonus_db_id[bonus.id] = bonus_id
        bonus_qnum_by_bonus_db_id[bonus.id] = pq.question_number
        category, subcategory = question_category(question)
        parts = []
        for part in sorted(bonus.bonus_parts, key=lambda p: p.part_number or 0):
            bonus_id_by_part_db_id[part.id] = bonus_id
            parts.append(
                {
                    "text": acf_sanitization.sanitize_question(part.part or ""),
                    "answer": part.answer or "",
                    "difficulty": difficulty_label(part.difficulty_modifier),
                }
            )
        rows.append(
            with_years(
                {
                    "bonus_id": bonus_id,
                    "question_set_event_id": qset.slug,
                    "packet": pq.packet_id,
                    "question_num": pq.question_number,
                    "category": category,
                    "subcategory": subcategory,
                    "leadin": acf_sanitization.sanitize_question(bonus.leadin or ""),
                    "parts": parts,
                },
                [year],
            )
        )
    return (
        sorted(rows, key=lambda r: r["bonus_id"]),
        bonus_id_by_part_db_id,
        bonus_id_by_bonus_db_id,
        bonus_qnum_by_bonus_db_id,
    )


def difficulty_label(value: str | None) -> str:
    labels = {"e": "easy", "m": "medium", "h": "hard"}
    return labels.get((value or "").casefold(), value or "")


def build_game_outputs(
    session,
    player_id_by_db_id: dict[int, str],
    tossup_id_by_db_id: dict[int, str],
    tossup_qnum_by_db_id: dict[int, int],
    bonus_id_by_part_db_id: dict[int, str],
    bonus_qnum_by_bonus_db_id: dict[int, int],
    strict: bool = False,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    game_rows = []
    buzz_rows = []
    bonus_response_rows = []
    warnings = []

    for game in session.query(models.Game).all():
        tournament = game.round.tournament
        year = tournament.start_date.year if tournament.start_date else None
        team_ids_by_db_id = {
            game.team_one_id: team_id_for(game.team_one),
            game.team_two_id: team_id_for(game.team_two),
        }
        scores = {game.team_one_id: 0, game.team_two_id: 0}

        buzzes_by_qnum: dict[int, list[models.Buzz]] = defaultdict(list)
        correct_buzz_by_qnum_team: dict[tuple[int, int], models.Buzz] = {}
        for buzz in game.buzzes:
            qnum = tossup_qnum_by_db_id.get(buzz.tossup_id)
            if qnum is None:
                warnings.append({"type": "buzz_without_question_number", "buzz_id": buzz.id})
                continue
            buzzes_by_qnum[qnum].append(buzz)
            if buzz.value > 0:
                team_id = buzz.player.team_id
                correct_buzz_by_qnum_team.setdefault((qnum, team_id), buzz)

        bonus_groups: dict[tuple[int, int, str], list[models.BonusPartDirect]] = defaultdict(list)
        for bpd in game.bonus_part_directs:
            bonus = bpd.bonus_part.bonus
            qnum = bonus_qnum_by_bonus_db_id.get(bonus.id)
            if qnum is None:
                warnings.append(
                    {"type": "bonus_response_without_question_number", "response_id": bpd.id}
                )
                continue
            bonus_id = bonus_id_by_part_db_id[bpd.bonus_part_id]
            bonus_groups[(qnum, bpd.team_id, bonus_id)].append(bpd)

        qnums = sorted(
            set(buzzes_by_qnum)
            | {qnum for qnum, _, _ in bonus_groups}
            | {
                pq.question_number
                for pq in game.round.packet.packet_questions
                if pq.question_number is not None
            }
        )

        for qnum in qnums:
            for buzz in sorted(
                buzzes_by_qnum.get(qnum, []),
                key=lambda b: ((b.buzz_position or 0), b.id),
            ):
                team_db_id = buzz.player.team_id
                opponent_db_id = other_team_id(scores, team_db_id)
                tossup = buzz.tossup
                token_position = max(
                    0,
                    (buzz.buzz_position or 0)
                    - acf_sanitization.get_buzz_offset(tossup.question_text or ""),
                )
                buzz_rows.append(
                    with_years(
                        {
                            "buzz_id": f"{game_id_for(game)}::buzz-{buzz.id}",
                            "tossup_id": tossup_id_by_db_id.get(buzz.tossup_id, ""),
                            "game_id": game_id_for(game),
                            "player_id": player_id_by_db_id[buzz.player_id],
                            "team_id": team_ids_by_db_id[team_db_id],
                            "token_position": token_position,
                            "num_tokens_total": len(
                                acf_sanitization.sanitize_question(
                                    tossup.question_text or ""
                                ).split()
                            ),
                            "correct": buzz.value > 0,
                            "team_score": scores[team_db_id],
                            "opponent_score": scores[opponent_db_id],
                            "question_number_in_game": qnum,
                            "value": buzz.value,
                        },
                        [year],
                    )
                )
                scores[team_db_id] += buzz.value or 0

            for (group_qnum, team_db_id, bonus_id), parts in sorted(bonus_groups.items()):
                if group_qnum != qnum:
                    continue
                opponent_db_id = other_team_id(scores, team_db_id)
                bonus_part_chunks = chunk_bonus_parts(parts)
                for chunk_index, chunk in enumerate(bonus_part_chunks, start=1):
                    sorted_parts = sorted(
                        chunk, key=lambda p: (p.bonus_part.part_number or 0, p.id)
                    )
                    if len(sorted_parts) != 3:
                        warnings.append(
                            {
                                "type": "non_three_part_bonus_response",
                                "game_id": game_id_for(game),
                                "bonus_id": bonus_id,
                                "team_id": team_ids_by_db_id[team_db_id],
                                "part_count": len(sorted_parts),
                            }
                        )
                    earning_buzz = correct_buzz_by_qnum_team.get((qnum, team_db_id))
                    if earning_buzz is None:
                        warnings.append(
                            {
                                "type": "missing_earning_player",
                                "game_id": game_id_for(game),
                                "bonus_id": bonus_id,
                                "team_id": team_ids_by_db_id[team_db_id],
                                "question_number_in_game": qnum,
                            }
                        )
                    response_id = (
                        f"{game_id_for(game)}::{bonus_id}::{team_ids_by_db_id[team_db_id]}"
                    )
                    if len(bonus_part_chunks) > 1:
                        response_id = f"{response_id}::attempt-{chunk_index}"
                    bonus_response_rows.append(
                        with_years(
                            {
                                "response_id": response_id,
                                "bonus_id": bonus_id,
                                "game_id": game_id_for(game),
                                "team_id": team_ids_by_db_id[team_db_id],
                                "earning_player_id": player_id_by_db_id[
                                    earning_buzz.player_id
                                ]
                                if earning_buzz
                                else "",
                                "parts_correct": [
                                    part.value > 0 for part in sorted_parts
                                ],
                                "total_points": sum(
                                    part.value or 0 for part in sorted_parts
                                ),
                                "team_score": scores[team_db_id],
                                "opponent_score": scores[opponent_db_id],
                                "question_number_in_game": qnum,
                            },
                            [year],
                        )
                    )
                    scores[team_db_id] += sum(part.value or 0 for part in sorted_parts)

        game_rows.append(
            with_years(
                {
                    "game_id": game_id_for(game),
                    "tournament_id": tournament.slug
                    or slugify(tournament.name, f"tournament-{tournament.id}"),
                    "round": f"round{game.round.number}",
                    "team_ids": [
                        team_ids_by_db_id[game.team_one_id],
                        team_ids_by_db_id[game.team_two_id],
                    ],
                    "final_scores": [
                        scores[game.team_one_id],
                        scores[game.team_two_id],
                    ],
                    "tossups_read": game.tossups_read or 0,
                },
                [year],
            )
        )

    if strict and warnings:
        raise RuntimeError(f"Validation warnings encountered: {warnings[:10]}")

    return (
        sorted(game_rows, key=lambda r: r["game_id"]),
        sorted(buzz_rows, key=lambda r: r["buzz_id"]),
        sorted(bonus_response_rows, key=lambda r: r["response_id"]),
        warnings,
    )


def chunk_bonus_parts(parts: list[Any]) -> list[list[Any]]:
    """Split duplicate bonus-part direct groups into normal three-part responses."""
    ordered = sorted(parts, key=lambda p: p.id)
    chunks = [ordered[i : i + 3] for i in range(0, len(ordered), 3)]
    return chunks or []


def other_team_id(scores: dict[int, int], team_id: int) -> int:
    opponents = [candidate for candidate in scores if candidate != team_id]
    if not opponents:
        raise ValueError(f"Could not find opponent for team_id={team_id}")
    return opponents[0]


def build_tables(
    db_path: str,
    identity_overrides_path: str = DEFAULT_IDENTITY_OVERRIDES,
    strict: bool = False,
) -> tuple[dict[str, list[dict]], list[dict]]:
    require_project_modules()
    db_for_session, cleanup_db = prepare_orm_compatible_db(db_path)
    session = models.create_session(db_for_session)
    try:
        overrides = load_identity_overrides(identity_overrides_path)
        occurrences = load_player_occurrences(session)
        player_id_by_db_id, player_rows, identity_warnings = build_player_identity_maps(
            occurrences, overrides
        )
        tournament_rows = create_tournaments(session)
        team_rows = create_teams(session, player_id_by_db_id)
        tossup_rows, tossup_id_by_db_id, tossup_qnum_by_db_id = create_tossups(session)
        (
            bonus_rows,
            bonus_id_by_part_db_id,
            _bonus_id_by_bonus_db_id,
            bonus_qnum_by_bonus_db_id,
        ) = create_bonuses(session)
        game_rows, buzz_rows, bonus_response_rows, timeline_warnings = build_game_outputs(
            session,
            player_id_by_db_id,
            tossup_id_by_db_id,
            tossup_qnum_by_db_id,
            bonus_id_by_part_db_id,
            bonus_qnum_by_bonus_db_id,
            strict=strict,
        )
    finally:
        session.close()
        cleanup_db()

    warnings = identity_warnings + timeline_warnings
    if strict and warnings:
        raise RuntimeError(f"Validation warnings encountered: {warnings[:10]}")

    return (
        {
            "tournaments": tournament_rows,
            "players": player_rows,
            "teams": team_rows,
            "games": game_rows,
            "tossups": tossup_rows,
            "bonuses": bonus_rows,
            "tossup_buzzes": buzz_rows,
            "bonus_responses": bonus_response_rows,
        },
        warnings,
    )


def prepare_orm_compatible_db(db_path: str) -> tuple[str, Any]:
    """Return a DB path that has optional ORM columns, without mutating the source DB."""
    if sqlite_has_column(db_path, "question", "question_set_edition_id"):
        return db_path, lambda: None

    temp_file = tempfile.NamedTemporaryFile(prefix="acf-hf-irt-", suffix=".db", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    shutil.copy2(db_path, temp_path)
    with sqlite3.connect(temp_path) as conn:
        conn.execute("ALTER TABLE question ADD COLUMN question_set_edition_id INTEGER")
        conn.execute(
            """
            UPDATE question
            SET question_set_edition_id = (
                SELECT packet.question_set_edition_id
                FROM packet_question
                JOIN packet ON packet.id = packet_question.packet_id
                WHERE packet_question.question_id = question.id
                ORDER BY packet_question.id
                LIMIT 1
            )
            """
        )
        conn.commit()

    def cleanup() -> None:
        Path(temp_path).unlink(missing_ok=True)

    return temp_path, cleanup


def sqlite_has_column(db_path: str, table: str, column: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def write_outputs(
    tables: dict[str, list[dict]],
    warnings: list[dict],
    output_dir: str,
    repo_id: str | None = None,
    push: bool = False,
    warning_samples: int = 5,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    readme_path = output_path / "README.md"

    for table_name, rows in tables.items():
        dataset_dict = dataset_dict_from_rows(rows)
        table_path = output_path / table_name
        dataset_dict.save_to_disk(str(table_path))
        if push:
            if not repo_id:
                raise ValueError("--repo-id is required when --push is set")
            dataset_dict.push_to_hub(repo_id, config_name=table_name)

    (output_path / "identity_review_report.json").write_text(
        json.dumps(warnings, indent=2, sort_keys=True)
    )
    (output_path / "warning_summary.md").write_text(
        warning_summary_markdown(warnings, sample_limit=warning_samples)
    )
    readme_path.write_text(render_dataset_readme(tables, warnings))

    if push:
        upload_dataset_readme(repo_id, readme_path)


def render_dataset_readme(tables: dict[str, list[dict]], warnings: list[dict]) -> str:
    template = DATASET_README_TEMPLATE.read_text()
    return (
        template.replace("{{TABLE_COUNTS}}", table_counts_markdown(tables))
        .replace("{{WARNING_COUNTS}}", warning_counts_markdown(warnings))
    )


def table_counts_markdown(tables: dict[str, list[dict]]) -> str:
    lines = ["| Config | Rows |", "| --- | ---: |"]
    for table_name, rows in tables.items():
        lines.append(f"| `{table_name}` | {len(rows)} |")
    return "\n".join(lines)


def warning_counts_markdown(warnings: list[dict]) -> str:
    if not warnings:
        return "No identity or timeline warnings were generated."
    counts: dict[str, int] = defaultdict(int)
    for warning in warnings:
        counts[warning.get("type", "unknown")] += 1
    lines = ["| Warning type | Count |", "| --- | ---: |"]
    for warning_type, count in sorted(counts.items()):
        lines.append(f"| `{warning_type}` | {count} |")
    return "\n".join(lines)


def warning_summary_markdown(warnings: list[dict], sample_limit: int = 5) -> str:
    if not warnings:
        return "# Warning Summary\n\nNo identity or timeline warnings were generated.\n"

    grouped = group_warnings_by_type(warnings)
    lines = [
        "# Warning Summary",
        "",
        "This file samples the warning records from `identity_review_report.json`.",
        "The examples are meant for inspection and debugging; the JSON file is the full report.",
        "",
        warning_counts_markdown(warnings),
        "",
    ]
    for warning_type in sorted(grouped):
        examples = grouped[warning_type][:sample_limit]
        lines.extend(
            [
                f"## `{warning_type}`",
                "",
                warning_type_description(warning_type),
                "",
                f"Showing {len(examples)} of {len(grouped[warning_type])} warning(s).",
                "",
                "```json",
                json.dumps(examples, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def warning_summary_text(warnings: list[dict], sample_limit: int = 3) -> str:
    if not warnings:
        return "No identity or timeline warnings were generated."

    grouped = group_warnings_by_type(warnings)
    lines = ["Warning summary:"]
    for warning_type in sorted(grouped):
        examples = grouped[warning_type][:sample_limit]
        lines.append(f"- {warning_type}: {len(grouped[warning_type])}")
        for example in examples:
            lines.append("  " + json.dumps(example, sort_keys=True))
    return "\n".join(lines)


def group_warnings_by_type(warnings: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for warning in warnings:
        grouped[warning.get("type", "unknown")].append(warning)
    return grouped


def warning_type_description(warning_type: str) -> str:
    descriptions = {
        "ambiguous_suffixed_slug_group": (
            "One warning per base player slug group with numeric variants. These are "
            "not auto-merged because suffixes sometimes identify different people."
        ),
        "raw_slug_date_overlap": (
            "One warning per raw player slug whose occurrences overlap in date. The "
            "builder splits those occurrences unless a scoped override is provided."
        ),
        "raw_slug_name_conflict": (
            "One warning per raw player slug that appears with incompatible normalized "
            "names."
        ),
        "missing_earning_player": (
            "One warning per generated bonus response whose earning tossup player could "
            "not be inferred from a positive buzz by the same team on the same game and "
            "question number."
        ),
        "non_three_part_bonus_response": (
            "One warning per generated bonus response chunk that does not contain exactly "
            "three bonus part rows."
        ),
        "buzz_without_question_number": (
            "One warning per buzz whose tossup could not be mapped to a packet question "
            "number."
        ),
        "bonus_response_without_question_number": (
            "One warning per raw bonus part response whose bonus could not be mapped to a "
            "packet question number."
        ),
    }
    return descriptions.get(warning_type, "No description is registered for this warning type.")


def upload_dataset_readme(repo_id: str | None, readme_path: Path) -> None:
    if not repo_id:
        raise ValueError("--repo-id is required when --push is set")
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'huggingface_hub' package is required to upload README.md."
        ) from exc

    HfApi().upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a HuggingFace IRT dataset from an ACF SQLite database."
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--identity-overrides", default=DEFAULT_IDENTITY_OVERRIDES)
    parser.add_argument("--repo-id")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--warning-samples",
        type=int,
        default=3,
        help="Number of example warning records to print per warning type.",
    )
    args = parser.parse_args()

    tables, warnings = build_tables(
        args.db_path,
        identity_overrides_path=args.identity_overrides,
        strict=args.strict,
    )
    write_outputs(
        tables,
        warnings,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        push=args.push,
        warning_samples=args.warning_samples,
    )

    print("Dataset written to", args.output_dir)
    for table_name, rows in tables.items():
        print(f"  {table_name}: {len(rows)} rows")
    print(f"  identity/timeline warnings: {len(warnings)}")
    print()
    print(warning_summary_text(warnings, sample_limit=args.warning_samples))


if __name__ == "__main__":
    main()
