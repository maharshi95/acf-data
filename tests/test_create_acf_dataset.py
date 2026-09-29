from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "create_acf_dataset.py"
SPEC = importlib.util.spec_from_file_location("create_acf_dataset", SCRIPT_PATH)
acf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = acf
SPEC.loader.exec_module(acf)


def lookup(
    row_id: int,
    team_id: int,
    name: str,
    slug: str,
    person_id: str | None,
):
    return acf.LookupRow(row_id, team_id, name, slug, person_id)


class IdentityResolutionTests(unittest.TestCase):
    def test_same_person_slugs_merge_to_deterministic_canonical_slug(self):
        resolution = acf.resolve_identities(
            [
                lookup(1, 10, "Ada Lovelace", "ada", "42"),
                lookup(2, 20, "Ada Lovelace", "ada-lovelace", "42"),
                lookup(3, 30, "Ada Lovelace", "ada-lovelace", "42"),
            ]
        )

        self.assertEqual(len(resolution.players), 1)
        self.assertEqual(resolution.players[0]["slug"], "ada-lovelace")
        self.assertEqual(set(resolution.player_id_by_db_id.values()), {"p-ada-lovelace"})
        self.assertEqual(len(resolution.inconsistencies["same_person_multiple_slugs"]), 1)

    def test_same_slug_different_people_are_all_disambiguated(self):
        resolution = acf.resolve_identities(
            [
                lookup(1, 10, "Alex Kim", "alex-kim", "100"),
                lookup(2, 20, "Alex Kim", "alex-kim", "200"),
            ]
        )

        self.assertEqual(
            {player["slug"] for player in resolution.players},
            {"alex-kim", "alex-kim-2"},
        )
        self.assertTrue(all(player["orig-acf-slug"] == "alex-kim" for player in resolution.players))
        self.assertEqual(len(resolution.inconsistencies["same_slug_multiple_people"]), 1)

    def test_missing_person_ids_are_assigned_alphabetically_from_50000(self):
        resolution = acf.resolve_identities(
            [
                lookup(7, 10, "Zoe Guest", "guest", None),
                lookup(8, 20, "Alice Guest", "guest", None),
            ]
        )

        players_by_name = {player["name"]: player for player in resolution.players}
        self.assertEqual(players_by_name["Alice Guest"]["person_id"], "50000")
        self.assertEqual(players_by_name["Alice Guest"]["slug"], "guest")
        self.assertEqual(players_by_name["Zoe Guest"]["person_id"], "50001")
        self.assertEqual(players_by_name["Zoe Guest"]["slug"], "guest-2")
        self.assertEqual(len(resolution.inconsistencies["missing_person_id"]), 2)

    def test_unsuffixed_alias_wins_over_numeric_variant(self):
        resolution = acf.resolve_identities(
            [
                lookup(1, 10, "Morgan Lee", "morgan-lee-3", "42"),
                lookup(2, 20, "Morgan Lee", "morgan-lee", "42"),
            ]
        )

        self.assertEqual(resolution.players[0]["slug"], "morgan-lee")
        self.assertEqual(resolution.players[0]["orig-acf-slug"], "morgan-lee")

    def test_single_alias_owner_reserves_base_from_multi_alias_person(self):
        resolution = acf.resolve_identities(
            [
                lookup(1, 10, "Morgan Lee", "morgan-lee", "42"),
                lookup(2, 20, "Morgan Lee", "morgan-lee-3", "42"),
                lookup(3, 30, "Another Morgan Lee", "morgan-lee", "99"),
            ]
        )

        players_by_person = {player["person_id"]: player for player in resolution.players}
        self.assertEqual(players_by_person["99"]["slug"], "morgan-lee")
        self.assertEqual(players_by_person["42"]["slug"], "morgan-lee-3")

    def test_collision_suffix_skips_an_existing_canonical_slug(self):
        resolution = acf.resolve_identities(
            [
                lookup(1, 10, "Alex Kim A", "alex-kim", "10"),
                lookup(2, 20, "Alex Kim B", "alex-kim", "20"),
                lookup(3, 30, "Alex Kim C", "alex-kim-2", "30"),
            ]
        )

        self.assertEqual(
            {player["slug"] for player in resolution.players},
            {"alex-kim", "alex-kim-2", "alex-kim-3"},
        )


class LookupValidationTests(unittest.TestCase):
    def test_lookup_mismatch_is_reported(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute("CREATE TABLE player (id INTEGER, team_id INTEGER, name TEXT, slug TEXT)")
        connection.execute("INSERT INTO player VALUES (1, 10, 'Ada Lovelace', 'ada-lovelace')")

        violations = acf.validate_lookup(
            connection, [lookup(1, 11, "Ada Lovelace", "ada-lovelace", "42")]
        )

        self.assertEqual(violations[0]["code"], "lookup_row_mismatch")
        self.assertIn("team_id", violations[0]["fields"])


class SplitTests(unittest.TestCase):
    def test_configs_have_year_and_full_splits_with_internal_fields_removed(self):
        configs = acf.partition_records(
            {
                "players": [
                    {"_years": ["2023", "2025"], "player_id": "p-ada"},
                    {"_years": ["2024"], "player_id": "p-grace"},
                ],
                "teams": [
                    {"_year": "2023", "team_id": "tm-1"},
                    {"_year": "2024", "team_id": "tm-2"},
                ],
            }
        )

        self.assertEqual(list(configs["players"]), ["2023", "2024", "2025", "full"])
        self.assertEqual(len(configs["players"]["full"]), 2)
        self.assertEqual(configs["players"]["2025"], [{"player_id": "p-ada"}])
        self.assertEqual(configs["teams"]["2025"], [])

    def test_unsupported_year_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported or missing year"):
            acf.partition_records({"teams": [{"_year": "2022", "team_id": "tm-1"}]})


class TournamentAndGameMetadataTests(unittest.TestCase):
    def test_question_set_id_prefixes(self):
        expected = {
            "2023-acf-regionals": "acf-regs-23",
            "2025-acf-nationals": "acf-nats-25",
            "2024-acf-winter": "acf-wint-24",
            "2025-acf-fall": "acf-fall-25",
            "2023-chicago-open": "co-23",
        }
        for slug, prefix in expected.items():
            with self.subTest(slug=slug):
                self.assertEqual(acf.question_set_id_prefix(slug), prefix)

    def test_unknown_question_set_slug_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported question set slug"):
            acf.question_set_id_prefix("2025-other-tournament")

    def test_games_use_question_set_ids_and_expose_source_slug(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            CREATE TABLE question_set (id INTEGER, slug TEXT);
            CREATE TABLE question_set_edition (id INTEGER, question_set_id INTEGER);
            CREATE TABLE tournament (id INTEGER, name TEXT, slug TEXT, level TEXT,
                                     start_date TEXT, question_set_edition_id INTEGER);
            CREATE TABLE packet (id INTEGER, name TEXT, descriptor TEXT);
            CREATE TABLE round (id INTEGER, number INTEGER, exclude_from_individual INTEGER,
                                packet_id INTEGER, tournament_id INTEGER);
            CREATE TABLE game (id INTEGER, tossups_read INTEGER, team_one_id INTEGER,
                               team_two_id INTEGER, round_id INTEGER);
            INSERT INTO question_set VALUES (1, '2024-acf-regionals'),
                                            (2, '2025-chicago-open');
            INSERT INTO question_set_edition VALUES (1, 1), (2, 2);
            INSERT INTO tournament VALUES
                (1, 'Berkeley', '2024-acf-regionals-berkeley', 'open', '2024-01-27', 1),
                (2, 'Chicago Open', '2025-chicago-open', 'open', '2025-06-01', 2);
            INSERT INTO packet VALUES (1, 'Packet 1', 'A'), (2, 'Packet 2', 'B');
            INSERT INTO round VALUES (1, 1, 0, 1, 1), (2, 1, 0, 2, 2);
            INSERT INTO game VALUES (10, 20, 1, 2, 1), (11, 20, 3, 4, 2);
        """)
        games, game_id_map = acf.build_games(
            connection,
            {1: "team-1", 2: "team-2", 3: "team-3", 4: "team-4"},
            {1: "acf-regs-24-tour-1", 2: "co-25-tour-2"},
        )

        self.assertEqual(game_id_map, {10: "acf-regs-24-g-10", 11: "co-25-g-11"})
        self.assertEqual([game["question_set"] for game in games],
                         ["2024-acf-regionals", "2025-chicago-open"])
        self.assertEqual(games[0]["round_id"], "acf-regs-24-r-1")
        self.assertEqual(games[1]["packet_id"], "co-25-p-2")

    def test_event_level_comes_from_question_set_slug(self):
        self.assertEqual(acf.infer_event_level("2025-acf-regionals"), "regionals")
        self.assertEqual(acf.infer_event_level("2024-acf-nationals"), "nationals")
        self.assertEqual(acf.infer_event_level("2024-chicago-open"), "open")
        self.assertEqual(acf.normalized_packet_set("2025-acf-regionals"), "2025-regionals")

    def test_dot_difficulty_uses_acf_packet_set_tier(self):
        self.assertEqual(acf.infer_dots("2025-acf-fall", "Easy (●)"), 1)
        self.assertEqual(acf.infer_dots("2025-acf-winter", "Medium"), 2)
        self.assertEqual(acf.infer_dots("2025-acf-regionals", "Open"), 3)
        self.assertEqual(acf.infer_dots("2024-chicago-open", "Open"), 4)

    def test_game_type_requires_an_explicit_packet_label(self):
        self.assertEqual(acf.infer_game_type("Prelims 3. Florida A"), "prelim")
        self.assertEqual(acf.infer_game_type("Playoffs 2. Editors"), "playoff")
        self.assertEqual(acf.infer_game_type("Finals 1. Editors"), "final")
        self.assertEqual(acf.infer_game_type("Play-In. Team A, Team B"), "play-in")
        self.assertIsNone(acf.infer_game_type("Packet A (Team A, Team B)"))


class BonusResponseTests(unittest.TestCase):
    def test_numeric_bonus_values_are_scored(self):
        self.assertEqual(acf.normalize_bonus_value(10), (10, True, True))
        self.assertEqual(acf.normalize_bonus_value(0), (0, False, True))

    def test_na_bonus_value_is_unscored_not_incorrect(self):
        self.assertEqual(acf.normalize_bonus_value("NA"), (None, None, False))
        self.assertEqual(acf.normalize_bonus_value(None), (None, None, False))


class DatasetCardTests(unittest.TestCase):
    def test_readme_upload_replaces_legacy_config_and_preserves_other_metadata(self):
        from huggingface_hub import DatasetCard

        card = DatasetCard("""---
configs:
- config_name: tournament
  data_files: tournament/*
- config_name: tournaments
  data_files: tournaments/*
- config_name: games
  data_files: games/*
dataset_info:
- config_name: tournament
  features: []
- config_name: tournaments
  features: []
- config_name: games
  features: []
---
# Generated card
""")
        with tempfile.TemporaryDirectory() as directory:
            readme_path = Path(directory) / "README.md"
            readme_path.write_text("# Detailed dataset card\n", encoding="utf-8")
            with patch("huggingface_hub.DatasetCard.load", return_value=card), patch(
                "huggingface_hub.HfApi"
            ) as api_class:
                api = api_class.return_value
                api.list_repo_files.return_value = [
                    "tournament/2024.parquet", "tournaments/2024.parquet", "games/2024.parquet"
                ]
                acf.upload_dataset_readme("org/repo", readme_path)

        api.delete_folder.assert_called_once_with(
            "tournament", "org/repo", repo_type="dataset"
        )
        payload = api.upload_file.call_args.kwargs["path_or_fileobj"].decode("utf-8")
        self.assertIn("config_name: tournaments", payload)
        self.assertIn("config_name: games", payload)
        self.assertNotIn("config_name: tournament\n", payload)
        self.assertIn("# Detailed dataset card", payload)
        self.assertNotIn("# Generated card", payload)
        self.assertEqual(api.upload_file.call_args.kwargs["repo_type"], "dataset")


if __name__ == "__main__":
    unittest.main()
