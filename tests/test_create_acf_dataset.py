from __future__ import annotations

import importlib.util
import sqlite3
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
