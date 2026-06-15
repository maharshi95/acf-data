from __future__ import annotations

import datetime as dt
import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "create_hf_irt_dataset.py"
SPEC = importlib.util.spec_from_file_location("create_hf_irt_dataset", SCRIPT_PATH)
hf_irt = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = hf_irt
SPEC.loader.exec_module(hf_irt)


def occurrence(
    player_db_id: int,
    slug: str,
    name: str,
    team_slug: str,
    tournament_slug: str,
    date: dt.date,
):
    return hf_irt.PlayerOccurrence(
        player_db_id=player_db_id,
        player_slug=slug,
        player_name=name,
        team_slug=team_slug,
        team_name=team_slug,
        tournament_slug=tournament_slug,
        tournament_name=tournament_slug,
        start_date=date,
        end_date=date,
    )


class PlayerIdentityTests(unittest.TestCase):
    def test_repeated_raw_slug_merges_when_dates_do_not_overlap(self):
        player_id_by_db_id, rows, warnings = hf_irt.build_player_identity_maps(
            [
                occurrence(1, "andy-yu", "Andy Yu", "chicago-a", "2024-acf-nationals", dt.date(2024, 4, 21)),
                occurrence(2, "andy-yu", "Andy Yu", "chicago-a", "2025-acf-nationals", dt.date(2025, 4, 19)),
            ]
        )

        self.assertEqual(player_id_by_db_id[1], "p-andy-yu")
        self.assertEqual(player_id_by_db_id[2], "p-andy-yu")
        self.assertEqual(len(rows), 1)
        self.assertEqual(warnings, [])

    def test_repeated_raw_slug_is_split_when_dates_overlap(self):
        player_id_by_db_id, rows, warnings = hf_irt.build_player_identity_maps(
            [
                occurrence(1, "kevin", "Kevin", "amherst-a", "2024-acf-winter-at-brandeis", dt.date(2024, 11, 16)),
                occurrence(2, "kevin", "Kevin", "michigan-b", "2024-acf-winter-at-ohio-state", dt.date(2024, 11, 16)),
            ]
        )

        self.assertNotEqual(player_id_by_db_id[1], player_id_by_db_id[2])
        self.assertEqual(len(rows), 2)
        self.assertTrue(any(w["type"] == "raw_slug_date_overlap" for w in warnings))

    def test_suffixed_variants_are_flagged_not_auto_merged(self):
        player_id_by_db_id, rows, warnings = hf_irt.build_player_identity_maps(
            [
                occurrence(1, "ethan", "Ethan", "williams-a", "2024-acf-winter-at-brandeis", dt.date(2024, 11, 16)),
                occurrence(2, "ethan-2", "Ethan", "rutgers-c", "2024-acf-winter-at-lehigh", dt.date(2024, 11, 16)),
                occurrence(3, "ethan-3", "Ethan", "cwru-d", "2024-acf-winter-at-ohio-state", dt.date(2024, 11, 16)),
            ]
        )

        self.assertEqual(len(set(player_id_by_db_id.values())), 3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(
            any(
                w["type"] == "ambiguous_suffixed_slug_group"
                and w["base_slug"] == "ethan"
                for w in warnings
            )
        )

    def test_accent_normalized_names_are_compatible(self):
        player_id_by_db_id, rows, warnings = hf_irt.build_player_identity_maps(
            [
                occurrence(1, "michal-gerasimiuk", "Michał Gerasimiuk", "stanford-a", "2024-acf-regionals-berkeley", dt.date(2024, 1, 27)),
                occurrence(2, "michal-gerasimiuk", "Michal Gerasimiuk", "stanford-a", "2025-acf-regionals-uk", dt.date(2025, 2, 1)),
            ]
        )

        self.assertEqual(player_id_by_db_id[1], "p-michal-gerasimiuk")
        self.assertEqual(player_id_by_db_id[2], "p-michal-gerasimiuk")
        self.assertEqual(len(rows), 1)
        self.assertFalse(any(w["type"] == "raw_slug_name_conflict" for w in warnings))

    def test_scoped_override_can_merge_ambiguous_variant(self):
        occurrences = [
            occurrence(1, "patrick-torre", "Patrick Torre", "maryland-c", "2024-acf-regionals-jmu", dt.date(2024, 1, 27)),
            occurrence(2, "patrick-torre-2", "Patrick Torre", "maryland-b", "2024-acf-regionals-jmu", dt.date(2024, 1, 27)),
        ]
        overrides = {
            (
                "2024-acf-regionals-jmu",
                "maryland-b",
                "patrick-torre-2",
            ): "p-patrick-torre",
        }
        player_id_by_db_id, rows, _warnings = hf_irt.build_player_identity_maps(
            occurrences, overrides
        )

        self.assertEqual(player_id_by_db_id[1], "p-patrick-torre")
        self.assertEqual(player_id_by_db_id[2], "p-patrick-torre")
        self.assertEqual(len(rows), 1)


class DatasetReadmeTests(unittest.TestCase):
    def test_readme_renders_counts_and_warnings(self):
        text = hf_irt.render_dataset_readme(
            {"players": [{"player_id": "p-a"}], "games": [{}, {}]},
            [
                {"type": "ambiguous_suffixed_slug_group"},
                {"type": "ambiguous_suffixed_slug_group"},
                {"type": "missing_earning_player"},
            ],
        )

        self.assertIn("| `players` | 1 |", text)
        self.assertIn("| `games` | 2 |", text)
        self.assertIn("| `ambiguous_suffixed_slug_group` | 2 |", text)
        self.assertNotIn("{{TABLE_COUNTS}}", text)
        self.assertNotIn("{{WARNING_COUNTS}}", text)

    def test_warning_summary_renders_examples(self):
        warnings = [
            {"type": "missing_earning_player", "game_id": "g1", "bonus_id": "b1"},
            {"type": "missing_earning_player", "game_id": "g2", "bonus_id": "b2"},
            {"type": "raw_slug_date_overlap", "slug": "kevin"},
        ]

        text = hf_irt.warning_summary_text(warnings, sample_limit=1)
        markdown = hf_irt.warning_summary_markdown(warnings, sample_limit=1)

        self.assertIn("missing_earning_player: 2", text)
        self.assertIn("raw_slug_date_overlap: 1", text)
        self.assertIn("Showing 1 of 2 warning(s).", markdown)
        self.assertIn('"game_id": "g1"', markdown)


if __name__ == "__main__":
    unittest.main()
