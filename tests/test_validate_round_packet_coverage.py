import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_round_packet_coverage.py"
SPEC = importlib.util.spec_from_file_location("validate_round_packet_coverage", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class RoundPacketCoverageTests(unittest.TestCase):
    def test_reports_round_shortfall_and_unassigned_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "coverage.db"
            with sqlite3.connect(db_path) as connection:
                connection.executescript("""
                    CREATE TABLE question_set (id INTEGER, slug TEXT);
                    CREATE TABLE question_set_edition
                        (id INTEGER, question_set_id INTEGER, slug TEXT);
                    CREATE TABLE tournament
                        (id INTEGER, name TEXT, start_date TEXT,
                         question_set_edition_id INTEGER);
                    CREATE TABLE team
                        (id INTEGER, tournament_id INTEGER, name TEXT);
                    CREATE TABLE packet
                        (id INTEGER, question_set_edition_id INTEGER,
                         name TEXT, number INTEGER);
                    CREATE TABLE packet_question
                        (id INTEGER, packet_id INTEGER, question_id INTEGER);
                    CREATE TABLE round
                        (id INTEGER, tournament_id INTEGER, number INTEGER,
                         packet_id INTEGER);
                    CREATE TABLE game
                        (id INTEGER, round_id INTEGER, team_one_id INTEGER,
                         team_two_id INTEGER);
                    INSERT INTO question_set VALUES (1, 'example-set');
                    INSERT INTO question_set_edition VALUES (1, 1, 'main');
                    INSERT INTO tournament VALUES (1, 'Example', '2025-01-01', 1);
                    INSERT INTO team VALUES
                        (1, 1, 'A'), (2, 1, 'B'), (3, 1, 'C'), (4, 1, 'D');
                    INSERT INTO packet VALUES
                        (1, 1, 'Packet 1', 1),
                        (2, 1, 'Packet 2', 2),
                        (3, 1, 'Unused packet', 3);
                    INSERT INTO packet_question VALUES
                        (1, 1, 10), (2, 1, 11), (3, 3, 10), (4, 3, 11);
                    INSERT INTO round VALUES (1, 1, 1, 1), (2, 1, 2, 2);
                    INSERT INTO game VALUES
                        (1, 1, 1, 2), (2, 1, 3, 4), (3, 2, 1, 2);
                """)
            rounds, packets, totals = validator.audit(db_path)
            self.assertEqual(totals["rounds"], 2)
            self.assertEqual(totals["games"], 3)
            self.assertEqual(totals["packets_without_rounds"], 1)
            self.assertEqual(totals["unassigned_exact_shadows"], 1)
            self.assertEqual(rounds[1]["shortfall_if_full_field"], 1)
            self.assertEqual(rounds[1]["shortfall_vs_tournament_peak"], 1)
            self.assertIn("3:C", rounds[1]["absent_registered_teams"])
            self.assertIn("4:D", rounds[1]["absent_registered_teams"])
            self.assertEqual(rounds[1]["possible_pair_if_full_field"], "3:C vs 4:D")
            self.assertEqual(packets[2]["round_count"], 0)
            self.assertEqual(packets[2]["best_used_packet_id"], 1)
            self.assertTrue(packets[2]["exact_shadow_of_used_packet"])


if __name__ == "__main__":
    unittest.main()
