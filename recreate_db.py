# %%
"""
Prunes and dumps a SQLite database by copying selected tables.

This module contains a function `prune_and_dump_db` that:
1. Copies tables prefixed with "buzzpoints_" from an input database.
2. Removes the "buzzpoints_" prefix from table names in the output.
3. Creates a new SQLite database with the pruned and renamed tables.

Args:
    input_db_path (str): Path to the input SQLite database.
    output_db_path (str): Path for the new, pruned SQLite database.
"""

import os
import sqlite3
from typing import Iterable

import pandas as pd


def prune_and_dump_db(input_db_path, output_db_path):
    # Connect to the input database
    input_conn = sqlite3.connect(input_db_path)
    input_cursor = input_conn.cursor()

    # Remove the output database if it exists
    if os.path.exists(output_db_path):
        os.remove(output_db_path)

    # Create a new output database
    output_conn = sqlite3.connect(output_db_path)
    output_cursor = output_conn.cursor()

    # Get all table names from the input database
    input_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables: Iterable[str] = input_cursor.fetchall()

    # Filter and copy tables that start with "buzzpoints_"
    for table in tables:
        table_name = table[0]
        if table_name.startswith("buzzpoints_"):
            # Get the new table name by removing "buzzpoints_" prefix
            new_table_name = table_name.removeprefix("buzzpoints_")

            # Copy table structure with new name
            input_cursor.execute(
                f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';"
            )
            create_table_sql = input_cursor.fetchone()[0]
            create_table_sql = create_table_sql.replace(table_name, new_table_name)
            output_cursor.execute(create_table_sql)

            # Copy table data
            input_cursor.execute(f"SELECT * FROM {table_name}")
            rows = input_cursor.fetchall()
            output_cursor.executemany(
                f"INSERT INTO {new_table_name} VALUES ({','.join(['?' for _ in range(len(rows[0]))])})",
                rows,
            )

    # Commit changes and close connections
    output_conn.commit()
    input_conn.close()
    output_conn.close()

    print(f"Pruned database saved to {output_db_path}")


def test_db_equality(input_db_path, output_db_path):
    input_conn = sqlite3.connect(input_db_path)
    output_conn = sqlite3.connect(output_db_path)

    # Get all table names from the input database
    input_cursor = input_conn.cursor()
    input_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = input_cursor.fetchall()

    for table in tables:
        table_name = table[0]
        if table_name.startswith("buzzpoints_"):
            new_table_name = table_name.removeprefix("buzzpoints_")

            # Load DataFrames
            input_df = pd.read_sql_query(f"SELECT * FROM {table_name}", input_conn)
            output_df = pd.read_sql_query(
                f"SELECT * FROM {new_table_name}", output_conn
            )

            # Check if DataFrames are equal
            if not input_df.equals(output_df):
                print(f"Mismatch found in table: {table_name}")
                return False

    print("All tables are equal between input and output databases.")
    return True


# Usage
input_db_path = "./data/sst-23-24.db"  # Replace with your input database path
output_db_path = "./data/sst-23-24-cleaned.db"  # Replace with your desired output path

prune_and_dump_db(input_db_path, output_db_path)

# Test the equality of the databases
test_result = test_db_equality(input_db_path, output_db_path)
print(f"Test result: {'Passed' if test_result else 'Failed'}")
