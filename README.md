# acf-data

## Project Overview
acf-data manages and analyzes Academic Competition Federation (ACF) tournament data. It stores and processes information on question sets, teams, players, and game statistics.

## Key Components
- [`docs/structure.md`](docs/structure.md) and [`models.py`](models.py): Define the data architecture
  - [`docs/structure.md`](docs/structure.md): Explains ACF tournament structure (tournaments, rounds, games, teams, questions)
  - [`models.py`](models.py): Implements database schema using SQLAlchemy ORM (QuestionSet, Tournament, Team, Player, Tossup, Buzz)

- [`recreate_db.py`](recreate_db.py): Cleans and prepares data
  - Copies selected tables from input database
  - Removes prefixes from table names
  - Creates new SQLite database with cleaned data

- [`merge_db.py`](merge_db.py): Consolidates data
  - Merges multiple tournament databases into one
  - Topological sorting of database tables for proper merge order
  - Uses algorithm respecting foreign key relationships
  - Handles conflicts and duplicates during merging

## Features
- Diff visualization for comparing records
- Data models for QuestionSet, Tournament, Team, Player, Tossup, and Buzz


## Installation
To install dependencies, run:
```bash
pip install -r requirements.txt
```

## Scripts

#### [`recreate_db.py`](recreate_db.py)
If you have tables of form `buzzpoints_{tablename}`, and you want to recreate the database with just the `{tablename}` tables, you can run:

```bash
python recreate_db.py data/sst-23-24.db data/sst-23-24-cleaned.db
```

#### [`add_missing_columns.py`](add_missing_columns.py)
In original database, `question` table is not directly linked to `question_set_edition` table, leading to some inconsistencies. This script inserts the new column to the input databases (if not present) and populates it with the correct values.
```bash
python add_missing_columns.py data/sst-23-24-cleaned.db data/nats24.db
```

#### [`merge_db.py`](merge_db.py)
To merge multiple databases with the same schema but potentially overlapping data:

1. Consolidates records from multiple input databases and creates a single unified database
2. Resolves duplicate entries across databases
3. Reassigns primary keys and updates foreign key references

Usage:
```bash
python merge_db.py -i data/sst-23-24-cleaned.db data/nats24.db -o data/acf-23-24.db
```

#### [`check_consistencies.py`](check_consistencies.py)
Check for various inconsistencies in a target database. E.g., it checks if:
- A buzz is associated with the same question set edition as the game and team's tournament
- A tossup is tournaments around the same date
- Tournament levels associated with a question set are the same

```bash
python check_consistencies.py data/sst-23-24-cleaned.db
```