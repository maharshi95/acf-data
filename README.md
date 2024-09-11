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
