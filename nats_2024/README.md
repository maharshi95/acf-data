# ACF Nationals 2024 Quiz Database

This project contains data analysis for the Academic Competition Federation (ACF) Nationals 2024 quiz tournament.

## Database Structure

The project uses an SQLite database (`nats24.db`) to store various tables related to the tournament. Here's a summary of the main tables and their contents:

1. **Teams**: Information about participating quiz teams
2. **Players**: Details of individual players
3. **Matches**: Data on individual quiz matches
4. **Questions**: Information about the questions used in the tournament
5. **Answers**: Recorded answers for each question in each match
6. **Tournaments**: Details about the overall tournament structure

## Scripts

- `cleanup_nats_24.py`: Python script for data cleaning and initial analysis
  - Connects to the SQLite database
  - Provides utility functions for querying tables
  - Generates table information including columns, types, and foreign key mappings

## Getting Started

1. Ensure you have the required Python libraries installed (pandas, sqlite3, tabulate)
2. Place the `nats24.db` file in the `./data/` directory
3. Run the `cleanup_nats_24.py` script to perform initial data exploration and cleaning

## Next Steps

- Develop specific analysis queries based on tournament statistics
- Create visualizations of team performances and individual player stats
- Generate reports on question difficulty, subject distribution, etc.

For more detailed information on the database structure and available data, refer to the output of the `cleanup_nats_24.py` script.

## Schema
```mermaid
erDiagram
    question_set ||--o{ question_set_edition : has
    question_set_edition ||--o{ packet : contains
    question_set_edition ||--o{ tournament : uses
    packet ||--o{ packet_question : has
    packet ||--o{ round : used_in
    question ||--o{ packet_question : in
    question ||--o{ tossup : has
    question ||--o{ bonus : has
    tossup ||--o{ tossup_hash : has
    bonus ||--o{ bonus_hash : has
    bonus ||--o{ bonus_part : contains
    tournament ||--o{ round : has
    tournament ||--o{ team : participates
    team ||--o{ player : has
    round ||--o{ game : includes
    game }o--|| team : team_one
    game }o--|| team : team_two
    game ||--o{ buzz : records
    game ||--o{ bonus_part_direct : records
    player ||--o{ buzz : makes
    tossup ||--o{ buzz : on
    team ||--o{ bonus_part_direct : answers
    bonus_part ||--o{ bonus_part_direct : of

    question_set {
        int id PK
        string name
        string slug
        string difficulty
    }
    question_set_edition {
        int id PK
        int question_set_id FK
        string name
        string slug
        date date
    }
    packet {
        int id PK
        int question_set_edition_id FK
        string name
    }
    packet_question {
        int id PK
        int packet_id FK
        int question_number
        int question_id FK
    }
    question {
        int id PK
        string slug
        string metadata
        string author
        string editor
        string category
        string category_slug
        string subcategory
        string subcategory_slug
        string subsubcategory
    }
    tossup {
        int id PK
        int question_id FK
        string question
        string answer
        string answer_sanitized
        string answer_primary
    }
    bonus {
        int id PK
        int question_id FK
        string leadin
        string leadin_sanitized
    }
    bonus_part {
        int id PK
        int part_number
        int bonus_id FK
        string part
        string part_sanitized
        string answer
        string answer_sanitized
        string answer_primary
        int value
        string difficulty_modifier
    }
    tossup_hash {
        string hash PK
        int question_id FK
        int tossup_id FK
    }
    bonus_hash {
        string hash PK
        int question_id FK
        int bonus_id FK
    }
    tournament {
        int id PK
        string name
        string slug
        int question_set_edition_id FK
        string location
        string level
        date start_date
        date end_date
    }
    round {
        int id PK
        int tournament_id FK
        int number
        int packet_id FK
        bit exclude_from_individual
    }
    team {
        int id PK
        int tournament_id FK
        string name
        string slug
    }
    player {
        int id PK
        int team_id FK
        string name
        string slug
    }
    game {
        int id PK
        int round_id FK
        int tossups_read
        int team_one_id FK
        int team_two_id FK
    }
    buzz {
        int id PK
        int player_id FK
        int game_id FK
        int tossup_id FK
        int buzz_position
        int value
    }
    bonus_part_direct {
        int id PK
        int team_id FK
        int game_id FK
        int bonus_part_id FK
        int value
    }
```