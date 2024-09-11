"""
Adds a `question_set_edition_id` column to the `question` table.

This column is necessary for ACF data consistency.

Example usage:

```bash
python add_missing_columns.py data/sst-23-24-cleaned.db data/nats24.db
```
"""

import sys

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from models import PacketQuestion, Question


def column_exists(engine, table_name, column_name):
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def add_column(engine, table_name, column_name, column_type):
    if not column_exists(engine, table_name, column_name):
        with engine.connect() as conn:
            conn.execute(
                text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            )
            conn.commit()
    else:
        print(f"Column {column_name} already exists in table {table_name}")


def determine_question_set_edition_id(question: Question) -> int:
    qset_edition_ids = set()
    for pq in question.packet_questions:
        qset_edition_ids.add(pq.packet.question_set_edition_id)
    assert len(qset_edition_ids) == 1
    return qset_edition_ids.pop()


def populate_question_set_edition_id(session):
    questions = session.query(Question).all()
    for question in questions:
        question_set_edition_id = determine_question_set_edition_id(question)
        if question.question_set_edition_id is None:
            question.question_set_edition_id = question_set_edition_id
        elif question.question_set_edition_id != question_set_edition_id:
            print(
                f"Question {question.id} has question_set_edition_id {question.question_set_edition_id} but should be {question_set_edition_id}"
            )
            assert False
    session.commit()


def validate_question_set_edition_constraint(session):
    packet_questions = session.query(PacketQuestion).all()
    for pq in packet_questions:
        assert pq.question.question_set_edition_id == pq.packet.question_set_edition_id


def add_question_set_edition_id_to_question(db_path: str):
    engine = create_engine(db_path)
    Session = sessionmaker(bind=engine)
    session = Session()

    print("Adding question_set_edition_id column...")
    add_column(engine, "question", "question_set_edition_id", "INTEGER")

    print("Populating question_set_edition_id...")
    populate_question_set_edition_id(session)

    print("Validating question_set_edition_id constraint...")
    validate_question_set_edition_constraint(session)


if __name__ == "__main__":
    db_paths = sys.argv[1:]
    for db_path in db_paths:
        print(f"\nAdding question_set_edition_id to {db_path}")
        add_question_set_edition_id_to_question(f"sqlite:///{db_path}")
