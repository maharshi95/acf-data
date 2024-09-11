import os
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Type

import sqlalchemy
from loguru import logger
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session

import models
from models import Base, all_classes, create_session
from utils.viz_utils import DiffVisualizer


def get_class_dependencies() -> Dict[str, set]:
    """
    Create a graph of class dependencies based on PK/FK relationships.
    :return: A dictionary representing the graph.
    """
    dependencies = defaultdict(set)

    # List of all classes
    classes = list(all_classes)

    # Add edges to the graph
    for cls in classes:
        mapper = inspect(cls)
        for column in mapper.columns:
            if column.foreign_keys:
                for fk in column.foreign_keys:
                    fk_class = fk.column.table.name
                    dependencies[fk_class].add(cls.__tablename__)

    return dependencies


def topo_sort_classes(dependencies: Dict[str, List[str]]) -> List[str]:
    """
    Perform topological sort on the classes based on their PK/FK dependencies.
    :return: A list of class objects in topologically sorted order.
    """
    in_degree = {t: 0 for t in dependencies}

    # Calculate in-degrees
    for table in dependencies:
        for dependent in dependencies[table]:
            in_degree.setdefault(dependent, 0)
            in_degree[dependent] += 1

    # Initialize the queue with classes having zero in-degree
    queue = deque([table for table in in_degree if in_degree[table] == 0])
    sorted_tables = []

    while queue:
        table = queue.popleft()
        sorted_tables.append(table)

        for dependent in dependencies[table]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_tables) != len(in_degree):
        raise ValueError("There exists a cycle in the dependencies")

    return sorted_tables


def merge_databases(source_db_path: str, target_db_path: str) -> None:
    """
    Merge records from a source database into a target database.
    """
    session_source = create_session(source_db_path)
    session_target = create_session(target_db_path, create_tables=True)

    dependencies = get_class_dependencies()
    sorted_tables = topo_sort_classes(dependencies)
    logger.info("Topologically sorted classes: %s", sorted_tables)

    name_to_class = {cls.__tablename__: cls for cls in all_classes}
    db_id_mapping = {}

    # Merge each table in topologically sorted order
    for table_name in sorted_tables:
        base_class = name_to_class[table_name]
        table_id_mapping = merge_table(
            session_source, session_target, base_class, db_id_mapping
        )

        # Update id_mapping with the new ids
        db_id_mapping[table_name] = table_id_mapping

    session_target.commit()
    session_target.close()
    session_source.close()


def create_diff_dict(old_record: Base, new_record: Base) -> Dict[str, Dict[str, Any]]:
    """
    Create a dictionary of differences between old_record and new_record.
    The dictionary will contain both old and new values for each differing field.
    """
    diff_dict = {}
    for column in old_record.__table__.columns:
        if column.primary_key:
            continue
        old_value = getattr(old_record, column.name)
        new_value = getattr(new_record, column.name)
        if old_value != new_value:
            diff_dict[column.name] = {"old": old_value, "new": new_value}
    return diff_dict


def merge_table(
    session_from: Session,
    session_to: Session,
    model_cls: Type[Base],
    db_id_mapping: Dict[str, Dict[int, int]],
) -> Dict[int, int]:
    """Merge records from a source database session into a target database session for a given model class.

    Pre-conditions:
      - Both sessions (session_from and session_to) must be valid and open SQLAlchemy sessions.
      - model_cls must be a SQLAlchemy model class derived from Base.
      - The tables corresponding to model_cls must exist in both databases.
      - The tables in both source and target databases must have the same schema.
      - Any foreign key relationships in model_cls must correspond to tables that
        have already been merged.

    Args:
      session_from: SQLAlchemy session for the source database
      session_to: SQLAlchemy session for the target database
      model_cls: The SQLAlchemy model class representing the table to be merged

    Returns:
      A dictionary mapping original record IDs to new IDs in the target database
    """

    # Get all records from both databases
    logger.info(f"Merging table: {model_cls.__tablename__}")
    records_from = session_from.query(model_cls).all()
    records_to = session_to.query(model_cls).all()

    logger.info(f"# Records in target db: {len(records_to)}")
    logger.info(f"# Records in source db: {len(records_from)}")

    table_id_mapping = {}

    for record in records_from:
        # Create a dictionary of non-PK and non-FK fields
        record_data = {}
        for col in model_cls.__table__.columns:
            if col.primary_key:
                continue
            if col.foreign_keys:
                # Use id_mapping to get the new foreign key
                assert len(col.foreign_keys) == 1, (
                    "Expected 1 foreign key, got composite foreign key of length "
                    f"{len(col.foreign_keys)} for column {col.name}"
                )
                fk = list(col.foreign_keys)[0]
                related_table = fk.column.table.name
                related_id = getattr(record, col.key)
                record_data[col.key] = db_id_mapping[related_table][related_id]
            else:
                record_data[col.key] = getattr(record, col.key)

        existing_record = record_exists(session_to, model_cls, record_data)

        new_record = model_cls(**record_data)
        if existing_record is None:
            session_to.add(new_record)
            session_to.flush()
            table_id_mapping[record.id] = new_record.id
        else:
            diff_dict = create_diff_dict(existing_record, new_record)
            if diff_dict:
                logger.info(
                    f"New record with id {new_record.id} has different values for: "
                    f"\n{DiffVisualizer(diff_dict)}"
                )
            table_id_mapping[record.id] = existing_record.id

    records_to = session_to.query(model_cls).all()
    logger.info(f"# Records in target db: {len(records_to)}")

    return table_id_mapping


def record_exists(
    session: Session, model_cls: Type[Base], record_data: Dict[str, Any]
) -> Optional[Base]:
    """
    Check if a record exists in the database and return its primary key, else return None.
    """
    # Get unique constraints for the table
    unique_constraints = model_cls.__table__.constraints
    unique_columns = set()
    for constraint in unique_constraints:
        if isinstance(constraint, sqlalchemy.UniqueConstraint):
            unique_columns.update(constraint.columns)

    # If no unique constraints, fall back to all non-primary key columns
    unique_columns = unique_columns or model_cls.non_pk_columns()

    # Build the filter condition based on unique columns
    filter_conditions = []
    for column in unique_columns:
        value = record_data[column.key]
        if value is not None:
            filter_conditions.append(getattr(model_cls, column.key) == value)

    # Query the database
    existing_record = session.query(model_cls).filter(*filter_conditions).first()

    return existing_record


# %%
if __name__ == "__main__":
    # Run the merge
    src_db_paths = [
        "data/sst-23-24-cleaned.db",
        "data/nats24.db",
    ]
    merged_db_path = "data/acf-23-24.db"

    if os.path.exists(merged_db_path):
        os.remove(merged_db_path)

    for src_db_path in src_db_paths:
        merge_databases(src_db_path, merged_db_path)

    session = create_session(merged_db_path)
    t = session.query(models.Tossup).first()
    print(t.question_text)
    print(t.question)

    print("\n\nDuplicate Questions\n")
    questions = session.query(models.Question).all()
    len(questions)

    question_by_slug = defaultdict(list)
    for q in questions:
        question_by_slug[q.slug + q.question_metadata + q.category_full].append(q)

    for slug, qs in question_by_slug.items():
        if len(qs) > 1:
            print(slug)
            for q in qs:
                print(q)
            print()

    print("\n\nDuplicate Teams\n")
    teams = session.query(models.Team).all()
    teams_by_uk = defaultdict(list)
    for team in teams:
        teams_by_uk[team.unique_key()].append(team)

    for slug, ts in teams_by_uk.items():
        if len(ts) > 1:
            print(slug)
            for t in ts:
                print(t)
            print()
