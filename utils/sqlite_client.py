# %%
import sqlite3 as sq
from collections import defaultdict

import pandas as pd
from tabulate import tabulate

import queries


def print_table(df: pd.DataFrame):
    print(tabulate(df, headers="keys", tablefmt="psql"))


class DBClient:
    def __init__(self, db_path: str):
        self.path = db_path
        self.con = sq.connect(db_path)

    def Q(self, q: str):
        return pd.read_sql(q, self.con)

    def __call__(self, query: str):
        return self.Q(query)

    def get_table(self, table_name: str, n_rows: int = -1):
        q = f"SELECT * FROM {table_name}"
        if n_rows > 0:
            q += f" LIMIT {n_rows}"
        df = self.Q(q)
        if "id" in df.columns and "index" in df.columns:
            assert (df["id"] == df["index"]).all()
            df = df.set_index("index")
        elif "id" in df.columns:
            df = df.set_index("id")
        return df

    def list_duplicates(self, table_name: str, column_names: str = "slug"):
        df = self.Q(f"SELECT * FROM {table_name};").set_index("id")
        if isinstance(column_names, str):
            column_names = [column_names]

        # List the pairs of rows that are duplicates
        duplicates = df[df[column_names].duplicated(keep=False)]
        return duplicates

    def table_head(self, table_name: str, n_rows: int = 10):
        return self.Q(f"SELECT * FROM {table_name} LIMIT {n_rows};")

    def get_tossups_info(self):
        return self.Q(queries.TOSSUP_INFO_QUERY).set_index("id")

    def get_game_info(self):
        return self.Q(queries.GAME_INFO_QUERY).set_index("id")

    def get_player_info(self):
        return self.Q(queries.PLAYER_INFO_QUERY).set_index("id")

    def get_buzzpoints_info(self):
        df = (
            self.Q(queries.BUZZPOINTS_INFO_QUERY)
            .set_index("id")
            .drop_duplicates(subset=["player_id", "tossup_id", "buzz_position"])
        )
        df = df[~df["buzz_position"].isna()]
        df["buzz_position"] = df["buzz_position"].astype(int)
        return df


def check_subset(entity_name: str, db1: DBClient, db2: DBClient):
    table_name = f"buzzpoints_{entity_name}"
    print(f"Checking if {table_name} in {db1.path} is a subset of {db2.path}")
    t1 = db1.get_table(table_name)
    t2 = db2.get_table(table_name)
    print(len(t1), len(t2))

    # Assert that all t1.slug are in t2.slug
    assert (t1["slug"].isin(t2["slug"])).all()

    # assert that all other columns are the same too
    columns_to_check = [
        c for c in t1.columns.difference(["id", "index"]) if not c.endswith("_id")
    ]
    for slug in t1["slug"]:
        entry1 = t1[t1["slug"] == slug].reset_index()
        entry2 = t2[t2["slug"] == slug].reset_index()
        if len(entry2) > 1:
            # print(f"Duplicate slugs found in {table_name}: {slug}")
            entry2 = entry2[:1]
        v1 = entry1[columns_to_check].values
        v2 = entry2[columns_to_check].values
        try:
            assert (v1 == v2).all()
        except AssertionError:
            # List the columns that are different.
            print("\nValues different for slug:", slug)
            for col in columns_to_check:
                v1 = entry1[col].iloc[0]
                v2 = entry2[col].iloc[0]
                if v1 != v2:
                    print(f"{col}: {v1} != {v2}")
        except ValueError as e:
            print(e)
            print(entry1)
            print(entry2)


# %%
# for entity_name in ["player", "team", "question", "question_set_edition"]:
#     check_subset(entity_name, regs24, sst24)


def list_dups(db: DBClient, table_name: str, columns: list[str]):
    try:
        dups = db.list_duplicates(table_name, columns)
    except Exception as e:
        print(f"Error listing duplicates for {table_name} in {db.path}: {e}")
        return
    if len(dups) > 0:
        print_table(dups)


nats24 = DBClient("./data/nats24.db")
sst24 = DBClient("./data/sst-23-24-cleaned.db")
acf24 = DBClient("./data/acf-23-24.db")


for db in [sst24, nats24, acf24]:
    print("Listing duplicates for", db.path)
    list_dups(db, "team", ["slug", "tournament_id"])
    list_dups(db, "player", ["slug", "team_id"])

slug_name_columns = [
    ("category_slug", "category"),
    ("subcategory_slug", "subcategory"),
    ("category_main_slug", "category_main"),
]


slug_map = defaultdict(set)

for db in [sst24, nats24]:
    for slug_col, name_col in slug_name_columns:
        df = db.get_table("question")
        for name, slug in df[[name_col, slug_col]].values:
            slug_map[slug].add(name)

# %%
for slug, names in slug_map.items():
    if len(names) > 1:
        print(f"{slug: >20} -> {names}")

# %%
if name == "__main__":
    game_df = nats24(GAME_INFO_QUERY)
    assert game_df["id"].nunique() == game_df.shape[0]
    game_df = game_df.set_index("id")
    game_df

    # %%
    player_df = nats24(PLAYER_INFO_QUERY)
    assert player_df["id"].nunique() == player_df.shape[0]
    player_df = player_df.set_index("id")
    player_df

    # %%
    tossup_df = nats24(TOSSUP_INFO_QUERY)
    assert tossup_df["id"].nunique() == tossup_df.shape[0]
    tossup_df = tossup_df.set_index("id")
    tossup_df

    # %%
    bonus_part_df = nats24(BONUS_PART_INFO_QUERY)
    assert bonus_part_df["id"].nunique() == bonus_part_df.shape[0]
    bonus_part_df = bonus_part_df.set_index("id")
    bonus_part_df

    # %%

    # Check if column bonus_id and question_id are one-to-one in bonus_part_df
    bonus_part_df[["bonus_id"]].shape[0]

    # %%
    bonus_df = nats24.get_table("bonus")
    bonus_df[["question_id"]].drop_duplicates().shape[0]
    # %%

    set(bonus_part_df["slug"]).intersection(set(tossup_df["slug"]))
    # %%
    nats24.get_table("question")["metadata"][1]
    # %%

    bonus_part_df.iloc[1]["leadin"]
    # %%
