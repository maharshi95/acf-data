# %%
import importlib
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate

import models
import utils
from utils import acf_sanitization

acf_sanitization = importlib.reload(acf_sanitization)
from utils.acf_sanitization import (
    get_buzz_offset,
    sanitize_question,
    tokenize,
)
from utils.qb_tokenization import get_clue_spans
from utils.sqlite_client import DBClient

# %%
session = models.create_session("./data/acf-23-24.db")
db = DBClient("./data/acf-23-24.db")
# %%


def get_char_index(text: str, token_index: int):
    """
    Get the character index of the 0-indexed token_index-th token in the text.
    """
    tokens = text.split()[: token_index + 1]
    return sum(len(token) for token in tokens) + token_index


def get_quizbowlstats_url(tossup_id: int):
    buzz = (
        buzz_df[buzz_df["tossup_id"] == tossup_id]
        .sort_values("buzz_position", ascending=False)
        .iloc[0]
    )
    game_id = buzz["game_id"]
    game = game_df.loc[game_id]
    qset_id = game["qset"]
    question_slug = tossup_df.loc[tossup_id]["slug"]
    return f"https://quizbowlstats.com/buzzpoints/set/{qset_id}/tossup/{question_slug}"


players = (
    session.query(models.Player)
    .with_entities(models.Player.slug, models.Player.name)
    .distinct()
    .all()
)
print(len(players))
# %%
players_df = db.get_player_info()
tossup_df = db.get_tossups_info()
game_df = db.get_game_info()
buzz_df = db.get_buzzpoints_info()

df_shape_data = [
    ["Player", db.get_table("player").shape[0], players_df.shape[0]],
    ["Buzz", db.get_table("buzz").shape[0], buzz_df.shape[0]],
    ["Tossup", db.get_table("tossup").shape[0], tossup_df.shape[0]],
    ["Game", db.get_table("game").shape[0], game_df.shape[0]],
]

headers = ["DataFrame", "Orig # rows", "Joined # rows"]
table = tabulate(df_shape_data, headers=headers, tablefmt="table")
print(table)

print("# Unique questions in tossup_df:", tossup_df["question_id"].nunique())

# %%
tossup_df["question_sanitized"] = tossup_df["question"].apply(sanitize_question)
# Filter questions with Note to moderator
print(
    "# Questions starting with <em>:",
    len(tossup_df[tossup_df["question_sanitized"].str.startswith("<em>")]),
)
# Add these back in
# tossup_df["spans"] = tossup_df["question_sanitized"].apply(get_clue_spans)
# tossup_df["n_clues"] = tossup_df["spans"].apply(len)

# # Distribution of n_clues
# plt.hist(tossup_df["n_clues"], bins=100)
# plt.show()

# %%

tossup_n_tokens = tossup_df["question"].apply(lambda x: len(tokenize(x)))
tossup_buzz_offsets = tossup_df["question"].apply(get_buzz_offset)

# Create a Series of buzz offsets indexed by tossup_id
tossup_buzz_offsets_series = tossup_buzz_offsets.reindex(tossup_df.index)

# Subtract the buzz offset from the buzz position
buzz_df["buzz_position"] = (
    buzz_df["buzz_position"] - buzz_df["tossup_id"].map(tossup_buzz_offsets_series)
).clip(lower=0)

buzz_df["buzz_position_char"] = buzz_df.apply(
    lambda x: get_char_index(
        tossup_df.loc[x["tossup_id"]]["question_sanitized"], x["buzz_position"]
    ),
    axis=1,
)
print("Buzz positions adjusted for instruction offsets")


# Group tossup_ids and get the maximum buzz position as integer
max_buzz_positions = (
    buzz_df.groupby("tossup_id")["buzz_position"]
    .max()
    .reset_index()
    .set_index("tossup_id")["buzz_position"]
    .astype(int)
    .sort_index()
)


print("# Unique Tossups:", buzz_df["tossup_id"].nunique())
print("# Unique Questions:", tossup_df["question_id"].nunique())

set(buzz_df["tossup_id"].values) - set(tossup_df.index)

# %%

# Convert each to dict and check if the values are in the range of 2
n_tokens_dict = tossup_n_tokens.to_dict()
max_pos_dict = max_buzz_positions.to_dict()
# Check if both dictionaries have the same keys
diff_keys_l = set(max_pos_dict.keys()) - set(n_tokens_dict.keys())
diff_keys_r = set(n_tokens_dict.keys()) - set(max_pos_dict.keys())
assert set(max_pos_dict) <= set(n_tokens_dict), "Keys do not match"


# Find the number of keys in max_pos_dict such that the values are not within 2 of n_tokens_dict
n_exact_matches = 0
n_t2_matches = 0
n_t5_matches = 0
n_invalid = 0
large_diffs = []
tossup_pos_diff_map = {}
for k, v in max_pos_dict.items():
    diff = n_tokens_dict[k] - v
    abs_diff = abs(diff)
    tossup_pos_diff_map[k] = abs_diff
    if abs_diff == 0:
        n_exact_matches += 1
    elif abs_diff <= 2:
        n_t2_matches += 1
    elif abs_diff <= 5:
        n_t5_matches += 1
    elif diff < 0:
        n_invalid += 1
        print(get_quizbowlstats_url(k))
        print("Max Buzz Position:", v, "Number of Tokens:", n_tokens_dict[k])
        question = tossup_df.loc[k]["question"]
        tokens = tokenize(question)
        print(question, "\n")
        print(tokens, "\n")
    else:
        large_diffs.append(diff)
        # print(get_quizbowlstats_url(k))
        # print("Max Buzz Position:", v, "Number of Tokens:", n_tokens_dict[k])
        # print(tossup_df.loc[k]["question"], "\n")

print()
print(f"# Invalid: {n_invalid / len(max_pos_dict) * 100:.2f}%")
print()
print(f"% Matches: {n_exact_matches / len(max_pos_dict) * 100:.2f}")
print(f"% Matches within 2: {n_t2_matches / len(max_pos_dict) * 100:.2f}")
print(f"% Matches within 5: {n_t5_matches / len(max_pos_dict) * 100:.2f}")
print(f"% Large Diff: {len(large_diffs) / len(max_pos_dict) * 100:.2f}")
print(f"Median Large Diff: {np.median(large_diffs)}")
# %%


tossup_ids_by_diff = defaultdict(list)
for k, v in tossup_pos_diff_map.items():
    bucket = v // 5
    tossup_ids_by_diff[bucket].append(k)

for k, V in sorted(tossup_ids_by_diff.items(), key=lambda x: x[0]):
    print(f"{k:2d}: {len(V)}")


diff_value = 22
n_samples = min(10, len(tossup_ids_by_diff[diff_value]))
rs = np.random.RandomState(0)
tossup_samples = rs.choice(tossup_ids_by_diff[diff_value], n_samples, replace=False)
for t in tossup_samples:
    print(get_quizbowlstats_url(t))
    print("Max Buzz Position:", max_pos_dict[t], "Number of Tokens:", n_tokens_dict[t])
    print(tossup_df.loc[t]["question"], "\n")


# Example usage:
# display_interactive_tokens(tokens)

# %%

buzz_df_dedup = buzz_df.drop_duplicates(
    subset=["player_id", "tossup_id", "buzz_position"]
).sort_values("buzz_position")
# Group by player_id and tossup_id and list entries with duplicates
duplicates = buzz_df_dedup.groupby(["player_id", "tossup_id"]).filter(
    lambda x: len(x) > 1
)
if not duplicates.empty:
    print("# Entries with duplicates:", duplicates.shape[0])
    for _, group in duplicates.groupby(["player_id", "tossup_id"]):
        player_id, tossup_id = group.iloc[0]["player_id"], group.iloc[0]["tossup_id"]
        buzz_positions = group["buzz_position"].tolist()
        values = group["value"].tolist()
        print(f"({player_id}, {tossup_id}) -> {buzz_positions} ({values})")
else:
    print("No duplicates found.")

print()
duplicates = buzz_df_dedup.groupby(["team", "tossup_id"]).filter(lambda x: len(x) > 1)
if not duplicates.empty:
    print("# Entries with duplicates:", duplicates.shape[0])
    for _, group in duplicates.groupby(["team", "tossup_id"]):
        team, tossup_id = group.iloc[0]["team"], group.iloc[0]["tossup_id"]
        players = group["player_id"].tolist()
        buzz_positions = group["buzz_position"].tolist()
        values = group["value"].tolist()
        print(f"({team}, {tossup_id}) -> {players} ({buzz_positions})")
else:
    print("No duplicates found.")


# plot player activity
# plt.plot(player_activity)


# plot tossup activity
# plt.plot(tossup_activity)

# %%

db2 = DBClient("./data/nats24.db")
players_df2 = db2.get_player_info()
tossup_df2 = db2.get_tossups_info()
game_df2 = db2.get_game_info()
buzz_df2 = db2.get_buzzpoints_info()

# %%
players_df = db.get_player_info()
# %%

# Check which player.slug are in db2 but not in db
set(players_df2["slug"]).intersection(set(players_df["slug"]))

# %%

session1 = models.create_session("./data/acf-23-24.db")
session2 = models.create_session("./data/sst-23-24-cleaned.db")
# %%
# List all buzzes from acf24 such that the player slug is X
acf_buzzes = (
    session1.query(models.Buzz)
    .filter(models.Buzz.player.has(slug="jeffrey-deremo"))
    .all()
)
acf_buzzes.sort(key=lambda x: (x.value, x.buzz_position))
for b in acf_buzzes:
    print(b.value, b.buzz_position)
print("- " * 50)
sst_buzzes = (
    session2.query(models.Buzz)
    .filter(models.Buzz.player.has(slug="jeffrey-deremo"))
    .all()
)
sst_buzzes.sort(key=lambda x: (x.value, x.buzz_position))
for b in sst_buzzes:
    print(b.value, b.buzz_position)

# %%
print(len(acf_buzzes), len(sst_buzzes))
# %%

session2.query(models.Buzz).filter(models.Buzz.player.has(id=1429)).all()
# %%
