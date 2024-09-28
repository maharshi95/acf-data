# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score

import models
from utils.sqlite_client import DBClient

# %%
db = DBClient("./data/acf-23-24.db")
full_buzz_df = db.get_buzzpoints_info()
tossup_df = db.get_tossups_info()

# Filter df such that tournament startswith 2024
buzz_df = full_buzz_df[full_buzz_df["tournament"].str.startswith("20")]

# List all tournaments in the df
tournaments = buzz_df["tournament"].unique()
print(tournaments)

# List categories and their counts in a beautiful way
# Get category counts and sort them in descending order
category_counts = (
    buzz_df["question_category"].value_counts().sort_values(ascending=False)
)


# List category, subcategory, and category main for each entry:
def non_redundant_category_main(cat, subcat, cat_main):
    if cat_main == f"{cat}-{subcat}":
        return False
    if cat_main == subcat:
        return False
    return True


def create_category_main(cat, sub_cat):
    if sub_cat.startswith(f"{cat}") or sub_cat.endswith(f"{cat}") or sub_cat == cat:
        return sub_cat
    return f"{cat}-{sub_cat}"


tossup_df["category_main"] = tossup_df.apply(
    lambda x: create_category_main(x["category"], x["subcategory"]), axis=1
)

# Create a beautiful display of categories and their counts for unique tossups
unique_tossups = buzz_df.drop_duplicates(subset=["tossup_id"])
unique_tossups["category_main"] = unique_tossups.apply(
    lambda x: create_category_main(x["question_category"], x["question_subcategory"]),
    axis=1,
)
category_counts = (
    unique_tossups["category_main"].value_counts().sort_values(ascending=False)
)

print("\n🏆 Unique Tossup Category Distribution 🏆")
print("=" * 50)
for category, count in category_counts.items():
    bar_length = int(count / max(category_counts) * 20)
    bar = "█" * bar_length + "▒" * (20 - bar_length)
    print(f"{category:<40} | {bar} | {count:>5}")
print("=" * 50)
print(f"Total Unique Tossup Categories: {len(category_counts)}")
print(f"Total Unique Tossups: {sum(category_counts)}")


# %%
player_key = "player"

n_players = buzz_df[player_key].nunique()
n_tossups = buzz_df["tossup_id"].nunique()

print(f"Number of players: {n_players}")
print(f"Number of tossups: {n_tossups}")

player_ids = buzz_df[player_key].unique()
tossup_ids = buzz_df["tossup_id"].unique()

# Create a DataFrame with players as index and tossups as columns
matrix_df = pd.DataFrame(0, index=player_ids, columns=tossup_ids, dtype=bool)

# Fill the DataFrame with 1s where a player buzzed on a tossup
for _, row in buzz_df.iterrows():
    matrix_df.loc[row[player_key], row["tossup_id"]] = True

k_connected_number = 1
matrix_df = (matrix_df @ (matrix_df.T @ matrix_df) ** k_connected_number) > 0


# Calculate player and tossup activity
player_activity = matrix_df.sum(axis=1)
tossup_activity = matrix_df.sum(axis=0)

# Convert to numpy array if needed for further operations
matrix = matrix_df.values

player_permutation = np.argsort(player_activity)[::-1]
tossup_permutation = np.argsort(tossup_activity)[::-1]
new_matrix = matrix[player_permutation, :][:, tossup_permutation]

# %%
# List tossup ids and difficulty for counts < 100
# Filter tossups with activity count < 100
low_activity_tossups = tossup_activity[tossup_activity < 300]

# Get the corresponding tossup IDs
low_activity_tossup_ids = set(low_activity_tossups.index.tolist())

# Filter unique_tossups DataFrame for these tossup IDs
low_activity_details = unique_tossups[
    unique_tossups["tossup_id"].isin(low_activity_tossup_ids)
]

low_activity_details.tournament
# Optional: Save to CSV for further analysis
# low_activity_details.to_csv('low_activity_tossups.csv', index=False)


# %%
plt.spy(new_matrix, aspect="auto", markersize=0.01)
plt.show()
# %%
print("Density:", 100 * new_matrix.sum() / new_matrix.size)
# %%
# Biclustering of new_matrix


# Define the number of clusters (you may need to adjust these)
n_row_clusters = 6
n_col_clusters = 6
n_clusters = (n_row_clusters, n_col_clusters)  # (n_row_clusters, n_column_clusters)

# Perform biclustering
model = SpectralBiclustering(
    n_clusters=n_clusters, method="bistochastic", random_state=0
)
model.fit(new_matrix)

# Get row and column labels
row_labels = model.row_labels_
col_labels = model.column_labels_


def mean_row_cluster_vector(cluster_id):
    vectors = new_matrix[row_labels == cluster_id]
    return vectors.mean(axis=0)


def mean_col_cluster_vector(cluster_id):
    vectors = new_matrix[:, col_labels == cluster_id]
    return vectors.mean(axis=1)


# Reorder the row_labels and col_labels such that cluster with similar mean row vector is next to each other
# Step 1: Calculate mean row vectors for each cluster
row_cluster_vectors = [mean_row_cluster_vector(i) for i in range(n_row_clusters)]

# Step 2: Calculate pairwise distances between cluster vectors
from scipy.spatial.distance import pdist, squareform

pairwise_distances = pdist(row_cluster_vectors, metric="euclidean")
distance_matrix = squareform(pairwise_distances)

# Step 3: Perform hierarchical clustering on the distance matrix
from scipy.cluster.hierarchy import dendrogram, linkage

linkage_matrix = linkage(distance_matrix, method="ward")

# Step 4: Get the leaf order from the dendrogram
dendrogram_order = dendrogram(linkage_matrix, no_plot=True)["leaves"]

# Step 5: Create a mapping from old cluster labels to new ordered labels
cluster_order_map = {old: new for new, old in enumerate(dendrogram_order)}

# Step 6: Reorder the row_labels based on the new ordering
new_row_labels = np.array([cluster_order_map[label] for label in row_labels])

# Step 7: Reorder the col_labels (optional, but can be done similarly if needed)
col_cluster_vectors = [mean_col_cluster_vector(i) for i in range(n_col_clusters)]
col_pairwise_distances = pdist(col_cluster_vectors, metric="euclidean")
col_distance_matrix = squareform(col_pairwise_distances)
col_linkage_matrix = linkage(col_distance_matrix, method="ward")
col_dendrogram_order = dendrogram(col_linkage_matrix, no_plot=True)["leaves"]
col_cluster_order_map = {old: new for new, old in enumerate(col_dendrogram_order)}
new_col_labels = np.array([col_cluster_order_map[label] for label in col_labels])


# Rearrange the data
fit_data = new_matrix[np.argsort(new_row_labels), :]
fit_data = fit_data[:, np.argsort(new_col_labels)]

# Plot the biclustered matrix
plt.figure(figsize=(12, 8))
plt.imshow(fit_data, aspect="auto", cmap="binary", interpolation="nearest")
# plt.spy(fit_data, aspect="auto", markersize=0.2)
plt.title("Biclustered Matrix")
plt.colorbar()

# Add lines to separate clusters
for i in range(1, n_row_clusters):
    plt.axhline(y=np.sum(new_row_labels < i) - 0.5, color="r", linestyle="--")
for i in range(1, n_col_clusters):
    plt.axvline(x=np.sum(new_col_labels < i) - 0.5, color="r", linestyle="--")

plt.show()
clusters = []
# Print some statistics about the clusters
for i in range(n_clusters[0]):
    for j in range(n_clusters[1]):
        cluster = fit_data[new_row_labels == i][:, new_col_labels == j]
        clusters.append(cluster)

clusters.sort(key=lambda x: x.mean(), reverse=True)
for cluster in clusters[:5]:
    print(f"Cluster ({i}, {j}):")
    print(f"  Size: {cluster.shape}")
    print(f"  Density: {cluster.mean() * 100:.2f}")
    print()

# %%
