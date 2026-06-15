"""Build collaboration summaries and visualizations for quiz bowl teams."""

# %%
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from rich import print as rprint

import core.models as models

DB_FILES = [
    "data/dbs/co24.db",
    "data/dbs_raw/acf-24-25.db",
]

FIGURES_DIR = Path("docs/stats")


def get_player_info(player: models.Player):
    team = player.team
    other_players = [p.slug for p in team.players] if team else []
    info = {
        "name": player.name,
        "team_slug": team.slug if team else None,
        "team": team.name if team else None,
        "other_players": other_players,
        "slug": player.slug,
        "qset": player.question_set.name,
        "tournament": team.tournament.slug,
    }
    return info


def get_players_info(player_slug, db_path):
    session = models.create_session(db_path)
    players = session.query(models.Player).filter_by(slug=player_slug).all()
    play_infos = [get_player_info(p) for p in players]
    session.close()
    return play_infos


def distinguish_players_by_slug(player1slug, player2slug, db1_path, db2_path=None):
    db2_path = db2_path or db1_path
    pinfo1 = get_players_info(player1slug, db1_path)
    pinfo2 = get_players_info(player2slug, db2_path)
    print(f"Comparing player slugs '{player1slug}' and '{player2slug}' from databases:")
    for info in pinfo1:
        rprint(info)
    for info in pinfo2:
        rprint(info)


distinguish_players_by_slug("ethan", "ethan-3", "data/dbs_raw/acf-24-25.db")

# %%


def load_player_records(db_paths, verbose=False):
    """Load all tournament/team/player records from the provided database files.

    Args:
        db_paths: List of database file paths to load
        verbose: If True, print progress messages during loading
    """
    records = []
    for db_path in db_paths:
        if verbose:
            print(f"Loading data from {db_path}...")
        session = models.create_session(db_path)
        tournaments = (
            session.query(models.Tournament)
            .order_by(models.Tournament.start_date)
            .all()
        )
        if verbose:
            print(f"  Found {len(tournaments)} tournament(s)")
        for tournament in tournaments:
            tourney_label = tournament.slug or tournament.name
            # Extract year from start_date if available
            year = tournament.start_date.year if tournament.start_date else None
            for team in tournament.teams:
                team_label = team.slug or team.name
                team_id = f"{tourney_label}:{team_label}"
                for player in team.players:
                    player_slug = player.slug or player.name
                    records.append(
                        {
                            "db": db_path,
                            "tournament": tourney_label,
                            "tournament_year": year,
                            "tournament_date": tournament.start_date,
                            "team": team_id,
                            "team_label": team_label,
                            "player_slug": player_slug,
                            "player_name": player.name,
                            "question_set": tournament.question_set_edition.question_set.name,
                        }
                    )
        session.close()

    players_df = pd.DataFrame(records)
    if players_df.empty:
        raise RuntimeError("No player data available in the provided databases")
    if verbose:
        print(
            f"Loaded {len(players_df)} player records from {len(db_paths)} database(s)"
        )
    return players_df


players_df = load_player_records(DB_FILES, verbose=True)
player_slugs = {r["player_slug"]: r["player_name"] for i, r in players_df.iterrows()}
player_name_group = defaultdict(set)
for slug, name in player_slugs.items():
    player_name_group[name].add(slug)
for name, slugs in player_name_group.items():
    s = {s.removesuffix(f"-2").removesuffix(f"-3") for s in slugs}
    if len(s) > 1:
        print(f"Player name '{name}' has multiple base slugs: {', '.join(s)}")
for s in player_slugs:
    if s.endswith("-2") or s.endswith("-3"):
        assert s[:-2] in player_slugs, f"Slug '{s}' has no base slug '{s[:-2]}'"
        if player_slugs[s] != player_slugs[s[:-2]]:
            print(
                f"Slug '{s}' name '{player_slugs[s]}' does not match base slug '{s[:-2]}' name '{player_slugs[s[:-2]]}'"
            )
len(players_df), len(player_slugs), len(player_name_group)
# %%


def create_tournament_lookup(db_paths):
    """Create a lookup map for tournaments with date and year metadata.

    Args:
        db_paths: List of database file paths

    Returns:
        Dictionary mapping tournament slug to metadata
    """
    tournament_map = {}

    for db_path in db_paths:
        session = models.create_session(db_path)
        tournaments = session.query(models.Tournament).all()

        for tournament in tournaments:
            slug = tournament.slug or tournament.name
            year = tournament.start_date.year if tournament.start_date else None
            tournament_map[slug] = {
                "name": tournament.name,
                "slug": slug,
                "start_date": tournament.start_date,
                "end_date": tournament.end_date,
                "year": year,
                "level": tournament.level,
                "location": tournament.location,
                "question_set": tournament.question_set_edition.question_set.name,
                "db": db_path,
            }

        session.close()

    return tournament_map


def inspect_duplicate_player_slugs(players_df):
    """Identify player slugs with a question set appearing more than once."""
    duplicate_qsets = (
        players_df.groupby(
            ["player_slug", "question_set"]
        )  # pylint: disable=maybe-no-member
        .size()
        .reset_index(name="count")
    )
    duplicate_qsets = duplicate_qsets[duplicate_qsets["count"] > 1]

    if duplicate_qsets.empty:
        print("No duplicate player slug question set entries found.")
        return []

    duplicate_slugs = duplicate_qsets["player_slug"].unique().tolist()
    print(f"Found {len(duplicate_slugs)} player slugs with repeated question sets:")
    for slug in duplicate_slugs:
        entries = duplicate_qsets[duplicate_qsets["player_slug"] == slug]
        qsets = [
            f"{row['question_set']} ({row['count']})" for _, row in entries.iterrows()
        ]
        print(f"Slug: {slug} - Repeated Question Sets: {', '.join(qsets)}")

    return duplicate_slugs


def report_slug_inconsistencies(players_df, verbose=True):
    """Surface players who share a name but appear under multiple slugs.

    Args:
        players_df: DataFrame with player records
        verbose: If True, print inconsistency report
    """
    print(players_df.columns)
    slug_variants = (
        players_df.groupby("player_name")["player_slug"]
        .nunique()
        .reset_index(name="slug_count")
    )
    multi_slug_players = slug_variants[slug_variants.slug_count > 1][
        "player_name"
    ].tolist()

    if not verbose:
        return multi_slug_players

    if not multi_slug_players:
        print("No players with multiple slugs detected.")
        return multi_slug_players

    inconsistent = (
        players_df[players_df.player_name.isin(multi_slug_players)]
        .groupby(["player_name", "player_slug"])
        .agg(
            {
                "tournament": lambda vals: sorted(set(vals)),
                "question_set": lambda vals: sorted(set(vals)),
            }
        )
        .reset_index()
    )
    print("Players with multiple slugs:")
    for name, group in inconsistent.groupby("player_name"):
        slugs = "\n\t\t".join(
            f"{row.player_slug}\t: ["
            + ", ".join(row.question_set)
            + f"] {', '.join(row.tournament)}"
            for _, row in group.iterrows()
        )
        print(f"  {name}:\n\t\t{slugs}")
    print()


def deduplicate_teams(players_df, verbose=False):
    """Collapse teams that share identical rosters across tournaments.

    Args:
        players_df: DataFrame with player records
        verbose: If True, print deduplication statistics
    """
    raw_team_to_players = {
        team_id: set(group.player_slug) for team_id, group in players_df.groupby("team")
    }
    if verbose:
        print(f"Processing {len(raw_team_to_players)} unique team entries...")

    team_meta = (
        players_df[["team", "team_label", "tournament"]]
        .drop_duplicates("team")
        .set_index("team")
        .to_dict("index")
    )

    roster_groups = defaultdict(list)
    for team_id, roster in raw_team_to_players.items():
        roster_key = tuple(sorted(roster))
        roster_groups[roster_key].append(team_id)

    name_usage = defaultdict(int)
    dedup_records = []
    for roster_key, team_ids in roster_groups.items():
        tournaments_map = {
            team_meta[tid]["tournament"]: team_meta[tid]["team_label"]
            for tid in team_ids
        }
        canonical_name = min(tournaments_map.values())
        name_usage[canonical_name] += 1
        label = (
            canonical_name
            if name_usage[canonical_name] == 1
            else f"{canonical_name}#{name_usage[canonical_name]}"
        )
        dedup_records.append(
            {
                "label": label,
                "canonical_name": canonical_name,
                "player_slugs": list(roster_key),
                "player_count": len(roster_key),
                "tournaments": tournaments_map,
            }
        )

    team_to_players = {rec["label"]: set(rec["player_slugs"]) for rec in dedup_records}
    dedup_df = (
        pd.DataFrame(
            [
                {
                    "team": rec["label"],
                    "player_count": rec["player_count"],
                    "tournaments": rec["tournaments"],
                }
                for rec in dedup_records
            ]
        )
        .sort_values("team")
        .reset_index(drop=True)
    )

    if verbose:
        print(f"Deduplicated to {len(dedup_records)} unique rosters")

    return dedup_records, team_to_players, dedup_df


def summarize_deduped_teams(dedup_df, dedup_records, sample_size=5, verbose=True):
    """Print summary statistics and samples of deduplicated teams.

    Args:
        dedup_df: DataFrame of deduplicated teams
        dedup_records: List of deduplication records
        sample_size: Number of samples to display
        verbose: If True, print detailed summary
    """
    if not verbose:
        return

    print(f"Deduplicated team rosters: {len(dedup_df)}")
    if dedup_df.empty:
        return

    print("Sample deduped teams:")
    for _, row in dedup_df.head(min(sample_size, len(dedup_df))).iterrows():
        print(f"  {row.team} -> tournaments {row.tournaments}")

    multi_entry = [rec for rec in dedup_records if len(rec["tournaments"]) > 1]
    if not multi_entry:
        print("No deduped teams span multiple tournaments.")
        return

    print("\nTeams appearing under multiple tournament entries:")
    for rec in multi_entry[: min(sample_size, len(multi_entry))]:
        print(
            f"  {rec['label']} (players: {rec['player_slugs']}) -> {rec['tournaments']}"
        )


def analyze_temporal_player_overlap(players_df, year_a=2024, year_b=2025, verbose=True):
    """Analyze player overlap between different tournament years.

    Args:
        players_df: DataFrame with player records including tournament_year
        year_a: First year to compare
        year_b: Second year to compare
        verbose: If True, print detailed analysis

    Returns:
        Dictionary with overlap statistics and player lists
    """
    year_a_df = players_df[players_df.tournament_year == year_a]
    year_b_df = players_df[players_df.tournament_year == year_b]

    players_a = set(year_a_df.player_slug.unique())
    players_b = set(year_b_df.player_slug.unique())

    returning_players = players_a & players_b
    new_players = players_b - players_a
    departed_players = players_a - players_b

    tournaments_a = sorted(year_a_df.tournament.unique())
    tournaments_b = sorted(year_b_df.tournament.unique())

    year_b_teams = year_b_df.groupby("team")["player_slug"].apply(set).to_dict()
    fully_returning_teams = []
    partially_returning_teams = []
    fully_new_teams = []

    for team_id, team_players in year_b_teams.items():
        returning_count = len(team_players & players_a)
        total_count = len(team_players)

        if returning_count == total_count:
            fully_returning_teams.append(
                {"team": team_id, "size": total_count, "players": sorted(team_players)}
            )
        elif returning_count > 0:
            partially_returning_teams.append(
                {
                    "team": team_id,
                    "size": total_count,
                    "returning": returning_count,
                    "new": total_count - returning_count,
                    "players": sorted(team_players),
                }
            )
        else:
            fully_new_teams.append(
                {"team": team_id, "size": total_count, "players": sorted(team_players)}
            )

    results = {
        "year_a": year_a,
        "year_b": year_b,
        "tournaments_a": tournaments_a,
        "tournaments_b": tournaments_b,
        "total_a": len(players_a),
        "total_b": len(players_b),
        "returning": len(returning_players),
        "new": len(new_players),
        "departed": len(departed_players),
        "returning_players": sorted(returning_players),
        "new_players": sorted(new_players),
        "departed_players": sorted(departed_players),
        "fully_returning_teams": fully_returning_teams,
        "partially_returning_teams": partially_returning_teams,
        "fully_new_teams": fully_new_teams,
    }

    if verbose:
        print()
        _print_temporal_analysis(results, players_df)
        print()

    return results


def _print_temporal_analysis(results, players_df):
    """Print temporal overlap analysis results."""
    year_a, year_b = results["year_a"], results["year_b"]
    print(f"=== Temporal Player Overlap: {year_a} vs {year_b} ===")
    print(f"\n{year_a} Tournaments: {', '.join(results['tournaments_a'])}")
    print(f"{year_b} Tournaments: {', '.join(results['tournaments_b'])}")

    retention_pct = results["returning"] / results["total_a"] * 100
    new_pct = results["new"] / results["total_b"] * 100
    turnover_pct = results["departed"] / results["total_a"] * 100

    print("\nPlayer Statistics:")
    print(f"  {year_a}: {results['total_a']} unique players")
    print(f"  {year_b}: {results['total_b']} unique players")
    print(f"  Returning from {year_a}: {results['returning']} ({retention_pct:.1f}%)")
    print(f"  New in {year_b}: {results['new']} ({new_pct:.1f}%)")
    print(f"  Departed after {year_a}: {results['departed']} ({turnover_pct:.1f}%)")

    total_teams = (
        len(results["fully_returning_teams"])
        + len(results["partially_returning_teams"])
        + len(results["fully_new_teams"])
    )
    print(f"\n{year_b} Team Composition Analysis:")
    print(f"  Total teams in {year_b}: {total_teams}")

    for label, teams in [
        ("Fully returning", results["fully_returning_teams"]),
        ("Partially returning", results["partially_returning_teams"]),
        ("Fully new", results["fully_new_teams"]),
    ]:
        pct = len(teams) / total_teams * 100
        print(f"  {label} teams: {len(teams)} ({pct:.1f}%)")

    if results["fully_returning_teams"]:
        print(f"\nFully Returning Teams (all players from {year_a}):")
        for info in results["fully_returning_teams"][:10]:
            print(f"  {info['team']} ({info['size']} players)")
        remaining = len(results["fully_returning_teams"]) - 10
        if remaining > 0:
            print(f"  ... and {remaining} more")

    name_lookup = (
        players_df.drop_duplicates("player_slug")
        .set_index("player_slug")["player_name"]
        .to_dict()
    )
    print("\nSample returning players (up to 10):")
    for slug in results["returning_players"][:10]:
        print(f"  {name_lookup.get(slug, slug)}")
    remaining = len(results["returning_players"]) - 10
    if remaining > 0:
        print(f"  ... and {remaining} more")


def build_player_graph(players_df, team_to_players, verbose=False):
    """Construct a graph where nodes are players and edges connect teammates.

    Args:
        players_df: DataFrame with player records
        team_to_players: Mapping of team labels to player sets
        verbose: If True, print graph construction statistics
    """
    graph = nx.Graph()
    name_lookup = (
        players_df.drop_duplicates("player_slug")
        .set_index("player_slug")["player_name"]
        .to_dict()
    )
    appearance_counts = players_df.groupby("player_slug")["team"].nunique().to_dict()
    if verbose:
        print(f"Building player graph with {len(appearance_counts)} nodes...")
    for slug, count in appearance_counts.items():
        graph.add_node(
            slug,
            label=name_lookup.get(slug, slug),
            appearances=count,
        )

    for roster in team_to_players.values():
        for a, b in combinations(sorted(roster), 2):
            if graph.has_edge(a, b):
                graph[a][b]["weight"] += 1
            else:
                graph.add_edge(a, b, weight=1)

    if verbose:
        print(f"Graph constructed with {graph.number_of_edges()} edges")

    return graph


def _save_figure(fig, filename, verbose=True):
    """Save figure to the figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    if verbose:
        print(f"Saved figure: {filepath}")


def analyze_team_overlap(team_to_players, verbose=True, show_heatmap=True):
    """Analyze and visualize team overlap patterns.

    Args:
        team_to_players: Mapping of team labels to player sets
        verbose: If True, print statistics and insights
        show_heatmap: If True and dataset is small enough, show heatmap
    """
    team_pairs = []
    for (team_a, players_a), (team_b, players_b) in combinations(
        team_to_players.items(), 2
    ):
        overlap = len(players_a & players_b)
        if overlap:
            team_pairs.append(
                {"team_a": team_a, "team_b": team_b, "shared_players": overlap}
            )

    if not team_pairs:
        print("No overlapping players across teams in this dataset.")
        return

    mix_df = pd.DataFrame(team_pairs)
    total_teams = len(team_to_players)
    teams_with_overlap = len(set(mix_df.team_a) | set(mix_df.team_b))
    avg_overlap = mix_df.shared_players.mean()
    max_overlap = mix_df.shared_players.max()

    team_overlap_counts = defaultdict(int)
    team_total_shared = defaultdict(int)
    for _, row in mix_df.iterrows():
        team_overlap_counts[row.team_a] += 1
        team_overlap_counts[row.team_b] += 1
        team_total_shared[row.team_a] += row.shared_players
        team_total_shared[row.team_b] += row.shared_players

    if verbose:
        _print_overlap_stats(
            total_teams,
            teams_with_overlap,
            len(mix_df),
            avg_overlap,
            max_overlap,
            team_overlap_counts,
            team_total_shared,
        )

    _plot_overlap_visualizations(
        mix_df,
        team_to_players,
        team_overlap_counts,
        avg_overlap,
        max_overlap,
        verbose,
    )

    if show_heatmap and total_teams <= 30:
        _plot_overlap_heatmap(mix_df, team_to_players, verbose)
    elif verbose and total_teams > 30:
        print(f"\nSkipping heatmap (too many teams: {total_teams})")


def _print_overlap_stats(
    total_teams,
    teams_with_overlap,
    total_overlaps,
    avg_overlap,
    max_overlap,
    team_overlap_counts,
    team_total_shared,
):
    """Print team overlap statistics."""
    print("=== Team Overlap Analysis ===")
    print(f"Total teams: {total_teams}")
    pct = teams_with_overlap / total_teams * 100
    print(f"Teams with overlap: {teams_with_overlap} ({pct:.1f}%)")
    print(f"Isolated teams: {total_teams - teams_with_overlap}")
    print(f"Total unique overlaps: {total_overlaps}")
    print(f"Average overlap size: {avg_overlap:.1f} players")
    print(f"Maximum overlap: {max_overlap} players")

    top_connected = sorted(
        team_overlap_counts.items(), key=lambda x: x[1], reverse=True
    )[:5]
    print("\nMost connected teams (by # of overlaps):")
    for team, count in top_connected:
        print(f"  {team}: {count} teams")

    top_sharing = sorted(team_total_shared.items(), key=lambda x: x[1], reverse=True)[
        :5
    ]
    print("\nTeams sharing most players (cumulative):")
    for team, total in top_sharing:
        print(f"  {team}: {total} player-overlaps")


def _plot_overlap_visualizations(
    mix_df, team_to_players, team_overlap_counts, avg_overlap, max_overlap, verbose
):
    """Create and save team overlap visualizations."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Overlap size distribution
    axes[0, 0].hist(
        mix_df.shared_players,
        bins=range(1, max_overlap + 2),
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )
    axes[0, 0].set_xlabel("Number of Shared Players")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Team Overlap Sizes")
    axes[0, 0].grid(axis="y", alpha=0.3)

    # Panel 2: Top connected teams
    connectivity = pd.Series(team_overlap_counts).sort_values(ascending=False)
    top_n = min(20, len(connectivity))
    axes[0, 1].barh(range(top_n), connectivity.head(top_n).values, color="coral")
    axes[0, 1].set_yticks(range(top_n))
    axes[0, 1].set_yticklabels(connectivity.head(top_n).index, fontsize=8)
    axes[0, 1].set_xlabel("Number of Teams with Overlap")
    axes[0, 1].set_title(f"Top {top_n} Most Connected Teams")
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis="x", alpha=0.3)

    # Panel 3: Largest overlaps
    top_overlaps = mix_df.nlargest(min(15, len(mix_df)), "shared_players")
    overlap_labels = [
        f"{row.team_a[:20]}\nvs\n{row.team_b[:20]}"
        for _, row in top_overlaps.iterrows()
    ]
    axes[1, 0].barh(
        range(len(top_overlaps)),
        top_overlaps.shared_players.values,
        color="mediumseagreen",
    )
    axes[1, 0].set_yticks(range(len(top_overlaps)))
    axes[1, 0].set_yticklabels(overlap_labels, fontsize=7)
    axes[1, 0].set_xlabel("Shared Players")
    axes[1, 0].set_title("Largest Team Overlaps")
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(axis="x", alpha=0.3)

    # Panel 4: Network graph
    _draw_team_network(axes[1, 1], mix_df, team_to_players, avg_overlap)

    plt.tight_layout()
    _save_figure(fig, "team_overlap_analysis.png", verbose)
    plt.show()


def _draw_team_network(ax, mix_df, team_to_players, avg_overlap):
    """Draw team overlap network on the given axes."""
    team_graph = nx.Graph()
    for team in team_to_players.keys():
        team_graph.add_node(team, size=len(team_to_players[team]))

    threshold = max(2, int(avg_overlap))
    for _, row in mix_df[mix_df.shared_players >= threshold].iterrows():
        team_graph.add_edge(row.team_a, row.team_b, weight=row.shared_players)

    if team_graph.number_of_edges() > 0:
        pos = nx.spring_layout(team_graph, k=1.5, iterations=50, seed=42)
        node_sizes = [team_graph.nodes[n]["size"] * 30 for n in team_graph.nodes()]
        edge_widths = [team_graph[u][v]["weight"] * 0.5 for u, v in team_graph.edges()]

        nx.draw_networkx_nodes(
            team_graph,
            pos,
            ax=ax,
            node_size=node_sizes,
            node_color="skyblue",
            alpha=0.7,
        )
        nx.draw_networkx_edges(
            team_graph, pos, ax=ax, width=edge_widths, alpha=0.5, edge_color="gray"
        )

        if len(team_graph.nodes()) <= 30:
            labels = {n: n[:15] for n in team_graph.nodes()}
            nx.draw_networkx_labels(
                team_graph, pos, labels, ax=ax, font_size=7, font_weight="bold"
            )

        ax.set_title(f"Team Overlap Network (≥{threshold} shared players)")
    else:
        ax.text(
            0.5,
            0.5,
            "No significant overlaps\nfor network visualization",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_title("Team Overlap Network")

    ax.axis("off")


def _plot_overlap_heatmap(mix_df, team_to_players, verbose):
    """Create and save detailed overlap heatmap."""
    base_matrix = mix_df.pivot(
        index="team_a", columns="team_b", values="shared_players"
    ).fillna(0)
    team_labels = sorted(team_to_players.keys())
    matrix = (base_matrix + base_matrix.T).reindex(
        index=team_labels, columns=team_labels, fill_value=0
    )

    size = max(8, 0.4 * len(team_labels))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar_kws={"label": "Shared Players"},
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Detailed Team Overlap Heatmap")
    ax.set_xlabel("Team")
    ax.set_ylabel("Team")
    plt.tight_layout()
    _save_figure(fig, "team_overlap_heatmap.png", verbose)
    plt.show()


def plot_largest_player_component(graph, verbose=True):
    """Highlight the biggest cluster of interconnected players."""
    if graph.number_of_edges() == 0:
        print("Player graph has no shared-team edges to visualize.")
        return

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    component_sizes = [len(c) for c in components]
    print(
        "Largest player clusters:",
        ", ".join(str(size) for size in component_sizes[:5]),
    )

    largest_component = graph.subgraph(components[0]).copy()
    degrees = dict(largest_component.degree())
    node_sizes = [
        220 + 140 * largest_component.nodes[n]["appearances"] for n in largest_component
    ]
    edge_widths = [
        0.8 + data["weight"] for _, _, data in largest_component.edges(data=True)
    ]
    highlighted = sorted(
        largest_component.nodes(), key=lambda n: degrees[n], reverse=True
    )[:15]
    labels = {node: largest_component.nodes[node]["label"] for node in highlighted}

    pos = nx.spring_layout(largest_component, k=0.45, seed=7)
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(
        largest_component,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color="royalblue",
        alpha=0.8,
    )
    nx.draw_networkx_edges(
        largest_component,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color="lightgray",
        alpha=0.9,
    )
    nx.draw_networkx_labels(
        largest_component,
        pos,
        labels=labels,
        ax=ax,
        font_size=9,
        font_weight="bold",
    )
    ax.set_title("Largest Player Connectivity Cluster")
    ax.axis("off")
    plt.tight_layout()
    _save_figure(fig, "player_connectivity_cluster.png", verbose)
    plt.show()


# %%
def main(verbose=True):
    """Main workflow for analyzing player collaboration across tournaments.

    Args:
        verbose: If True, print detailed progress and statistics
    """
    tournament_lookup = create_tournament_lookup(DB_FILES)
    if verbose:
        print("=== Tournament Metadata ===")
        print(f"Total tournaments: {len(tournament_lookup)}")
        tournaments_by_qset = defaultdict(list)
        for slug, info in tournament_lookup.items():
            qset = info["question_set"]
            tournaments_by_qset[qset].append(slug)
        print("Tournaments by question set:")
        for qset, slugs in tournaments_by_qset.items():
            print(f"* {qset}:")
            for slug in slugs:
                info = tournament_lookup[slug]
                year = info["year"] or "N/A"
                date_str = (
                    info["start_date"].strftime("%Y-%m-%d")
                    if info["start_date"]
                    else "N/A"
                )
                print(f"   {slug}: {info['name']} ({year}, {date_str})")
        print()

    players_df = load_player_records(DB_FILES, verbose=verbose)
    if verbose:
        print(f"\nTotal player entries: {len(players_df)}")
        unique_players = players_df[["player_slug", "player_name"]].drop_duplicates()
        print(f"Unique players tracked: {len(unique_players)}")
        print()

    inspect_duplicate_player_slugs(players_df)
    report_slug_inconsistencies(players_df, verbose=verbose)

    return

    analyze_temporal_player_overlap(
        players_df, year_a=2024, year_b=2025, verbose=verbose
    )

    dedup_records, team_to_players, dedup_df = deduplicate_teams(
        players_df, verbose=verbose
    )
    if verbose:
        print()
    summarize_deduped_teams(dedup_df, dedup_records, verbose=verbose)
    if verbose:
        print()

    analyze_team_overlap(team_to_players, verbose=verbose)
    if verbose:
        print()

    player_graph = build_player_graph(players_df, team_to_players, verbose=verbose)
    if verbose:
        print()
    plot_largest_player_component(player_graph, verbose=verbose)

    return {
        "players_df": players_df,
        "tournament_lookup": tournament_lookup,
        "team_to_players": team_to_players,
        "player_graph": player_graph,
    }


if __name__ == "__main__":
    main()
# %%
