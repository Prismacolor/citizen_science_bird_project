"""
Script 3 — Visualizations
--------------------------
Reads analysis CSVs from Script 2 and produces publication-quality
charts saved as high-resolution PNGs ready for PowerPoint.

Species-agnostic: reads species names from the data, auto-assigns
colors, and adjusts subplot layouts dynamically.

Outputs saved to ./figures/:
    01_habitat_by_species.png       — % sightings per habitat, side by side
    02_habitat_by_season.png        — habitat use shifts across seasons
    03_seasonal_detection_rate.png  — monthly detection rates (effort-corrected)
    04_hotspot_density_map.png      — lat/lon hexbin of detections
    05_coverage_gap_map.png         — observed / water / no data grid
    06_underbirded_counties.png     — bar chart of lowest-coverage counties
    07_landscape_change_pie.png     — stable vs changed landscape split
    08_change_by_species.png        — change type breakdown per species
    09_change_by_habitat.png        — which habitats are shifting

Requirements:
    pip install pandas matplotlib seaborn numpy
"""

import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

ANALYSIS_FOLDER = "./analysis_results"
FIGURES_FOLDER  = "./figures"
ENRICHED_PATH   = "./data/vulture_sightings_with_habitat.csv"

os.makedirs(FIGURES_FOLDER, exist_ok=True)

# ── STYLE ─────────────────────────────────────────────────────────────────────
# Species colors are assigned dynamically from this pool.
# First species gets index 0, second gets index 1, etc.

SPECIES_COLOR_POOL = [
    "#2C2C2C",  # dark charcoal
    "#C0392B",  # crimson
    "#2980B9",  # steel blue
    "#27AE60",  # emerald
    "#8E44AD",  # purple
    "#D35400",  # burnt orange
    "#16A085",  # teal
    "#F39C12",  # amber
]


def _get_species_colors(species_list: list[str]) -> dict[str, str]:
    """Assign a color to each species from the pool."""
    return {
        species: SPECIES_COLOR_POOL[i % len(SPECIES_COLOR_POOL)]
        for i, species in enumerate(sorted(species_list))
    }


SEASON_COLORS = {
    "Spring": "#27AE60", "Summer": "#F39C12",
    "Fall":   "#E74C3C", "Winter": "#2980B9",
}

CHANGE_COLORS = {
    "Stable":             "#2D6A4F",
    "Urban Gain":         "#E74C3C",
    "Cropland Change":    "#F39C12",
    "Wetland/Water Flux": "#3498DB",
    "Rare Change":        "#9B59B6",
    "Common Change":      "#E67E22",
}

BG_COLOR   = "#FAFAFA"
GRID_COLOR = "#E8E8E8"
FONT_TITLE = {"fontsize": 16, "fontweight": "bold", "color": "#1A1A2E"}
FONT_LABEL = {"fontsize": 11, "color": "#333333"}
FONT_TICK  = {"labelsize": 9}
DPI        = 180

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    BG_COLOR,
    "figure.facecolor":  "white",
    "axes.grid":         True,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def save(fig, name: str):
    path = os.path.join(FIGURES_FOLDER, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# Habitat Percentage by Species
def fig_habitat_by_species():
    df = pd.read_csv(os.path.join(ANALYSIS_FOLDER, "habitat_pct.csv"))
    species_list = sorted(df["COMMON NAME"].unique())
    n = len(species_list)
    palette = _get_species_colors(species_list)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    title = " vs ".join(species_list) if n <= 3 else "Habitat Use by Species"
    fig.suptitle(f"Habitat Use: {title}", **FONT_TITLE, y=1.02)

    for ax, species in zip(axes, species_list):
        grp = df[df["COMMON NAME"] == species].sort_values("PCT", ascending=True)
        bars = ax.barh(grp["HABITAT_GROUP"], grp["PCT"], color="#5DADE2", edgecolor="white", linewidth=0.5)
        for bar, pct in zip(bars, grp["PCT"]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", **FONT_LABEL)
        ax.set_title(species, fontsize=13, fontweight="bold",
                     color=palette.get(species, "#333333"))
        ax.set_xlabel("% of Sightings", **FONT_LABEL)
        ax.tick_params(**FONT_TICK)
        ax.set_xlim(0, grp["PCT"].max() + 12)

    fig.tight_layout()
    save(fig, "habitat_by_species.png")


# Habitat by Season Heatmap
def fig_habitat_by_season():
    df = pd.read_csv(os.path.join(ANALYSIS_FOLDER, "habitat_by_season.csv"))
    season_order = ["Spring", "Summer", "Fall", "Winter"]
    species_list = sorted(df["COMMON NAME"].unique())
    n = len(species_list)
    palette = _get_species_colors(species_list)

    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle("Habitat Use by Season", **FONT_TITLE, y=1.02)

    for ax, species in zip(axes, species_list):
        pivot = (
            df[df["COMMON NAME"] == species]
            .pivot_table(index="HABITAT_GROUP", columns="SEASON", values="PCT", fill_value=0)
            .reindex(columns=season_order)
        )
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="YlOrBr",
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "% of Sightings", "shrink": 0.7},
                    annot_kws={"size": 9})
        ax.set_title(species, fontsize=13, fontweight="bold",
                     color=palette.get(species, "#333333"))
        ax.set_xlabel("")
        ax.set_ylabel("Habitat Type", **FONT_LABEL)
        ax.tick_params(**FONT_TICK)

    fig.tight_layout()
    save(fig, "habitat_by_season.png")


# Seasonal Detection Rate

def fig_seasonal_detection_rate():
    df = pd.read_csv(os.path.join(ANALYSIS_FOLDER, "seasonal_counts.csv"))
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    df["MONTH_NAME"] = pd.Categorical(df["MONTH_NAME"], categories=month_order, ordered=True)
    df = df.sort_values("MONTH_NAME")

    species_list = sorted(df["COMMON NAME"].unique())
    palette = _get_species_colors(species_list)

    fig, ax = plt.subplots(figsize=(13, 6))
    for species, grp in df.groupby("COMMON NAME"):
        color = palette.get(str(species), "#555555")
        ax.plot(grp["MONTH_NAME"], grp["DETECTION_RATE"],
                marker="o", linewidth=2.5, markersize=7, color=color, label=species)
        ax.fill_between(grp["MONTH_NAME"], grp["DETECTION_RATE"], alpha=0.1, color=color)

    season_spans = [
        ("Winter", 0, 2, SEASON_COLORS["Winter"]),
        ("Spring", 2, 5, SEASON_COLORS["Spring"]),
        ("Summer", 5, 8, SEASON_COLORS["Summer"]),
        ("Fall",   8, 11, SEASON_COLORS["Fall"]),
        ("Winter", 11, 12, SEASON_COLORS["Winter"]),
    ]
    for _, x0, x1, color in season_spans:
        ax.axvspan(x0 - 0.5, x1 - 0.5, alpha=0.07, color=color, zorder=0)

    ax.set_title("Monthly Detection Rate (Effort-Corrected)\nDetections per 100 Complete Checklists", **FONT_TITLE)
    ax.set_xlabel("Month", **FONT_LABEL)
    ax.set_ylabel("Detection Rate (%)", **FONT_LABEL)
    ax.tick_params(**FONT_TICK)
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    save(fig, "seasonal_detection_rate.png")


# Hotspot Density Map
def fig_hotspot_density_map():
    if not os.path.exists(ENRICHED_PATH):
        print("  Enriched CSV not found — skipping hotspot map.")
        return

    raw = pd.read_csv(ENRICHED_PATH, usecols=["COMMON NAME", "LATITUDE", "LONGITUDE"])
    species_list = sorted(raw["COMMON NAME"].unique())
    n = len(species_list)
    palette = _get_species_colors(species_list)

    # Auto-detect map bounds from the data with a small pad
    lat_pad = (raw["LATITUDE"].max() - raw["LATITUDE"].min()) * 0.05
    lon_pad = (raw["LONGITUDE"].max() - raw["LONGITUDE"].min()) * 0.05
    xlim = (raw["LONGITUDE"].min() - lon_pad, raw["LONGITUDE"].max() + lon_pad)
    ylim = (raw["LATITUDE"].min() - lat_pad, raw["LATITUDE"].max() + lat_pad)

    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
    if n == 1:
        axes = [axes]
    fig.suptitle("Detection Hotspots", **FONT_TITLE, y=1.01)

    for ax, species in zip(axes, species_list):
        sub = raw[raw["COMMON NAME"] == species]
        hb = ax.hexbin(sub["LONGITUDE"], sub["LATITUDE"], gridsize=60, cmap="YlOrRd",
                       mincnt=1, linewidths=0.2)
        fig.colorbar(hb, ax=ax, shrink=0.7).set_label("Detections per cell", fontsize=9)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(species, fontsize=13, fontweight="bold",
                     color=palette.get(species, "#333333"))
        ax.set_xlabel("Longitude", **FONT_LABEL)
        ax.set_ylabel("Latitude", **FONT_LABEL)
        ax.tick_params(**FONT_TICK)

    fig.tight_layout()
    save(fig, "hotspot_density_map.png")


# Coverage Gap Map
def fig_coverage_gap_map():
    df = pd.read_csv(os.path.join(ANALYSIS_FOLDER, "grid_coverage.csv"))
    status_colors = {"Observed": "#27AE60", "No Data": "#E8E8E8", "Open Water": "#5DADE2"}

    # Auto-detect bounds from the grid
    xlim = (df["GRID_LON"].min() - 0.5, df["GRID_LON"].max() + 0.5)
    ylim = (df["GRID_LAT"].min() - 0.5, df["GRID_LAT"].max() + 0.5)

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.suptitle("Where Do We Have Data? Identifying Coverage Gaps", **FONT_TITLE)

    for status, grp in df.groupby("STATUS"):
        color = status_colors.get(str(status), "#CCCCCC")
        ax.scatter(grp["GRID_LON"], grp["GRID_LAT"], c=color, s=4, alpha=0.7, linewidths=0, label=status)

    patches = [mpatches.Patch(color=v, label=k) for k, v in status_colors.items()]
    ax.legend(handles=patches, loc="lower left", fontsize=10, framealpha=0.95,
              title="Cell Status", title_fontsize=10)
    ax.set_xlabel("Longitude", **FONT_LABEL)
    ax.set_ylabel("Latitude", **FONT_LABEL)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(**FONT_TICK)
    ax.text(xlim[0] + 0.2, ylim[0] + 0.3,
            "Grey = No Data ≠ Absence\nBlue = Open Water (expected absence)",
            fontsize=8.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CCCCCC", alpha=0.9))
    fig.tight_layout()
    save(fig, "coverage_gap_map.png")


# Underbirded Counties
def fig_underbirded_counties():
    df = pd.read_csv(os.path.join(ANALYSIS_FOLDER, "underbirded_counties.csv"))
    df = df.sort_values("TOTAL_CHECKLISTS").head(25)
    df["LABEL"] = df["COUNTY"] + ", " + df["STATE"]

    # Auto-assign colors per state
    states = sorted(df["STATE"].unique())
    state_pool = ["#C0392B", "#2980B9", "#27AE60", "#8E44AD", "#D35400", "#16A085", "#F39C12", "#2C3E50"]
    state_colors = {s: state_pool[i % len(state_pool)] for i, s in enumerate(states)}
    colors = [state_colors.get(s, "#999999") for s in df["STATE"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(df["LABEL"], df["TOTAL_CHECKLISTS"], color=colors, edgecolor="white")
    for bar, val in zip(bars, df["TOTAL_CHECKLISTS"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va="center", fontsize=8)

    ax.set_title("Most Underbirded Counties\n(Fewest Complete Checklists Submitted)", **FONT_TITLE)
    ax.set_xlabel("Total Complete Checklists", **FONT_LABEL)
    ax.tick_params(**FONT_TICK)
    patches = [mpatches.Patch(color=v, label=k) for k, v in state_colors.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save(fig, "underbirded_counties.png")


# Landscape Change Pie
def fig_landscape_change_pie():
    df = pd.read_csv(os.path.join(ANALYSIS_FOLDER, "landscape_change.csv"))
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#2D6A4F", "#E74C3C"]
    wedges, texts, autotexts = ax.pie(
        df["PCT"], labels=df["STATUS"], colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 13})
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("Sightings: Stable vs Shifting Landscapes", **FONT_TITLE)
    fig.tight_layout()
    save(fig, "landscape_change_pie.png")


# Change Type by Species
def fig_change_by_species():
    df = pd.read_csv(os.path.join(ANALYSIS_FOLDER, "change_by_species.csv"))
    changed = df[df["CHANGE_TYPE"] != "Stable"]
    if changed.empty:
        print("  No change data — skipping change_by_species chart.")
        return

    species_list = sorted(changed["COMMON NAME"].unique())
    n = len(species_list)
    palette = _get_species_colors(species_list)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("What Kind of Landscape Change?", **FONT_TITLE, y=1.02)

    for ax, species in zip(axes, species_list):
        grp = changed[changed["COMMON NAME"] == species].sort_values("COUNT", ascending=True)
        colors = [CHANGE_COLORS.get(ct, "#999999") for ct in grp["CHANGE_TYPE"]]
        bars = ax.barh(grp["CHANGE_TYPE"], grp["COUNT"], color=colors, edgecolor="white")
        for bar, count in zip(bars, grp["COUNT"]):
            ax.text(bar.get_width() + max(grp["COUNT"]) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{count:,}", va="center", fontsize=10)
        ax.set_title(species, fontsize=13, fontweight="bold",
                     color=palette.get(species, "#333333"))
        ax.set_xlabel("Number of Sightings", **FONT_LABEL)
        ax.tick_params(**FONT_TICK)

    fig.tight_layout()
    save(fig, "change_by_species.png")


# Change × Habitat Heatmap
def fig_change_by_habitat():
    path = os.path.join(ANALYSIS_FOLDER, "change_by_habitat.csv")
    if not os.path.exists(path):
        print("  No change_by_habitat data — skipping.")
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    pivot = df.pivot_table(index="HABITAT_GROUP", columns="CHANGE_TYPE", values="COUNT", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, max(5, len(pivot) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="OrRd",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Sightings", "shrink": 0.7},
                annot_kws={"size": 10}, ax=ax)
    ax.set_title("Shifting Landscapes × Habitat Type\nWhere is change happening?", **FONT_TITLE)
    ax.set_xlabel("Change Type", **FONT_LABEL)
    ax.set_ylabel("Habitat", **FONT_LABEL)
    ax.tick_params(**FONT_TICK)
    fig.tight_layout()
    save(fig, "change_by_habitat.png")


def main():
    print("Generating figures...\n")
    fig_habitat_by_species()
    fig_habitat_by_season()
    fig_seasonal_detection_rate()
    fig_hotspot_density_map()
    fig_coverage_gap_map()
    fig_underbirded_counties()
    fig_landscape_change_pie()
    fig_change_by_species()
    fig_change_by_habitat()
    print(f"\nAll figures saved to ./{FIGURES_FOLDER}/")


if __name__ == "__main__":
    main()