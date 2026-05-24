"""
Reads analysis CSVs and produces publication-quality
charts and maps saved as high-resolution PNGs.

Outputs saved to ./figures/:
    1. habitat_by_species.png       — percentage of sightings per habitat, side by side
    2. habitat_by_season.png        — habitat use shifts across seasons
    3. seasonal_detection_rate.png  — monthly detection rates (effort-corrected)
    4. hotspot_density_map.png      — lat/lon heatmap of vulture detections
    5. coverage_gap_map.png         — no data vs water vs observed grid map
    6. underbirded_counties.png     — bar chart of lowest-coverage counties
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


ANALYSIS_FOLDER = "./analysis_results"
FIGURES_FOLDER  = "./figures"
os.makedirs(FIGURES_FOLDER, exist_ok=True)

# ── STYLE ─────────────────────────────────────────────────────────────────────
# Clean, nature-inspired palette that works well in presentations

PALETTE = {
    "Black Vulture":          "#2C2C2C",
    "Turkey Vulture":         "#C0392B",
    "Forest":                 "#2D6A4F",
    "Grassland/Agriculture":  "#B7950B",
    "Developed":              "#7D6608",
    "Shrub/Scrub":            "#A9CCE3",
    "Wetlands":               "#1A5276",
    "Barren":                 "#BDC3C7",
    "Other":                  "#717D7E",
    "Water":                  "#5DADE2",
}

SEASON_COLORS = {
    "Spring": "#27AE60",
    "Summer": "#F39C12",
    "Fall":   "#E74C3C",
    "Winter": "#2980B9",
}

BG_COLOR    = "#FAFAFA"
GRID_COLOR  = "#E8E8E8"
FONT_TITLE  = {"fontsize": 16, "fontweight": "bold", "color": "#1A1A2E"}
FONT_LABEL  = {"fontsize": 11, "color": "#333333"}
FONT_TICK   = {"labelsize": 9}
DPI         = 180   # high enough for PowerPoint slides

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


# Habitat percentages by species
def fig_habitat_by_species():
    df = pd.read_csv(os.path.join(ANALYSIS_FOLDER, "habitat_pct.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        "Habitat Use: Black Vulture vs Turkey Vulture",
        **FONT_TITLE, y=1.02
    )

    for ax, (species, grp) in zip(axes, df.groupby("COMMON NAME")):
        grp = grp.sort_values("PCT", ascending=True)
        colors = [PALETTE.get(h, "#999999") for h in grp["HABITAT_GROUP"]]
        bars = ax.barh(grp["HABITAT_GROUP"], grp["PCT"], color=colors, edgecolor="white", linewidth=0.5)

        # Value labels on bars
        for bar, pct in zip(bars, grp["PCT"]):
            ax.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", **FONT_LABEL
            )

        ax.set_title(species, fontsize=13, fontweight="bold",
                     color=PALETTE.get(str(species), "#333333"))
        ax.set_xlabel("Percentage of Sightings", **FONT_LABEL)
        ax.tick_params(**FONT_TICK)
        ax.set_xlim(0, grp["PCT"].max() + 12)

    fig.tight_layout()
    save(fig, "habitat_by_species.png")


# Habitat by Season (Heatmap)
def fig_habitat_by_season():
    df = pd.read_csv(os.path.join(ANALYSIS_FOLDER, "habitat_by_season.csv"))

    season_order  = ["Spring", "Summer", "Fall", "Winter"]
    species_list  = df["COMMON NAME"].unique()

    fig, axes = plt.subplots(1, len(species_list), figsize=(16, 6))
    if len(species_list) == 1:
        axes = [axes]

    fig.suptitle(
        "Habitat Use by Season",
        **FONT_TITLE, y=1.02
    )

    for ax, species in zip(axes, species_list):
        pivot = (
            df[df["COMMON NAME"] == species]
            .pivot_table(index="HABITAT_GROUP", columns="SEASON", values="PCT", fill_value=0)
            .reindex(columns=season_order)
        )

        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".1f", cmap="YlOrBr",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "% of Sightings", "shrink": 0.7},
            annot_kws={"size": 9},
        )
        ax.set_title(species, fontsize=13, fontweight="bold",
                     color=PALETTE.get(species, "#333333"))
        ax.set_xlabel("")
        ax.set_ylabel("Habitat Type", **FONT_LABEL)
        ax.tick_params(**FONT_TICK)

    fig.tight_layout()
    save(fig, "habitat_by_season.png")


# Seasonal Detection Rate
def fig_seasonal_detection_rate():
    df = pd.read_csv(os.path.join(ANALYSIS_FOLDER, "seasonal_counts.csv"))

    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    df["MONTH_NAME"] = pd.Categorical(df["MONTH_NAME"], categories=month_order, ordered=True)
    df = df.sort_values("MONTH_NAME")

    fig, ax = plt.subplots(figsize=(13, 6))

    for species, grp in df.groupby("COMMON NAME"):
        color = PALETTE.get(str(species), "#555555")  # convert to string, because pandas likely reads in as categorical, not string
        ax.plot(
            grp["MONTH_NAME"], grp["DETECTION_RATE"],
            marker="o", linewidth=2.5, markersize=7,
            color=color, label=species
        )
        ax.fill_between(
            grp["MONTH_NAME"], grp["DETECTION_RATE"],
            alpha=0.1, color=color
        )

    # Season shading bands
    season_spans = [
        ("Winter",  0,   2,  SEASON_COLORS["Winter"]),
        ("Spring",  2,   5,  SEASON_COLORS["Spring"]),
        ("Summer",  5,   8,  SEASON_COLORS["Summer"]),
        ("Fall",    8,   11, SEASON_COLORS["Fall"]),
        ("Winter", 11,   12, SEASON_COLORS["Winter"]),
    ]
    for label, x0, x1, color in season_spans:
        ax.axvspan(x0 - 0.5, x1 - 0.5, alpha=0.07, color=color, zorder=0)

    ax.set_title(
        "Monthly Detection Rate (Effort-Corrected)\nDetections per 100 Complete Checklists",
        **FONT_TITLE
    )
    ax.set_xlabel("Month", **FONT_LABEL)
    ax.set_ylabel("Detection Rate (%)", **FONT_LABEL)
    ax.tick_params(**FONT_TICK)
    ax.legend(fontsize=10, framealpha=0.9)

    fig.tight_layout()
    save(fig, "seasonal_detection_rate.png")


#  Hotspot Density Map
def fig_hotspot_density_map():
    """
    Density map of vulture detections overlaid on a state outline.
    Uses only the analysis grid data — no internet required.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Vulture Detection Hotspots — TX, AR, OK, LA",
        **FONT_TITLE, y=1.01
    )

    # We need the raw observation lat/lon — reload previous data for this
    data_path = "./data/vulture_sightings_with_habitat.csv"
    if not os.path.exists(data_path):
        print("  vulture_sightings_with_habitat.csv not found — skipping hotspot map.")
        plt.close(fig)
        return

    raw = pd.read_csv(data_path, usecols=["COMMON NAME", "LATITUDE", "LONGITUDE"])

    xlim = (-106.6, -88.8)
    ylim = (25.8,   36.5)

    for ax, species in zip(axes, ["Black Vulture", "Turkey Vulture"]):
        sub = raw[raw["COMMON NAME"] == species]
        color = PALETTE.get(species, "#333333")

        hb = ax.hexbin(
            sub["LONGITUDE"], sub["LATITUDE"],
            gridsize=60, cmap="YlOrRd",
            mincnt=1, linewidths=0.2,
        )
        cb = fig.colorbar(hb, ax=ax, shrink=0.7)
        cb.set_label("Detections per cell", fontsize=9)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(species, fontsize=13, fontweight="bold", color=color)
        ax.set_xlabel("Longitude", **FONT_LABEL)
        ax.set_ylabel("Latitude",  **FONT_LABEL)
        ax.tick_params(**FONT_TICK)

    fig.tight_layout()
    save(fig, "hotspot_density_map.png")


# Coverage Gap Map
def fig_coverage_gap_map():
    df = pd.read_csv(os.path.join(ANALYSIS_FOLDER, "grid_coverage.csv"))

    status_colors = {
        "Observed": "#27AE60",
        "No Data": "#E8E8E8",
        "Open Water": "#5DADE2",
    }

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.suptitle(
        "Where Do We Have Data? Identifying Coverage Gaps",
        **FONT_TITLE
    )

    for status, grp in df.groupby("STATUS"):
        color = status_colors.get(str(status), "#CCCCCC")
        ax.scatter(
            grp["GRID_LON"], grp["GRID_LAT"],
            c=color, s=4, alpha=0.7, linewidths=0, label=status
        )

    # Legend
    patches = [
        mpatches.Patch(color=v, label=k)
        for k, v in status_colors.items()
    ]
    ax.legend(
        handles=patches, loc="lower left",
        fontsize=10, framealpha=0.95, title="Cell Status", title_fontsize=10
    )

    ax.set_xlabel("Longitude", **FONT_LABEL)
    ax.set_ylabel("Latitude",  **FONT_LABEL)
    ax.set_xlim(-106.6, -88.8)
    ax.set_ylim(25.8,   36.5)
    ax.tick_params(**FONT_TICK)

    # Annotation box explaining the key distinction
    ax.text(
        -106.4, 26.3,
        "Grey = No Data (does not mean absence),\nBlue = Open Water (expected absence)",
        fontsize=8.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CCCCCC", alpha=0.9)
    )

    fig.tight_layout()
    save(fig, "coverage_gap_map.png")


# Underbirded Counties Bar Chart
def fig_underbirded_counties():
    df = pd.read_csv(os.path.join(ANALYSIS_FOLDER, "underbirded_counties.csv"))
    df = df.sort_values("TOTAL_CHECKLISTS").head(25)  # bottom 25
    df["LABEL"] = df["COUNTY"] + ", " + df["STATE"]

    state_colors = {
        "Texas":     "#C0392B",
        "Oklahoma":  "#2980B9",
        "Arkansas":  "#27AE60",
        "Louisiana": "#8E44AD",
    }

    colors = [state_colors.get(s, "#999999") for s in df["STATE"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(df["LABEL"], df["TOTAL_CHECKLISTS"], color=colors, edgecolor="white")

    for bar, val in zip(bars, df["TOTAL_CHECKLISTS"]):
        ax.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            str(int(val)), va="center", fontsize=8
        )

    ax.set_title(
        "Most Underbirded Counties\n(Fewest Complete Checklists Submitted)",
        **FONT_TITLE
    )
    ax.set_xlabel("Total Complete Checklists", **FONT_LABEL)
    ax.tick_params(**FONT_TICK)

    # State legend
    patches = [mpatches.Patch(color=v, label=k) for k, v in state_colors.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    save(fig, "underbirded_counties.png")


def main():
    print("Generating figures...\n")
    fig_habitat_by_species()
    fig_habitat_by_season()
    fig_seasonal_detection_rate()
    fig_hotspot_density_map()
    fig_coverage_gap_map()
    fig_underbirded_counties()
    print(f"\nAll figures saved to ./{FIGURES_FOLDER}/")


if __name__ == "__main__":
    main()