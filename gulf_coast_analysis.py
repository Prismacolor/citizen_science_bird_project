"""
Gulf Coast Vulture Analysis
================================
Focuses on vulture sightings in the Gulf of Mexico coastal zone
using NOAA C-CAP land cover data to classify what habitat these
"open water" birds are actually standing on.

Data sources:
  - eBird vulture sightings (same CSV from data folder)
  - NOAA C-CAP Regional 30m Land Cover (2016)
    Download from: https://chs.coast.noaa.gov/htdata/raster1/landcover/bulkdownload/30m_lc/
    File: conus_2016_ccap_landcover_20200311.tif (~402 MB)
    Place in: ./data/noaa_ccap_data/
"""

import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import pyproj


warnings.filterwarnings("ignore")

DATA_PATH = "./data/vulture_sightings_with_habitat.csv"
CCAP_FOLDER   = "./data/noaa_ccap_data"
CCAP_FILENAME = "conus_2016_ccap_landcover_20200311.tif"
CCAP_PATH     = os.path.join(CCAP_FOLDER, CCAP_FILENAME)

CCAP_URL = (
    "https://chs.coast.noaa.gov/htdata/raster1/landcover/bulkdownload/30m_lc/"
    + CCAP_FILENAME
)

OUTPUT_FOLDER  = "./analysis_results/gulf_coast"
FIGURES_FOLDER = "./figures/gulf_coast"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)

# Gulf of Mexico bounding box: South to South Padre, north to Galveston Bay, east to LA coast, west to TX coast
GULF_BBOX = {
    "lat_min": 25.8,
    "lat_max": 30.5,
    "lon_min": -97.8,
    "lon_max": -88.8,
}

# C-CAP LAND COVER CODES
CCAP_LABELS = {
    0:  "Background",
    1:  "Unclassified",
    2:  "Developed, High Intensity",
    3:  "Developed, Medium Intensity",
    4:  "Developed, Low Intensity",
    5:  "Developed, Open Space",
    6:  "Cultivated Crops",
    7:  "Pasture/Hay",
    8:  "Grassland/Herbaceous",
    9:  "Deciduous Forest",
    10: "Evergreen Forest",
    11: "Mixed Forest",
    12: "Scrub/Shrub",
    13: "Palustrine Forested Wetland",  # palustrine means areas with ocean derives salts that are non-tidal
    14: "Palustrine Scrub/Shrub Wetland",
    15: "Palustrine Emergent Wetland",
    16: "Estuarine Forested Wetland",
    17: "Estuarine Scrub/Shrub Wetland",
    18: "Estuarine Emergent Wetland",
    19: "Unconsolidated Shore",
    20: "Barren Land",
    21: "Open Water",
    22: "Palustrine Aquatic Bed",
    23: "Estuarine Aquatic Bed",
    24: "Tundra",
    25: "Perennial Ice/Snow",
}

# Group C-CAP classes into broader categories for visualization
CCAP_GROUPS = {
    "Background":                      "No Data",
    "Unclassified":                    "No Data",
    "Developed, High Intensity":       "Developed",
    "Developed, Medium Intensity":     "Developed",
    "Developed, Low Intensity":        "Developed",
    "Developed, Open Space":           "Developed",
    "Cultivated Crops":                "Agriculture",
    "Pasture/Hay":                     "Agriculture",
    "Grassland/Herbaceous":            "Grassland",
    "Deciduous Forest":                "Forest",
    "Evergreen Forest":                "Forest",
    "Mixed Forest":                    "Forest",
    "Scrub/Shrub":                     "Scrub/Shrub",
    "Palustrine Forested Wetland":     "Freshwater Wetland",
    "Palustrine Scrub/Shrub Wetland":  "Freshwater Wetland",
    "Palustrine Emergent Wetland":     "Freshwater Wetland",
    "Estuarine Forested Wetland":      "Salt Marsh/Estuarine",
    "Estuarine Scrub/Shrub Wetland":   "Salt Marsh/Estuarine",
    "Estuarine Emergent Wetland":      "Salt Marsh/Estuarine",
    "Unconsolidated Shore":            "Beach/Tidal Flat",
    "Barren Land":                     "Barren",
    "Open Water":                      "Open Water",
    "Palustrine Aquatic Bed":          "Aquatic Vegetation",
    "Estuarine Aquatic Bed":           "Aquatic Vegetation",
    "Tundra":                          "Other",
    "Perennial Ice/Snow":              "Other",
}

SPECIES_COLOR_POOL = [
    "#2C2C2C", "#C0392B", "#2980B9", "#27AE60",
    "#8E44AD", "#D35400", "#16A085", "#F39C12",
]

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


def _get_species_colors(species_list: list[str]) -> dict[str, str]:
    return {
        species: SPECIES_COLOR_POOL[i % len(SPECIES_COLOR_POOL)]
        for i, species in enumerate(sorted(species_list))
    }


def _save_csv(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(OUTPUT_FOLDER, name)
    df.to_csv(path, index=False)
    print(f"  Saved CSV: {path}")
    return path


def _save_fig(fig, name: str) -> str:
    path = os.path.join(FIGURES_FOLDER, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {path}")
    return path


def check_ccap_raster() -> bool:
    """Check if C-CAP raster exists. Print download instructions if not."""
    if os.path.exists(CCAP_PATH):
        print(f"C-CAP raster found: {CCAP_PATH}")
        return True

    print(f"\nC-CAP raster not found at: {CCAP_PATH}")
    print(f"Please download it manually at: {CCAP_URL} and save to: {CCAP_PATH}")
    return False


def load_gulf_sightings(path: str) -> pd.DataFrame:
    """
    Load the vulture CSV and filter to the Gulf bounding box.

    Params: path: str - path to data file

    Returns: dataframe with long and lat
    """
    print(f"Loading sightings from {path}...")
    df = pd.read_csv(path, parse_dates=["OBSERVATION DATE"], low_memory=False)

    gulf = df[
        (df["LATITUDE"] >= GULF_BBOX["lat_min"]) &
        (df["LATITUDE"] <= GULF_BBOX["lat_max"]) &
        (df["LONGITUDE"] >= GULF_BBOX["lon_min"]) &
        (df["LONGITUDE"] <= GULF_BBOX["lon_max"])
    ].copy()

    print(f"  Total sightings: {len(df):,}")
    return gulf


def ccap_raster_lookup(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Look up C-CAP pixel values for lat/lon arrays.

    Params: lats: np.ndarray - latitude values, lons: np.ndarray - longitude values

    Returns: np array with c-cap data
    """
    with rasterio.open(CCAP_PATH) as src:
        transformer = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows, cols = np.array(rows), np.array(cols)

        band = src.read(1)
        height, width = band.shape

        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        values = np.full(len(rows), fill_value=-1, dtype=np.int16)
        values[valid] = band[rows[valid], cols[valid]]

    return values


def add_ccap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add C-CAP habitat classification to Gulf sightings.

    Params: dataframe - vulture data

    Returns: dataframe - data with land cover codes
    """
    print("Running C-CAP raster lookup...")
    codes = ccap_raster_lookup(df["LATITUDE"].values, df["LONGITUDE"].values)

    df = df.copy()
    df["CCAP_CODE"] = codes
    df["CCAP_HABITAT"] = df["CCAP_CODE"].map(CCAP_LABELS).fillna("Unknown")
    df["CCAP_GROUP"] = df["CCAP_HABITAT"].map(CCAP_GROUPS).fillna("Other")

    print("C-CAP habitat breakdown:")
    print(df["CCAP_GROUP"].value_counts().to_string())
    return df


# Gulf Density Map
def gulf_density_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a density summary and hexbin map of Gulf coastal vulture
    sightings, colored by species. Also save a habitat breakdown CSV.

    Params: dataframe - vulture data

    Returns: dataframe - density data to be saved to csv
    """
    print("\n── Analysis 1: Gulf Density Map ──")

    # CSV: habitat breakdown for gulf sightings
    habitat_counts = (
        df.groupby(["COMMON NAME", "CCAP_GROUP"])
        .size().reset_index(name="COUNT")
    )
    habitat_counts["PCT"] = habitat_counts.groupby("COMMON NAME")["COUNT"].transform(
        lambda x: (x / x.sum() * 100).round(1)
    )
    habitat_counts = habitat_counts.sort_values(
        ["COMMON NAME", "COUNT"], ascending=[True, False]
    )
    _save_csv(habitat_counts, "gulf_habitat_breakdown.csv")

    # Figure 1: hexbin density map
    species_list = sorted(df["COMMON NAME"].unique())
    n = len(species_list)
    palette = _get_species_colors(species_list)
    bbox = GULF_BBOX

    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
    if n == 1:
        axes = [axes]
    fig.suptitle("Gulf Coastal Vulture Detections", **FONT_TITLE, y=1.01)

    for ax, species in zip(axes, species_list):
        sub = df[df["COMMON NAME"] == species]
        hb = ax.hexbin(
            sub["LONGITUDE"], sub["LATITUDE"],
            gridsize=50, cmap="YlOrRd", mincnt=1, linewidths=0.2,
        )
        fig.colorbar(hb, ax=ax, shrink=0.7).set_label("Detections per cell", fontsize=9)
        ax.set_xlim(bbox["lon_min"], bbox["lon_max"])
        ax.set_ylim(bbox["lat_min"], bbox["lat_max"])
        ax.set_title(species, fontsize=13, fontweight="bold",
                     color=palette.get(species, "#333333"))
        ax.set_xlabel("Longitude", **FONT_LABEL)
        ax.set_ylabel("Latitude", **FONT_LABEL)
        ax.tick_params(**FONT_TICK)

    fig.tight_layout()
    _save_fig(fig, "gulf_density_map.png")

    # Figure 2: Coastal habitat breakdown by species
    # Filter out Open Water and No Data to focus on the coastal types
    coastal = habitat_counts[~habitat_counts["CCAP_GROUP"].isin(["Open Water", "No Data", "Other"])]
    if not coastal.empty:
        species_list = sorted(coastal["COMMON NAME"].unique())
        n = len(species_list)
        palette = _get_species_colors(species_list)

        fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharey=True)
        if n == 1:
            axes = [axes]

        title = " vs ".join(species_list) if n <= 3 else "Coastal Habitat by Species"
        fig.suptitle(f"Gulf Coastal Habitat Use: {title}", **FONT_TITLE, y=1.02)

        for ax, species in zip(axes, species_list):
            grp = coastal[coastal["COMMON NAME"] == species].sort_values("PCT", ascending=True)
            bars = ax.barh(
                grp["CCAP_GROUP"], grp["PCT"],
                color="#5DADE2", edgecolor="white", linewidth=0.5,
            )
            for bar, pct in zip(bars, grp["PCT"]):
                ax.text(
                    bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", **FONT_LABEL,
                )
            ax.set_title(
                species, fontsize=13, fontweight="bold",
                color=palette.get(species, "#333333"),
            )
            ax.set_xlabel("% of Gulf Sightings", **FONT_LABEL)
            ax.tick_params(**FONT_TICK)
            ax.set_xlim(0, grp["PCT"].max() + 12)

        fig.tight_layout()
        _save_fig(fig, "gulf_habitat_by_species.png")

    return habitat_counts


# Seasonal Fluctuation
def seasonal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Effort-corrected monthly detection rates for the Gulf coastal zone.
    Detection rate = checklists with vultures / total complete checklists.

    Params: dataframe - vulture data

    Returns: dataframe - percentage rates for detections in coastal range
    """
    print("\n── Analysis 2: Seasonal Fluctuation ──")

    df = df.copy()
    df["MONTH"] = pd.to_datetime(df["OBSERVATION DATE"]).dt.month

    total_by_month = (
        df.groupby("MONTH")["SAMPLING EVENT IDENTIFIER"]
        .nunique().reset_index(name="TOTAL_CHECKLISTS")
    )
    detected_by_month = (
        df.groupby(["COMMON NAME", "MONTH"])["SAMPLING EVENT IDENTIFIER"]
        .nunique().reset_index(name="DETECTED_CHECKLISTS")
    )
    merged = detected_by_month.merge(total_by_month, on="MONTH")
    merged["DETECTION_RATE"] = (
        merged["DETECTED_CHECKLISTS"] / merged["TOTAL_CHECKLISTS"] * 100
    ).round(2)

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    merged["MONTH_NAME"] = merged["MONTH"].map(month_names)
    _save_csv(merged, "gulf_seasonal_rates.csv")

    # Figure: monthly detection rate line chart
    species_list = sorted(merged["COMMON NAME"].unique())
    palette = _get_species_colors(species_list)
    month_order = list(month_names.values())
    merged["MONTH_NAME"] = pd.Categorical(
        merged["MONTH_NAME"], categories=month_order, ordered=True
    )
    merged = merged.sort_values("MONTH_NAME")

    fig, ax = plt.subplots(figsize=(13, 6))
    for species, grp in merged.groupby("COMMON NAME"):
        color = palette.get(str(species), "#555555")
        ax.plot(
            grp["MONTH_NAME"], grp["DETECTION_RATE"],
            marker="o", linewidth=2.5, markersize=7, color=color, label=species,
        )
        ax.fill_between(grp["MONTH_NAME"], grp["DETECTION_RATE"], alpha=0.1, color=color)

    ax.set_title(
        "Gulf Coastal Detection Rate by Month (Effort-Corrected)"
        "\nDetections per 100 Complete Checklists",
        **FONT_TITLE,
    )
    ax.set_xlabel("Month", **FONT_LABEL)
    ax.set_ylabel("Detection Rate (%)", **FONT_LABEL)
    ax.tick_params(**FONT_TICK)
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    _save_fig(fig, "gulf_seasonal_rates.png")

    return merged


# ── ANALYSIS 3: Year-over-Year Trend ─────────────────────────────────────────

def yearly_trend_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Effort-corrected annual detection rates to track whether
    coastal vulture use is changing over time.
    """
    print("\n── Analysis 3: Year-over-Year Trend ──")

    df = df.copy()
    df["YEAR"] = pd.to_datetime(df["OBSERVATION DATE"]).dt.year

    total_by_year = (
        df.groupby("YEAR")["SAMPLING EVENT IDENTIFIER"]
        .nunique().reset_index(name="TOTAL_CHECKLISTS")
    )
    detected_by_year = (
        df.groupby(["COMMON NAME", "YEAR"])["SAMPLING EVENT IDENTIFIER"]
        .nunique().reset_index(name="DETECTED_CHECKLISTS")
    )
    merged = detected_by_year.merge(total_by_year, on="YEAR")
    merged["DETECTION_RATE"] = (
        merged["DETECTED_CHECKLISTS"] / merged["TOTAL_CHECKLISTS"] * 100
    ).round(2)

    _save_csv(merged, "gulf_yearly_trends.csv")

    # Figure: year-over-year line chart
    species_list = sorted(merged["COMMON NAME"].unique())
    palette = _get_species_colors(species_list)

    fig, ax1 = plt.subplots(figsize=(14, 7))

    for species, grp in merged.groupby("COMMON NAME"):
        grp = grp.sort_values("YEAR")
        color = palette.get(str(species), "#555555")
        ax1.plot(
            grp["YEAR"], grp["DETECTION_RATE"],
            marker="o", linewidth=2.5, markersize=5, color=color, label=species,
        )

    ax1.set_title(
        "Gulf Coastal Vulture Detection Rate Over Time (Effort-Corrected)"
        "\nAre vultures using the coast more or less than they used to?",
        **FONT_TITLE,
    )
    ax1.set_xlabel("Year", **FONT_LABEL)
    ax1.set_ylabel("Detection Rate (%)", **FONT_LABEL)
    ax1.tick_params(**FONT_TICK)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)

    # Overlay total checklists as context (secondary axis)
    ax2 = ax1.twinx()
    yearly_totals = total_by_year.sort_values("YEAR")
    ax2.bar(
        yearly_totals["YEAR"], yearly_totals["TOTAL_CHECKLISTS"],
        alpha=0.15, color="#999999", width=0.8, label="Total Checklists",
    )
    ax2.set_ylabel("Total Complete Checklists (grey bars)", fontsize=10, color="#999999")
    ax2.tick_params(axis="y", labelcolor="#999999")

    fig.tight_layout()
    _save_fig(fig, "gulf_yearly_trends.png")

    return merged


def main():
    # Check for C-CAP raster
    if not check_ccap_raster():
        return

    # Check for enriched eBird data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Enriched vulture CSV not found at: {DATA_PATH}\n"
            f"Run data_extraction.py first to create it."
        )

    # Load and filter to Gulf bounding box
    gulf_df = load_gulf_sightings(DATA_PATH)
    if len(gulf_df) == 0:
        print("No sightings found in Gulf bounding box. Check your data.")
        return

    gulf_df = add_ccap(gulf_df)

    gulf_density_analysis(gulf_df)
    seasonal_analysis(gulf_df)
    yearly_trend_analysis(gulf_df)

    print(f"\nGulf coastal analysis complete.")
    print(f"  CSVs:    {OUTPUT_FOLDER}/")
    print(f"  Figures: {FIGURES_FOLDER}/")


if __name__ == "__main__":
    main()