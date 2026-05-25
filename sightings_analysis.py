"""
Statistical Analysis
---------------------------------
Reads the vulture CSV from extraction script and produces analysis CSVs.

Outputs (all saved to ./analysis_results/):
    - habitat_pct.csv            : percent of sightings per habitat group, by species
    - habitat_by_season.csv      : habitat perecent broken down by season
    - seasonal_counts.csv        : monthly detection rates (effort-corrected, percentage of total checklists)
    - county_coverage.csv        : checklist density per county
    - underbirded_counties.csv   : counties with < threshold checklists
    - grid_coverage.csv          : lat/lon grid with observation/water/gap status
    - landscape_change.csv       : overall stable vs changed %
    - change_by_species.csv      : change type breakdown per species
    - change_by_habitat.csv      : which habitats overlap with change
    - change_by_season.csv       : does landscape change exposure vary by season

"""

import os
import numpy as np
import pandas as pd
import rasterio
import pyproj


INPUT_FILE    = "./data/vulture_sightings_with_habitat.csv"
OUTPUT_FOLDER = "./analysis_results"
LANDCOVER_PATH = os.path.join("./data/nlcd_landcover_data", "Annual_NLCD_LndCov_2024_CU_C1V1.tif")
UNDERBIRDED_THRESHOLD = 10
GRID_RESOLUTION = 0.1

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def _save(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(OUTPUT_FOLDER, name)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, parse_dates=["OBSERVATION DATE"], low_memory=False)
    print(f"  Rows: {len(df):,}")
    return df


# Habitat Percentage by Species
def habitat_percentages(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["HABITAT_GROUP"] != "Open Water"]
    counts = (
        filtered.groupby(["COMMON NAME", "HABITAT_GROUP"])
        .size().reset_index(name="COUNT")
    )
    counts["PCT"] = counts.groupby("COMMON NAME")["COUNT"].transform(
        lambda x: (x / x.sum() * 100).round(1)
    )

    _save(counts, "habitat_pct.csv")

    print("\n── Habitat Percentages saved ──")
    return counts


# Habitat Percentages by Season
def habitat_by_season(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["HABITAT_GROUP"] != "Open Water"]
    season_order = ["Spring", "Summer", "Fall", "Winter"]
    counts = (
        filtered.groupby(["COMMON NAME", "SEASON", "HABITAT_GROUP"])
        .size().reset_index(name="COUNT")
    )
    counts["PCT"] = counts.groupby(["COMMON NAME", "SEASON"])["COUNT"].transform(
        lambda x: (x / x.sum() * 100).round(1)
    )
    counts["SEASON"] = pd.Categorical(counts["SEASON"], categories=season_order, ordered=True)
    counts = counts.sort_values(["COMMON NAME", "SEASON", "PCT"], ascending=[True, True, False])

    _save(counts, "habitat_by_season.csv")

    print("\n── Habitat by Season saved ──")
    return counts


# Seasonal Detection Rates
def seasonal_detection_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MONTH"] = pd.to_datetime(df["OBSERVATION DATE"]).dt.month
    total = df.groupby("MONTH")["SAMPLING EVENT IDENTIFIER"].nunique().reset_index(name="TOTAL_CHECKLISTS")

    detected = (
        df.groupby(["COMMON NAME", "MONTH"])["SAMPLING EVENT IDENTIFIER"]
        .nunique().reset_index(name="DETECTED_CHECKLISTS")
    )

    merged = detected.merge(total, on="MONTH")
    merged["DETECTION_RATE"] = (merged["DETECTED_CHECKLISTS"] / merged["TOTAL_CHECKLISTS"] * 100).round(2)

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    season_map = {1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall",12:"Winter"}

    merged["MONTH_NAME"] = merged["MONTH"].map(month_names)
    merged["SEASON"] = merged["MONTH"].map(season_map)

    _save(merged, "seasonal_counts.csv")

    print("\n── Seasonal Detection Rates saved ──")
    return merged


# County Coverage
def county_coverage(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = (
        df.groupby(["STATE", "COUNTY"]).agg(
            TOTAL_CHECKLISTS=("SAMPLING EVENT IDENTIFIER", "nunique"),
            VULTURE_DETECTIONS=("COMMON NAME", "count"),
            LAT_MEAN=("LATITUDE", "mean"),
            LON_MEAN=("LONGITUDE", "mean"),
        ).reset_index()
    )
    coverage["DETECTION_RATE"] = (coverage["VULTURE_DETECTIONS"] / coverage["TOTAL_CHECKLISTS"] * 100).round(2)
    coverage["DATA_STATUS"] = coverage["TOTAL_CHECKLISTS"].apply(
        lambda x: "Underbirded" if x < UNDERBIRDED_THRESHOLD else "Adequate Coverage"
    )
    underbirded = coverage[coverage["DATA_STATUS"] == "Underbirded"].sort_values("TOTAL_CHECKLISTS")

    _save(coverage, "county_coverage.csv")
    _save(underbirded, "underbirded_counties.csv")

    print("\n── County Coverage saved ──")
    return coverage, underbirded


# Grid Coverage Gaps, find out if any areas are low on data
def grid_coverage_gaps(df: pd.DataFrame) -> pd.DataFrame:
    lat_bins = np.arange(25.8, 36.5, GRID_RESOLUTION).round(2)
    lon_bins = np.arange(-106.6, -88.8, GRID_RESOLUTION).round(2)
    grid_lats, grid_lons = np.meshgrid(lat_bins, lon_bins)
    grid = pd.DataFrame({"GRID_LAT": grid_lats.ravel(), "GRID_LON": grid_lons.ravel()})

    df = df.copy()
    df["GRID_LAT"] = (np.floor(df["LATITUDE"] / GRID_RESOLUTION) * GRID_RESOLUTION).round(2)
    df["GRID_LON"] = (np.floor(df["LONGITUDE"] / GRID_RESOLUTION) * GRID_RESOLUTION).round(2)

    observed = df.groupby(["GRID_LAT", "GRID_LON"]).agg(
        CHECKLIST_COUNT=("SAMPLING EVENT IDENTIFIER", "nunique"),
        DETECTION_COUNT=("COMMON NAME", "count"),
        DOMINANT_HABITAT=("HABITAT_GROUP", lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown"),
    ).reset_index()
    grid = grid.merge(observed, on=["GRID_LAT", "GRID_LON"], how="left")

    # Water from land cover raster
    print("  Looking up NLCD for grid water cells...")
    with rasterio.open(LANDCOVER_PATH) as src:
        transformer = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        band = src.read(1)
        h, w = band.shape
        xs, ys = transformer.transform(grid["GRID_LON"].values, grid["GRID_LAT"].values)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows, cols = np.array(rows), np.array(cols)
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        codes = np.full(len(rows), -1, dtype=np.int16)
        codes[valid] = band[rows[valid], cols[valid]]

    grid["STATUS"] = "No Data"
    grid.loc[codes == 11, "STATUS"] = "Open Water"
    grid.loc[grid["CHECKLIST_COUNT"].notna() & (grid["DETECTION_COUNT"] > 0), "STATUS"] = "Observed"

    _save(grid, "grid_coverage.csv")

    print("\n── Gap Coverage saved ──")
    return grid


# Landscape Change
def landscape_change_analysis(df: pd.DataFrame) -> None:
    """
    Analyze vulture sightings in stable vs changing landscapes.
    Produces four CSVs covering the change story from different angles.
    """
    try:
        total = len(df)
        changed = df[df["LANDSCAPE_CHANGE"] == True]
        stable = df[df["LANDSCAPE_CHANGE"] == False]

        # Overall summary
        summary = pd.DataFrame({
            "STATUS": ["Stable Habitat", "Changed Landscape"],
            "COUNT": [len(stable), len(changed)],
            "PCT": [round(len(stable)/total*100, 1), round(len(changed)/total*100, 1)],
        })
        _save(summary, "landscape_change.csv")

        # Change type by species
        by_species = df.groupby(["COMMON NAME", "CHANGE_TYPE"]).size().reset_index(name="COUNT")
        by_species["PCT"] = by_species.groupby("COMMON NAME")["COUNT"].transform(
            lambda x: (x / x.sum() * 100).round(1)
        )
        by_species = by_species.sort_values(["COMMON NAME", "COUNT"], ascending=[True, False])
        _save(by_species, "change_by_species.csv")

        print(f"\n── Change by Species ──")
        print(by_species.to_string(index=False))

        # Which habitats overlap with change
        if len(changed) > 0:
            by_habitat = (
                changed.groupby(["HABITAT_GROUP", "CHANGE_TYPE"])
                .size().reset_index(name="COUNT")
            )
            by_habitat = by_habitat.sort_values("COUNT", ascending=False)
            _save(by_habitat, "change_by_habitat.csv")

            print(f"\n── Change by Habitat (top 10) ──")
            print(by_habitat.head(10).to_string(index=False))
    except Exception as e:
        print(e)
        return


def main():
    df = load_data(INPUT_FILE)

    habitat_percentages(df)
    habitat_by_season(df)
    seasonal_detection_rates(df)
    county_coverage(df)
    grid_coverage_gaps(df)
    landscape_change_analysis(df)

    print(f"\nAll analysis saved to {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()