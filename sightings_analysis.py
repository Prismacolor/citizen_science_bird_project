"""
Reads vulture CSV from extraction script and produces
statistical tables used for visualization.

Outputs :
    - habitat_pct.csv          : % of sightings per habitat group, by species
    - habitat_by_season.csv    : habitat % broken down by season
    - seasonal_counts.csv      : monthly detection rates (per 100 checklists)
    - county_coverage.csv      : checklist density per county (underbirding map)
    - underbirded_counties.csv : counties with < threshold checklists
    - grid_coverage.csv        : lat/lon grid cells with/without data

pyproj is used for mapping and transforming coordinates
rasterio is used because GIS uses GeoTiff format to store data, and we need to be able to read that data
"""
import os
import pandas as pd
import numpy as np
import rasterio
import pyproj


INPUT_FILE    = "./data/vulture_sightings_with_habitat.csv"
OUTPUT_FOLDER = "analysis_results"
NLCD_PATH = "./data/nlcd/lcnext-1.0-stratum-map-Clipped.tif"

# Counties with fewer than this many checklists are flagged as underbirded (underrepresented)
UNDERBIRDED_THRESHOLD = 5

# Grid resolution in degrees for coverage gap analysis (~11km at this latitude)
GRID_RESOLUTION = 0.1


def load_data(path: str) -> pd.DataFrame:
    """
    Loads raw data from joined data file
    Parameter: str (path to collected data file)
    Return: dataframe
    """
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, parse_dates=["OBSERVATION DATE"], low_memory=False)
    print(f"  Rows loaded: {len(df):,}")
    return df


#  Habitat percentages by species
def habitat_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines what percentage of each species' sightings fall in each habitat group.
    Excludes water — a vulture over open water is likely just flying over.

    Params: dataframe - vulture data

    Returns: dataframe - data with added habitat info
    """
    counts = (
        df.groupby(["COMMON NAME", "HABITAT_GROUP"])
        .size()
        .reset_index(name="COUNT")
    )

    counts["PCT"] = counts.groupby("COMMON NAME")["COUNT"].transform(
        lambda x: (x / x.sum() * 100).round(1)
    )

    out_path = os.path.join(OUTPUT_FOLDER, "habitat_pct.csv")
    counts.to_csv(out_path, index=False)
    print(f"\nHabitat percentage by species saved to: {out_path}")

    return counts


# Habitat percentage by season
def habitat_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes if habitat use shifts between seasons

    Parameters: dataframe - vulture data

    Returns: dataframe - data with seasonality info
    """

    season_order = ["Spring", "Summer", "Fall", "Winter"]

    counts = (
        df.groupby(["COMMON NAME", "SEASON", "HABITAT_GROUP"])
        .size()
        .reset_index(name="COUNT")
    )

    counts["PCT"] = counts.groupby(["COMMON NAME", "SEASON"])["COUNT"].transform(
        lambda x: (x / x.sum() * 100).round(1)
    )

    counts["SEASON"] = pd.Categorical(counts["SEASON"], categories=season_order, ordered=True)
    counts = counts.sort_values(["COMMON NAME", "SEASON", "PCT"], ascending=[True, True, False])

    out_path = os.path.join(OUTPUT_FOLDER, "habitat_by_season.csv")
    counts.to_csv(out_path, index=False)
    print(f"\nHabitat by season saved to: {out_path}")

    return counts


# Seasonal Detection Rates
def seasonal_detection_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly detection rate = (checklists with species / total checklists) * 100
    This is effort-corrected so busy months don't distort results.
    We approximate total checklists per month using unique SAMPLING EVENT IDENTIFIERs.
    The SEI identifies checklists

    Params: dataframe - vulture data

    Returns: dataframe - data joined by month
    """
    df["MONTH"] = pd.to_datetime(df["OBSERVATION DATE"]).dt.month
    df["MONTH_NAME"] = pd.to_datetime(df["OBSERVATION DATE"]).dt.strftime("%b")

    # Total unique checklists per month across all observers
    total_checklists = (
        df.groupby("MONTH")["SAMPLING EVENT IDENTIFIER"]
        .nunique()
        .reset_index(name="TOTAL_CHECKLISTS")
    )

    # Checklists where each species was detected
    detections = (
        df.groupby(["COMMON NAME", "MONTH"])["SAMPLING EVENT IDENTIFIER"]
        .nunique()
        .reset_index(name="DETECTED_CHECKLISTS")
    )

    merged = detections.merge(total_checklists, on="MONTH")
    merged["DETECTION_RATE"] = (
        merged["DETECTED_CHECKLISTS"] / merged["TOTAL_CHECKLISTS"] * 100
    ).round(2)

    month_names = {
        1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
        7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
    }
    merged["MONTH_NAME"] = merged["MONTH"].map(month_names)

    season_map = {
        1:"Winter", 2:"Winter", 3:"Spring", 4:"Spring", 5:"Spring",
        6:"Summer", 7:"Summer", 8:"Summer", 9:"Fall", 10:"Fall",
        11:"Fall",  12:"Winter"
    }
    merged["SEASON"] = merged["MONTH"].map(season_map)

    out_path = os.path.join(OUTPUT_FOLDER, "seasonal_counts.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nSeasonal detection rates saved to: {out_path}")

    return merged


# County Coverage (Underbirding)
def county_coverage(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Count unique checklists per county.
    Counties below UNDERBIRDED_THRESHOLD are flagged as potential data gaps.
    Low checklists + no detections = not sure if species is present
    High checklists + no detections = species is likely absent
    We divide by total number of checklists to check proportions of sightings.
    That way data is not skewed by more birders being present that month.

    Params: dataframe - vulture data

    Returns: tuple(dataframe, dataframe) - coverage data, underrepped areas
    """
    coverage = (
        df.groupby(["STATE", "COUNTY"])
        .agg(
            TOTAL_CHECKLISTS=("SAMPLING EVENT IDENTIFIER", "nunique"),
            VULTURE_DETECTIONS=("COMMON NAME", "count"),
            LAT_MEAN=("LATITUDE", "mean"),
            LON_MEAN=("LONGITUDE", "mean"),
        )
        .reset_index()
    )

    coverage["DETECTION_RATE"] = (
        coverage["VULTURE_DETECTIONS"] / coverage["TOTAL_CHECKLISTS"] * 100
    ).round(2)

    coverage["DATA_STATUS"] = coverage["TOTAL_CHECKLISTS"].apply(
        lambda x: "Underbirded" if x < UNDERBIRDED_THRESHOLD else "Adequate Coverage"
    )

    underbirded = coverage[coverage["DATA_STATUS"] == "Underbirded"].sort_values(
        "TOTAL_CHECKLISTS"
    )

    coverage_path    = os.path.join(OUTPUT_FOLDER, "county_coverage.csv")
    underbirded_path = os.path.join(OUTPUT_FOLDER, "underbirded_counties.csv")

    coverage.to_csv(coverage_path, index=False)
    underbirded.to_csv(underbirded_path, index=False)

    print(f"\nCounty coverage saved to: {coverage_path}")

    return coverage, underbirded


# Spatial Grid Coverage Gaps
def grid_coverage_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Divide the four-state region into a lat/lon grid.
    Mark each cell as:
      - 'Observed'              : vultures detected here
      - 'Open Water'            : NLCD raster says this is water
      - 'No Data'               : no checklists at all

    Uses the NLCD raster directly for water classification so
    the Gulf of Mexico and lakes are tagged correctly even without
    observation data.

    Params:
    df: Vulture observation DataFrame

    Returns:
    DataFrame — grid map with STATUS column
    """
    raster_path = NLCD_PATH

    # Build grid over the bounding box of the four states
    lat_min, lat_max = 25.8, 36.5
    lon_min, lon_max = -106.6, -88.8

    lat_bins = np.arange(lat_min, lat_max, GRID_RESOLUTION).round(2)
    lon_bins = np.arange(lon_min, lon_max, GRID_RESOLUTION).round(2)

    grid_lats, grid_lons = np.meshgrid(lat_bins, lon_bins)
    grid = pd.DataFrame({
        "GRID_LAT": grid_lats.ravel(),
        "GRID_LON": grid_lons.ravel(),
    })

    # Assign each observation to a grid cell
    df["GRID_LAT"] = (np.floor(df["LATITUDE"] / GRID_RESOLUTION) * GRID_RESOLUTION).round(2)
    df["GRID_LON"] = (np.floor(df["LONGITUDE"] / GRID_RESOLUTION) * GRID_RESOLUTION).round(2)

    # Aggregate observations per grid cell
    observed_cells = df.groupby(["GRID_LAT", "GRID_LON"]).agg(
        CHECKLIST_COUNT=("SAMPLING EVENT IDENTIFIER", "nunique"),
        DETECTION_COUNT=("COMMON NAME", "count"),
        DOMINANT_HABITAT=("HABITAT_GROUP", lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown"),
    ).reset_index()

    grid = grid.merge(observed_cells, on=["GRID_LAT", "GRID_LON"], how="left")

    # Tag water cells directly from the NLCD raster
    print("Looking up NLCD land cover for grid cells...")

    with rasterio.open(raster_path) as src:
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", src.crs, always_xy=True
        )
        band = src.read(1)
        height, width = band.shape

        lons = grid["GRID_LON"].values
        lats = grid["GRID_LAT"].values

        # Reproject grid cell centers to raster coordinates
        xs, ys = transformer.transform(lons, lats)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows = np.array(rows)
        cols = np.array(cols)

        # Look up NLCD value at each grid cell center
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        nlcd_codes = np.full(len(rows), fill_value=-1, dtype=np.int16)
        nlcd_codes[valid] = band[rows[valid], cols[valid]]

    grid["NLCD_GRID_CODE"] = nlcd_codes
    grid["IS_WATER"] = grid["NLCD_GRID_CODE"] == 11

    # Assign status (priority: observations > surveyed > water > no data)
    def assign_status(row):
        if pd.notna(row["CHECKLIST_COUNT"]) and row["DETECTION_COUNT"] > 0:
            return "Observed"
        elif row["IS_WATER"] == True:
            return "Open Water"
        else:
            return "No Data"

    grid["STATUS"] = grid.apply(assign_status, axis=1)

    # Clean up helper columns before saving
    grid = grid.drop(columns=["NLCD_GRID_CODE", "IS_WATER"])

    out_path = os.path.join(OUTPUT_FOLDER, "grid_coverage.csv")
    grid.to_csv(out_path, index=False)
    print(f"\nGrid coverage saved to: {out_path}")

    return grid


def main():
    df = load_data(INPUT_FILE)

    filtered = df[df["HABITAT_GROUP"] != "Water"]
    habitat_percentages(filtered)
    habitat_by_season(filtered)

    seasonal_detection_rates(df)
    county_coverage(df)
    grid_coverage_gaps(df)

    print("\nAll analysis files saved to ./analysis/")


if __name__ == "__main__":
    main()