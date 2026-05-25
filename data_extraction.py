"""
 Load eBird and land cover data and join
-----------------------------------------------------
Reads large eBird .txt files using Dask, filters for Black Vulture
and Turkey Vulture, then performs TWO spatial joins:

  1. Land Cover raster (codes 11–95) → what habitat IS this?
  2. Strata raster (codes 100–104)  → is this landscape changing?

Also assigns season
"""

import os
import glob
import dask.dataframe as dd
import pandas as pd
import numpy as np
import rasterio
import pyproj


DATA_FOLDER     = "./data/ebird_data_files"
OUTPUT_FILE     = "./data/vulture_sightings_with_habitat.csv"

# landcover tells habitat, strata shows changes
LANDCOVER_PATH  = os.path.join("./data/nlcd_landcover_data", "Annual_NLCD_LndCov_2024_CU_C1V1.tif")
STRATA_PATH     = os.path.join("./data/nlcd_strata_data", "lcnext-1.0-stratum-map-Clipped.tif")

TARGET_COMMON_NAMES     = ["Black Vulture", "Turkey Vulture"]
TARGET_SCIENTIFIC_NAMES = ["Coragyps atratus", "Cathartes aura"]

COLS_NEEDED = [
    "GLOBAL UNIQUE IDENTIFIER", "COMMON NAME", "SCIENTIFIC NAME",
    "OBSERVATION COUNT", "LATITUDE", "LONGITUDE", "OBSERVATION DATE",
    "STATE", "COUNTY", "LOCALITY TYPE", "SAMPLING EVENT IDENTIFIER",
    "ALL SPECIES REPORTED", "DURATION MINUTES", "EFFORT DISTANCE KM",
    "NUMBER OBSERVERS",
]

NLCD_LABELS = {
    11: "Open Water", 12: "Perennial Ice/Snow",
    21: "Developed - Open Space", 22: "Developed - Low Intensity",
    23: "Developed - Medium Intensity", 24: "Developed - High Intensity",
    31: "Barren Land", 41: "Deciduous Forest", 42: "Evergreen Forest",
    43: "Mixed Forest", 51: "Dwarf Scrub", 52: "Shrub/Scrub",
    71: "Grassland/Herbaceous", 72: "Sedge/Herbaceous",
    73: "Lichens", 74: "Moss",
    81: "Pasture/Hay", 82: "Cultivated Crops",
    90: "Woody Wetlands", 95: "Emergent Herbaceous Wetlands",
}

HABITAT_GROUPS = {
    "Open Water": "Open Water",
    "Perennial Ice/Snow": "Ice/Snow",
    "Developed - Open Space": "Developed: Open Space",
    "Developed - Low Intensity": "Developed: Low Intensity",
    "Developed - Medium Intensity": "Developed: Medium Intensity",
    "Developed - High Intensity": "Developed: High Intensity",
    "Barren Land": "Barren Land",
    "Deciduous Forest": "Deciduous Forest",
    "Evergreen Forest": "Evergreen Forest",
    "Mixed Forest": "Mixed Forest",
    "Dwarf Scrub": "Dwarf/Scrub",
    "Shrub/Scrub": "Shrub/Scrub",
    "Grassland/Herbaceous": "Grassland",
    "Sedge/Herbaceous": "Sedge",
    "Pasture/Hay": "Pasture",
    "Cultivated Crops": "Cultivated Crops",
    "Woody Wetlands": "Woody Wetlands",
    "Emergent Herbaceous Wetlands": "Emergent Wetlands",
    "Lichens": "Lichens",
    "Moss": "Moss",
}

HABITAT_CHANGE_LABELS = {
    100: "Urban Gain",
    101: "Cropland Change",
    102: "Wetland/Water Flux",
    103: "Rare Change",
    104: "Common Change",
}


def assign_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


def raster_lookup(lats: np.ndarray, lons: np.ndarray, raster_path: str) -> np.ndarray:
    """Look up raster pixel values for lat/lon arrays. Returns -1 for out-of-bounds."""
    with rasterio.open(raster_path) as src:
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


def extract_ebird_data(folder: str) -> pd.DataFrame:
    """Load eBird .txt files with Dask, keep only vulture records.

    Params: folder: str - path to ebird data

    Returns: dataframe with preprocessed data
    """
    files = glob.glob(os.path.join(folder, "*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {folder}")

    print(f"\nFound {len(files)} eBird file(s). Loading with Dask...")

    df = dd.read_csv(
        files, sep="\t", dtype=str, on_bad_lines="skip",
        engine="python", quoting=3,
        usecols=lambda c: c.strip() in COLS_NEEDED,
    )
    df.columns = df.columns.str.strip()

    common_mask = df["COMMON NAME"].str.strip().isin(TARGET_COMMON_NAMES)
    sci_mask = df["SCIENTIFIC NAME"].str.strip().isin(TARGET_SCIENTIFIC_NAMES)
    df = df[common_mask | sci_mask]
    df = df[df["ALL SPECIES REPORTED"].str.strip() == "1"]

    print("Computing filtered results. This may take a few minutes...")
    result = df.compute()

    result["LATITUDE"] = pd.to_numeric(result["LATITUDE"], errors="coerce")
    result["LONGITUDE"] = pd.to_numeric(result["LONGITUDE"], errors="coerce")
    result["OBSERVATION DATE"] = pd.to_datetime(result["OBSERVATION DATE"], errors="coerce")
    result = result.dropna(subset=["LATITUDE", "LONGITUDE", "OBSERVATION DATE"])

    print(f"Vulture records loaded: {len(result):,}")
    return result


def join_landcover_and_strata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two raster lookups per observation:
      1. Land Cover → NLCD_CODE, HABITAT, HABITAT_GROUP
      2. Strata     → STRATA_CODE, LANDSCAPE_CHANGE, CHANGE_TYPE

     Params: df: dataframe - preprocessed vulture data

     Return: new dataframe with raster info
    """
    lats = df["LATITUDE"].values
    lons = df["LONGITUDE"].values

    print(f"\nLand cover lookup: {LANDCOVER_PATH}")
    lc_codes = raster_lookup(lats, lons, LANDCOVER_PATH)

    df = df.copy()
    df["NLCD_CODE"] = lc_codes
    df["HABITAT"] = df["NLCD_CODE"].map(NLCD_LABELS).fillna("Unknown")
    df["HABITAT_GROUP"] = df["HABITAT"].map(HABITAT_GROUPS).fillna("Other")

    print("Land cover join complete:")
    print(f"\nStrata lookup: {STRATA_PATH}")

    strata_codes = raster_lookup(lats, lons, STRATA_PATH)

    df["STRATA_CODE"] = strata_codes
    df["LANDSCAPE_CHANGE"] = df["STRATA_CODE"].isin(HABITAT_CHANGE_LABELS.keys())
    df["CHANGE_TYPE"] = df["STRATA_CODE"].map(HABITAT_CHANGE_LABELS).fillna("Stable")

    print(f"\nDual-raster join complete:")

    return df


def main():
    for path, name in [(LANDCOVER_PATH, "Land Cover"), (STRATA_PATH, "Strata")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} raster not found at: {path}")

    df = extract_ebird_data(DATA_FOLDER)

    df["MONTH"] = df["OBSERVATION DATE"].dt.month
    df["YEAR"] = df["OBSERVATION DATE"].dt.year
    df["SEASON"] = df["MONTH"].apply(assign_season)

    df = join_landcover_and_strata(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()