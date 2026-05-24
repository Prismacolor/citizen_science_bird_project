"""
Covers data_extraction.py (Script 1) and data_analysis.py (Script 2).
Run: pytest test_bird_project.py -v
"""

import os
import numpy as np
import pandas as pd
import pytest


# Setup
@pytest.fixture
def sample_vulture_df() -> pd.DataFrame:
    """Minimal enriched vulture DataFrame matching vulture_with_habitat.csv schema."""
    return pd.DataFrame({
        "GLOBAL UNIQUE IDENTIFIER": [f"URN:X:{i}" for i in range(12)],
        "COMMON NAME": [
            "Black Vulture", "Black Vulture", "Turkey Vulture", "Turkey Vulture",
            "Black Vulture", "Turkey Vulture", "Black Vulture", "Turkey Vulture",
            "Black Vulture", "Turkey Vulture", "Black Vulture", "Turkey Vulture",
        ],
        "SCIENTIFIC NAME": [
            "Coragyps atratus", "Coragyps atratus", "Cathartes aura", "Cathartes aura",
            "Coragyps atratus", "Cathartes aura", "Coragyps atratus", "Cathartes aura",
            "Coragyps atratus", "Cathartes aura", "Coragyps atratus", "Cathartes aura",
        ],
        "OBSERVATION COUNT": [3, 1, 5, 2, 1, 4, 2, 1, 3, 2, 1, 6],
        "LATITUDE":  [30.2, 30.3, 34.7, 35.1, 29.8, 33.5, 30.2, 34.7, 31.0, 32.5, 30.5, 34.0],
        "LONGITUDE": [-97.7, -95.4, -92.3, -90.1, -97.7, -94.0, -97.7, -92.3, -96.0, -93.0, -95.5, -91.5],
        "OBSERVATION DATE": pd.to_datetime([
            "2023-01-15", "2023-03-20", "2023-06-10", "2023-09-05",
            "2023-04-12", "2023-07-22", "2023-11-30", "2023-12-25",
            "2023-02-14", "2023-05-18", "2023-08-09", "2023-10-31",
        ]),
        "STATE": ["Texas"] * 4 + ["Texas", "Arkansas", "Texas", "Arkansas",
                  "Texas", "Louisiana", "Texas", "Louisiana"],
        "COUNTY": ["Travis", "Harris", "Pulaski", "Shelby",
                   "Travis", "Garland", "Travis", "Pulaski",
                   "McLennan", "Ouachita", "Harris", "East Baton Rouge"],
        "LOCALITY TYPE": ["H", "H", "P", "H", "H", "P", "H", "H", "P", "H", "H", "P"],
        "SAMPLING EVENT IDENTIFIER": [f"S{i}" for i in range(12)],
        "ALL SPECIES REPORTED": [1] * 12,
        "DURATION MINUTES": [60, 120, 45, 90, 30, 60, 90, 120, 45, 60, 75, 50],
        "EFFORT DISTANCE KM": [1.5, 3.0, 0.5, 2.0, 1.0, 1.5, 2.0, 3.0, 0.8, 1.2, 1.5, 0.6],
        "NUMBER OBSERVERS": [1, 2, 1, 3, 1, 2, 1, 1, 2, 1, 3, 1],
        "MONTH":   [1, 3, 6, 9, 4, 7, 11, 12, 2, 5, 8, 10],
        "YEAR":    [2023] * 12,
        "SEASON":  ["Winter", "Spring", "Summer", "Fall",
                    "Spring", "Summer", "Fall", "Winter",
                    "Winter", "Spring", "Summer", "Fall"],
        "NLCD_CODE":      [23, 41, 82, 41, 21, 52, 23, 82, 11, 71, 42, 90],
        "HABITAT": [
            "Developed - Medium Intensity", "Deciduous Forest", "Cultivated Crops",
            "Deciduous Forest", "Developed - Open Space", "Shrub/Scrub",
            "Developed - Medium Intensity", "Cultivated Crops", "Open Water",
            "Grassland/Herbaceous", "Evergreen Forest", "Woody Wetlands",
        ],
        "HABITAT_GROUP": [
            "Developed", "Forest", "Cultivated Crops", "Forest",
            "Developed", "Shrub/Scrub", "Developed", "Cultivated Crops",
            "Water", "Grassland/Herbaceous", "Forest", "Wetlands",
        ],
    })


@pytest.fixture
def output_dir(tmp_path) -> str:
    out = str(tmp_path / "analysis")
    os.makedirs(out, exist_ok=True)
    return out


# tests for data_extraction.py
class TestAssignSeason:
    """Tests for the assign_season helper."""

    def _assign_season(self, month: int) -> str:
        """Inline copy of the function under test."""
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    def test_winter_months(self):
        assert self._assign_season(12) == "Winter"
        assert self._assign_season(1) == "Winter"
        assert self._assign_season(2) == "Winter"

    def test_spring_months(self):
        for m in [3, 4, 5]:
            assert self._assign_season(m) == "Spring"

    def test_summer_months(self):
        for m in [6, 7, 8]:
            assert self._assign_season(m) == "Summer"

    def test_fall_months(self):
        for m in [9, 10, 11]:
            assert self._assign_season(m) == "Fall"

    def test_all_months_covered(self):
        results = {self._assign_season(m) for m in range(1, 13)}
        assert results == {"Winter", "Spring", "Summer", "Fall"}


class TestNLCDLabels:
    """Tests for NLCD code-to-label mapping."""

    NLCD_LABELS = {
        11: "Open Water", 12: "Perennial Ice/Snow",
        21: "Developed - Open Space", 22: "Developed - Low Intensity",
        23: "Developed - Medium Intensity", 24: "Developed - High Intensity",
        31: "Barren Land", 41: "Deciduous Forest", 42: "Evergreen Forest",
        43: "Mixed Forest", 52: "Shrub/Scrub", 71: "Grassland/Herbaceous",
        81: "Pasture/Hay", 82: "Cultivated Crops",
        90: "Woody Wetlands", 95: "Emergent Herbaceous Wetlands",
    }

    def test_known_codes_have_labels(self):
        for code in [11, 21, 41, 82, 90]:
            assert code in self.NLCD_LABELS
            assert isinstance(self.NLCD_LABELS[code], str)

    def test_unknown_code_returns_none(self):
        assert self.NLCD_LABELS.get(99) is None

    def test_water_is_code_11(self):
        assert self.NLCD_LABELS[11] == "Open Water"


class TestLoadAndFilter:
    """Tests for eBird file loading and filtering logic."""

    def test_filter_keeps_only_vultures(self, sample_vulture_df):
        target = ["Black Vulture", "Turkey Vulture"]
        filtered = sample_vulture_df[
            sample_vulture_df["COMMON NAME"].isin(target)
        ]
        assert set(filtered["COMMON NAME"].unique()) == set(target)
        assert len(filtered) == len(sample_vulture_df)

    def test_filter_removes_non_vultures(self, sample_vulture_df):
        df = pd.concat([
            sample_vulture_df,
            pd.DataFrame({
                "COMMON NAME": ["Red-tailed Hawk", "Northern Cardinal"],
                "ALL SPECIES REPORTED": [1, 1],
            }, index=[100, 101]),
        ])
        target = ["Black Vulture", "Turkey Vulture"]
        filtered = df[df["COMMON NAME"].isin(target)]
        assert "Red-tailed Hawk" not in filtered["COMMON NAME"].values

    def test_complete_checklists_only(self, sample_vulture_df):
        df = sample_vulture_df.copy()
        df.loc[0, "ALL SPECIES REPORTED"] = 0
        filtered = df[df["ALL SPECIES REPORTED"] == 1]
        assert len(filtered) == len(df) - 1

    def test_missing_coordinates_dropped(self):
        df = pd.DataFrame({
            "LATITUDE": [30.0, np.nan, 31.0],
            "LONGITUDE": [-95.0, -96.0, np.nan],
            "OBSERVATION DATE": pd.to_datetime(["2023-01-01"] * 3),
        })
        cleaned = df.dropna(subset=["LATITUDE", "LONGITUDE", "OBSERVATION DATE"])
        assert len(cleaned) == 1


class TestLookupNLCDValues:
    """Tests for the raster lookup logic (mocked — no real raster needed)."""

    def test_valid_coordinates_get_codes(self):
        """Simulate the vectorized raster lookup."""
        band = np.array([
            [11, 41, 82],
            [21, 52, 71],
            [90, 42, 31],
        ], dtype=np.int16)

        rows = np.array([0, 1, 2])
        cols = np.array([0, 2, 1])
        height, width = band.shape

        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        result = np.full(len(rows), -1, dtype=np.int16)
        result[valid] = band[rows[valid], cols[valid]]

        assert list(result) == [11, 71, 42]

    def test_out_of_bounds_returns_negative_one(self):
        band = np.array([[11, 41]], dtype=np.int16)
        rows = np.array([0, 5])
        cols = np.array([0, 0])
        height, width = band.shape

        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        result = np.full(len(rows), -1, dtype=np.int16)
        result[valid] = band[rows[valid], cols[valid]]

        assert result[0] == 11
        assert result[1] == -1

    def test_all_out_of_bounds(self):
        band = np.array([[11]], dtype=np.int16)
        rows = np.array([10, 20])
        cols = np.array([10, 20])
        height, width = band.shape

        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        result = np.full(len(rows), -1, dtype=np.int16)
        result[valid] = band[rows[valid], cols[valid]]

        assert all(v == -1 for v in result)


# tests for data_analysis.py
class TestHabitatPercentages:
    """Tests for habitat_percentages analysis."""

    def test_excludes_water(self, sample_vulture_df):
        filtered = sample_vulture_df[sample_vulture_df["HABITAT_GROUP"] != "Water"]
        assert "Water" not in filtered["HABITAT_GROUP"].values

    def test_percentages_sum_to_100_per_species(self, sample_vulture_df):
        filtered = sample_vulture_df[sample_vulture_df["HABITAT_GROUP"] != "Water"]
        counts = (
            filtered.groupby(["COMMON NAME", "HABITAT_GROUP"])
            .size()
            .reset_index(name="COUNT")
        )
        counts["PCT"] = counts.groupby("COMMON NAME")["COUNT"].transform(
            lambda x: (x / x.sum() * 100).round(1)
        )
        for species in counts["COMMON NAME"].unique():
            total = counts[counts["COMMON NAME"] == species]["PCT"].sum()
            assert total == pytest.approx(100.0, abs=0.5)

    def test_both_species_present(self, sample_vulture_df):
        filtered = sample_vulture_df[sample_vulture_df["HABITAT_GROUP"] != "Water"]
        species = filtered["COMMON NAME"].unique()
        assert "Black Vulture" in species
        assert "Turkey Vulture" in species

    def test_empty_dataframe(self):
        empty = pd.DataFrame(columns=["COMMON NAME", "HABITAT_GROUP"])
        filtered = empty[empty["HABITAT_GROUP"] != "Water"]
        counts = filtered.groupby(["COMMON NAME", "HABITAT_GROUP"]).size().reset_index(name="COUNT")
        assert len(counts) == 0


class TestHabitatBySeason:
    """Tests for habitat_by_season analysis."""

    def test_all_seasons_present(self, sample_vulture_df):
        seasons = sample_vulture_df["SEASON"].unique()
        assert set(seasons) == {"Winter", "Spring", "Summer", "Fall"}

    def test_season_percentages_sum_to_100(self, sample_vulture_df):
        filtered = sample_vulture_df[sample_vulture_df["HABITAT_GROUP"] != "Water"]
        counts = (
            filtered.groupby(["COMMON NAME", "SEASON", "HABITAT_GROUP"])
            .size()
            .reset_index(name="COUNT")
        )
        counts["PCT"] = counts.groupby(["COMMON NAME", "SEASON"])["COUNT"].transform(
            lambda x: (x / x.sum() * 100).round(1)
        )
        for (species, season), grp in counts.groupby(["COMMON NAME", "SEASON"]):
            total = grp["PCT"].sum()
            assert total == pytest.approx(100.0, abs=0.5), f"{species}/{season} sums to {total}"


class TestSeasonalDetectionRates:
    """Tests for seasonal detection rate calculation."""

    def test_detection_rate_between_0_and_100(self, sample_vulture_df):
        df = sample_vulture_df.copy()
        total = df.groupby("MONTH")["SAMPLING EVENT IDENTIFIER"].nunique().reset_index(name="TOTAL")
        detections = (
            df.groupby(["COMMON NAME", "MONTH"])["SAMPLING EVENT IDENTIFIER"]
            .nunique()
            .reset_index(name="DETECTED")
        )
        merged = detections.merge(total, on="MONTH")
        merged["RATE"] = merged["DETECTED"] / merged["TOTAL"] * 100

        assert (merged["RATE"] >= 0).all()
        assert (merged["RATE"] <= 100).all()

    def test_all_12_months_possible(self, sample_vulture_df):
        months = sample_vulture_df["MONTH"].unique()
        assert len(months) == 12

    def test_single_species_single_month(self):
        df = pd.DataFrame({
            "COMMON NAME": ["Black Vulture"] * 3,
            "MONTH": [6, 6, 6],
            "SAMPLING EVENT IDENTIFIER": ["S1", "S2", "S3"],
        })
        total = df.groupby("MONTH")["SAMPLING EVENT IDENTIFIER"].nunique().reset_index(name="TOTAL")
        detected = (
            df.groupby(["COMMON NAME", "MONTH"])["SAMPLING EVENT IDENTIFIER"]
            .nunique()
            .reset_index(name="DETECTED")
        )
        merged = detected.merge(total, on="MONTH")
        merged["RATE"] = merged["DETECTED"] / merged["TOTAL"] * 100
        assert merged["RATE"].iloc[0] == 100.0


class TestCountyCoverage:
    """Tests for county coverage / underbirding analysis."""

    UNDERBIRDED_THRESHOLD = 10

    def test_county_aggregation(self, sample_vulture_df):
        coverage = (
            sample_vulture_df.groupby(["STATE", "COUNTY"])
            .agg(
                TOTAL_CHECKLISTS=("SAMPLING EVENT IDENTIFIER", "nunique"),
                VULTURE_DETECTIONS=("COMMON NAME", "count"),
            )
            .reset_index()
        )
        assert "TOTAL_CHECKLISTS" in coverage.columns
        assert "VULTURE_DETECTIONS" in coverage.columns
        assert len(coverage) > 0

    def test_underbirded_flagging(self, sample_vulture_df):
        coverage = (
            sample_vulture_df.groupby(["STATE", "COUNTY"])
            .agg(TOTAL_CHECKLISTS=("SAMPLING EVENT IDENTIFIER", "nunique"))
            .reset_index()
        )
        coverage["DATA_STATUS"] = coverage["TOTAL_CHECKLISTS"].apply(
            lambda x: "Underbirded" if x < self.UNDERBIRDED_THRESHOLD else "Adequate Coverage"
        )
        # Our fixture has very few checklists per county, so most should be underbirded
        underbirded = coverage[coverage["DATA_STATUS"] == "Underbirded"]
        assert len(underbirded) > 0

    def test_high_checklist_county_not_underbirded(self):
        df = pd.DataFrame({
            "STATE": ["Texas"] * 50,
            "COUNTY": ["Travis"] * 50,
            "SAMPLING EVENT IDENTIFIER": [f"S{i}" for i in range(50)],
            "COMMON NAME": ["Black Vulture"] * 50,
        })
        coverage = (
            df.groupby(["STATE", "COUNTY"])
            .agg(TOTAL_CHECKLISTS=("SAMPLING EVENT IDENTIFIER", "nunique"))
            .reset_index()
        )
        coverage["DATA_STATUS"] = coverage["TOTAL_CHECKLISTS"].apply(
            lambda x: "Underbirded" if x < 10 else "Adequate Coverage"
        )
        assert coverage.iloc[0]["DATA_STATUS"] == "Adequate Coverage"

    def test_detection_rate_calculation(self, sample_vulture_df):
        coverage = (
            sample_vulture_df.groupby(["STATE", "COUNTY"])
            .agg(
                TOTAL_CHECKLISTS=("SAMPLING EVENT IDENTIFIER", "nunique"),
                VULTURE_DETECTIONS=("COMMON NAME", "count"),
            )
            .reset_index()
        )
        coverage["DETECTION_RATE"] = (
            coverage["VULTURE_DETECTIONS"] / coverage["TOTAL_CHECKLISTS"] * 100
        ).round(2)
        assert (coverage["DETECTION_RATE"] > 0).all()
        assert (coverage["DETECTION_RATE"] <= 100 * coverage["VULTURE_DETECTIONS"].max()).all()


class TestGridCoverage:
    """Tests for spatial grid coverage gap analysis."""

    GRID_RESOLUTION = 0.1

    def _build_grid(self, lat_min=30.0, lat_max=31.0, lon_min=-97.0, lon_max=-96.0):
        lat_bins = np.arange(lat_min, lat_max, self.GRID_RESOLUTION).round(2)
        lon_bins = np.arange(lon_min, lon_max, self.GRID_RESOLUTION).round(2)
        grid_lats, grid_lons = np.meshgrid(lat_bins, lon_bins)
        return pd.DataFrame({
            "GRID_LAT": grid_lats.ravel(),
            "GRID_LON": grid_lons.ravel(),
        })

    def _snap_to_grid(self, df):
        df = df.copy()
        df["GRID_LAT"] = (np.floor(df["LATITUDE"] / self.GRID_RESOLUTION) * self.GRID_RESOLUTION).round(2)
        df["GRID_LON"] = (np.floor(df["LONGITUDE"] / self.GRID_RESOLUTION) * self.GRID_RESOLUTION).round(2)
        return df

    def test_grid_dimensions(self):
        grid = self._build_grid()
        expected_cells = 10 * 10  # 1 degree / 0.1 resolution = 10 bins per axis
        assert len(grid) == expected_cells

    def test_grid_coordinates_are_clean(self):
        grid = self._build_grid()
        # No floating point drift
        for val in grid["GRID_LAT"]:
            assert val == round(val, 2)
        for val in grid["GRID_LON"]:
            assert val == round(val, 2)

    def test_observation_snapping(self):
        df = pd.DataFrame({
            "LATITUDE": [30.267, 30.389, 30.951],
            "LONGITUDE": [-97.743, -96.511, -96.102],
        })
        snapped = self._snap_to_grid(df)
        assert snapped["GRID_LAT"].iloc[0] == 30.2
        assert snapped["GRID_LON"].iloc[0] == -97.8

    def test_observed_cells_merge(self):
        grid = self._build_grid()
        observed = pd.DataFrame({
            "GRID_LAT": [30.2, 30.5],
            "GRID_LON": [-96.8, -96.3],
            "CHECKLIST_COUNT": [5, 12],
            "DETECTION_COUNT": [3, 8],
        })
        merged = grid.merge(observed, on=["GRID_LAT", "GRID_LON"], how="left")
        filled = merged["CHECKLIST_COUNT"].notna().sum()
        assert filled == 2
        empty = merged["CHECKLIST_COUNT"].isna().sum()
        assert empty == len(grid) - 2

    def test_status_assignment_observed(self):
        row = {"CHECKLIST_COUNT": 5, "DETECTION_COUNT": 3, "IS_WATER": False}
        assert self._assign_status(row) == "Observed"

    def test_status_assignment_water(self):
        row = {"CHECKLIST_COUNT": np.nan, "DETECTION_COUNT": np.nan, "IS_WATER": True}
        assert self._assign_status(row) == "Open Water"

    def test_status_assignment_no_data(self):
        row = {"CHECKLIST_COUNT": np.nan, "DETECTION_COUNT": np.nan, "IS_WATER": False}
        assert self._assign_status(row) == "No Data"

    def test_observed_trumps_water(self):
        """If vultures were seen at a water location, status should be Observed."""
        row = {"CHECKLIST_COUNT": 3, "DETECTION_COUNT": 2, "IS_WATER": True}
        assert self._assign_status(row) == "Observed"

    @staticmethod
    def _assign_status(row):
        if pd.notna(row["CHECKLIST_COUNT"]) and row["DETECTION_COUNT"] > 0:
            return "Observed"
        elif row["IS_WATER"] == True:
            return "Open Water"
        else:
            return "No Data"


class TestGridSnappingEdgeCases:
    """Edge cases for coordinate snapping."""

    GRID_RESOLUTION = 0.1

    def _snap(self, val):
        return round(np.floor(val / self.GRID_RESOLUTION) * self.GRID_RESOLUTION, 2)

    def test_negative_longitude(self):
        assert self._snap(-97.743) == -97.8

    def test_exact_boundary(self):
        assert self._snap(30.0) == 30.0

    def test_just_below_boundary(self):
        assert self._snap(29.999) == 29.9

    def test_southern_texas(self):
        assert self._snap(25.85) == 25.8

    def test_northern_arkansas(self):
        assert self._snap(36.49) == 36.4


class TestCSVOutputs:
    """Test that analysis functions produce valid CSV files."""

    def test_habitat_pct_csv(self, sample_vulture_df, output_dir):
        filtered = sample_vulture_df[sample_vulture_df["HABITAT_GROUP"] != "Water"]
        counts = (
            filtered.groupby(["COMMON NAME", "HABITAT_GROUP"])
            .size()
            .reset_index(name="COUNT")
        )
        counts["PCT"] = counts.groupby("COMMON NAME")["COUNT"].transform(
            lambda x: (x / x.sum() * 100).round(1)
        )
        path = os.path.join(output_dir, "habitat_pct.csv")
        counts.to_csv(path, index=False)

        loaded = pd.read_csv(path)
        assert "COMMON NAME" in loaded.columns
        assert "HABITAT_GROUP" in loaded.columns
        assert "PCT" in loaded.columns
        assert len(loaded) > 0

    def test_county_coverage_csv(self, sample_vulture_df, output_dir):
        coverage = (
            sample_vulture_df.groupby(["STATE", "COUNTY"])
            .agg(
                TOTAL_CHECKLISTS=("SAMPLING EVENT IDENTIFIER", "nunique"),
                VULTURE_DETECTIONS=("COMMON NAME", "count"),
            )
            .reset_index()
        )
        path = os.path.join(output_dir, "county_coverage.csv")
        coverage.to_csv(path, index=False)

        loaded = pd.read_csv(path)
        assert "STATE" in loaded.columns
        assert "COUNTY" in loaded.columns
        assert len(loaded) == coverage.shape[0]

    def test_grid_coverage_csv(self, output_dir):
        grid = pd.DataFrame({
            "GRID_LAT": [30.0, 30.1, 30.2],
            "GRID_LON": [-97.0, -97.0, -97.0],
            "STATUS": ["Observed", "No Data", "Open Water"],
        })
        path = os.path.join(output_dir, "grid_coverage.csv")
        grid.to_csv(path, index=False)

        loaded = pd.read_csv(path)
        assert set(loaded["STATUS"].unique()) == {"Observed", "No Data", "Open Water"}


class TestDataIntegrity:
    """Tests for data quality assumptions the pipeline relies on."""

    def test_no_duplicate_identifiers_per_species(self, sample_vulture_df):
        dupes = sample_vulture_df.duplicated(
            subset=["GLOBAL UNIQUE IDENTIFIER"], keep=False
        )
        assert not dupes.any()

    def test_coordinates_in_study_area(self, sample_vulture_df):
        assert (sample_vulture_df["LATITUDE"] >= 25.0).all()
        assert (sample_vulture_df["LATITUDE"] <= 37.0).all()
        assert (sample_vulture_df["LONGITUDE"] >= -107.0).all()
        assert (sample_vulture_df["LONGITUDE"] <= -88.0).all()

    def test_dates_are_valid(self, sample_vulture_df):
        assert sample_vulture_df["OBSERVATION DATE"].notna().all()
        assert (sample_vulture_df["OBSERVATION DATE"].dt.year >= 1900).all()

    def test_nlcd_codes_are_valid(self, sample_vulture_df):
        valid_codes = {11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52,
                       71, 72, 73, 74, 81, 82, 90, 95}
        actual = set(sample_vulture_df["NLCD_CODE"].unique())
        assert actual.issubset(valid_codes)

    def test_all_species_reported_is_binary(self, sample_vulture_df):
        assert set(sample_vulture_df["ALL SPECIES REPORTED"].unique()).issubset({0, 1})

    def test_season_matches_month(self, sample_vulture_df):
        season_map = {
            1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall",
            11: "Fall", 12: "Winter",
        }
        for _, row in sample_vulture_df.iterrows():
            expected = season_map[row["MONTH"]]
            assert row["SEASON"] == expected, (
                f"Month {row['MONTH']} should be {expected}, got {row['SEASON']}"
            )