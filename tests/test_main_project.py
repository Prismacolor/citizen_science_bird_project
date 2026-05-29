"""
test_main_project.py — Tests for eBird Vulture Analysis
-----------------------------------------------------------
Tests the actual functions in data_extraction and sightings_analysis
by calling them with a sample DataFrame and checking the results.

I/O is mocked:
  - raster_lookup is patched so no TIF files are needed
  - _save is patched so no CSVs are written to disk

Run from project root:  pytest tests/test_main_project.py -v
"""

import importlib
import importlib.util
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ── PATH SETUP ────────────────────────────────────────────────────────────────
# test lives in tests/, scripts live in project root (one level up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


# ── MODULE LOADERS ────────────────────────────────────────────────────────────
# We use spec_from_file_location so we can import files that don't have
# standard module names. These point at the project root, not tests/.

def _load_extraction_module():
    """Import data_extraction.py from the project root."""
    spec = importlib.util.spec_from_file_location(
        "data_extraction",
        os.path.join(PROJECT_ROOT, "data_extraction.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_analysis_module():
    """Import sightings_analysis.py from the project root."""
    spec = importlib.util.spec_from_file_location(
        "sightings_analysis",
        os.path.join(PROJECT_ROOT, "sightings_analysis.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Minimal enriched vulture DataFrame that mimics what Script 1 produces.
    12 rows, 2 species, 4 states, all 4 seasons, mix of stable/changed.
    """
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
        "STATE": [
            "Texas", "Texas", "Arkansas", "Arkansas",
            "Texas", "Arkansas", "Texas", "Arkansas",
            "Texas", "Louisiana", "Texas", "Louisiana",
        ],
        "COUNTY": [
            "Travis", "Harris", "Pulaski", "Shelby",
            "Travis", "Garland", "Travis", "Pulaski",
            "McLennan", "Ouachita", "Harris", "East Baton Rouge",
        ],
        "LOCALITY TYPE": ["H"] * 12,
        "SAMPLING EVENT IDENTIFIER": [f"S{i}" for i in range(12)],
        "ALL SPECIES REPORTED": [1] * 12,
        "DURATION MINUTES": [60, 120, 45, 90, 30, 60, 90, 120, 45, 60, 75, 50],
        "EFFORT DISTANCE KM": [1.5, 3.0, 0.5, 2.0, 1.0, 1.5, 2.0, 3.0, 0.8, 1.2, 1.5, 0.6],
        "NUMBER OBSERVERS": [1, 2, 1, 3, 1, 2, 1, 1, 2, 1, 3, 1],
        "MONTH":  [1, 3, 6, 9, 4, 7, 11, 12, 2, 5, 8, 10],
        "YEAR":   [2023] * 12,
        "SEASON": [
            "Winter", "Spring", "Summer", "Fall",
            "Spring", "Summer", "Fall", "Winter",
            "Winter", "Spring", "Summer", "Fall",
        ],
        # Land cover columns
        "NLCD_CODE":     [23, 41, 82, 41, 21, 52, 23, 82, 11, 71, 42, 90],
        "HABITAT": [
            "Developed - Medium Intensity", "Deciduous Forest", "Cultivated Crops",
            "Deciduous Forest", "Developed - Open Space", "Shrub/Scrub",
            "Developed - Medium Intensity", "Cultivated Crops", "Open Water",
            "Grassland/Herbaceous", "Evergreen Forest", "Woody Wetlands",
        ],
        "HABITAT_GROUP": [
            "Developed: Medium Intensity", "Deciduous Forest", "Cultivated Crops",
            "Deciduous Forest", "Developed: Open Space", "Shrub/Scrub",
            "Developed: Medium Intensity", "Cultivated Crops", "Open Water",
            "Grassland", "Evergreen Forest", "Woody Wetlands",
        ],
        # Strata columns
        "STRATA_CODE":      [23, 41, 101, 104, 21, 52, 100, 82, 11, 71, 103, 90],
        "LANDSCAPE_CHANGE": [False, False, True, True, False, False, True, False,
                             False, False, True, False],
        "CHANGE_TYPE":      [
            "Stable", "Stable", "Cropland Change", "Common Change",
            "Stable", "Stable", "Urban Gain", "Stable",
            "Stable", "Stable", "Rare Change", "Stable",
        ],
    })


# ══════════════════════════════════════════════════════════════════════════════
# SCRIPT 1 TESTS — join_landcover_and_strata
# ══════════════════════════════════════════════════════════════════════════════

class TestJoinLandcoverAndStrata:
    """
    Tests the actual join_landcover_and_strata function from data_extraction.py.
    We mock raster_lookup to return known code arrays, then verify
    the function correctly maps those codes to labels and flags.
    """

    def _make_input_df(self):
        """Bare-minimum df with just the columns the join function reads."""
        return pd.DataFrame({
            "LATITUDE":  [30.2, 34.7, 29.8],
            "LONGITUDE": [-97.7, -92.3, -95.0],
        })

    def test_landcover_columns_added(self):
        """join function adds NLCD_CODE, HABITAT, HABITAT_GROUP from land cover codes."""
        mod = _load_extraction_module()
        input_df = self._make_input_df()
        lc_codes = np.array([41, 82, 11], dtype=np.int16)
        strata_codes = np.array([0, 0, 0], dtype=np.int16)

        with patch.object(mod, "raster_lookup", side_effect=[lc_codes, strata_codes]):
            result = mod.join_landcover_and_strata(input_df)

        assert "NLCD_CODE" in result.columns
        assert "HABITAT" in result.columns
        assert "HABITAT_GROUP" in result.columns
        assert list(result["NLCD_CODE"]) == [41, 82, 11]
        assert result["HABITAT"].iloc[0] == "Deciduous Forest"
        assert result["HABITAT"].iloc[1] == "Cultivated Crops"
        assert result["HABITAT"].iloc[2] == "Open Water"

    def test_strata_columns_added(self):
        """join function adds STRATA_CODE, LANDSCAPE_CHANGE, CHANGE_TYPE from strata codes."""
        mod = _load_extraction_module()
        input_df = self._make_input_df()
        lc_codes = np.array([41, 82, 11], dtype=np.int16)
        strata_codes = np.array([100, 41, 104], dtype=np.int16)

        with patch.object(mod, "raster_lookup", side_effect=[lc_codes, strata_codes]):
            result = mod.join_landcover_and_strata(input_df)

        assert "STRATA_CODE" in result.columns
        assert "LANDSCAPE_CHANGE" in result.columns
        assert "CHANGE_TYPE" in result.columns

        assert result["LANDSCAPE_CHANGE"].iloc[0] == True
        assert result["LANDSCAPE_CHANGE"].iloc[1] == False
        assert result["LANDSCAPE_CHANGE"].iloc[2] == True

        assert result["CHANGE_TYPE"].iloc[0] == "Urban Gain"
        assert result["CHANGE_TYPE"].iloc[1] == "Stable"
        assert result["CHANGE_TYPE"].iloc[2] == "Common Change"

    def test_unknown_landcover_code_becomes_unknown(self):
        """Codes not in NLCD_LABELS should map to 'Unknown' habitat."""
        mod = _load_extraction_module()
        input_df = pd.DataFrame({"LATITUDE": [30.0], "LONGITUDE": [-95.0]})
        lc_codes = np.array([-1], dtype=np.int16)
        strata_codes = np.array([-1], dtype=np.int16)

        with patch.object(mod, "raster_lookup", side_effect=[lc_codes, strata_codes]):
            result = mod.join_landcover_and_strata(input_df)

        assert result["HABITAT"].iloc[0] == "Unknown"
        assert result["HABITAT_GROUP"].iloc[0] == "Other"

    def test_all_change_codes_recognized(self):
        """Every strata code 100-104 maps to the correct CHANGE_TYPE."""
        mod = _load_extraction_module()
        input_df = pd.DataFrame({
            "LATITUDE":  [30.0, 31.0, 32.0, 33.0, 34.0],
            "LONGITUDE": [-95.0, -96.0, -97.0, -94.0, -93.0],
        })
        lc_codes = np.array([41, 41, 41, 41, 41], dtype=np.int16)
        strata_codes = np.array([100, 101, 102, 103, 104], dtype=np.int16)

        with patch.object(mod, "raster_lookup", side_effect=[lc_codes, strata_codes]):
            result = mod.join_landcover_and_strata(input_df)

        expected = ["Urban Gain", "Cropland Change", "Wetland/Water Flux",
                    "Rare Change", "Common Change"]
        assert list(result["CHANGE_TYPE"]) == expected
        assert result["LANDSCAPE_CHANGE"].all()

    def test_input_df_not_mutated(self):
        """join function should .copy() internally, not modify the input."""
        mod = _load_extraction_module()
        input_df = self._make_input_df()
        original_cols = set(input_df.columns)
        lc_codes = np.array([41, 82, 11], dtype=np.int16)
        strata_codes = np.array([0, 0, 0], dtype=np.int16)

        with patch.object(mod, "raster_lookup", side_effect=[lc_codes, strata_codes]):
            mod.join_landcover_and_strata(input_df)

        assert set(input_df.columns) == original_cols


# ══════════════════════════════════════════════════════════════════════════════
# SCRIPT 2 TESTS — Analysis functions from sightings_analysis.py
# ══════════════════════════════════════════════════════════════════════════════

class TestHabitatPercentages:

    def test_water_excluded(self, sample_df):
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            result = mod.habitat_percentages(sample_df)
        assert "Open Water" not in result["HABITAT_GROUP"].values

    def test_percentages_sum_per_species(self, sample_df):
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            result = mod.habitat_percentages(sample_df)
        for species in result["COMMON NAME"].unique():
            total = result[result["COMMON NAME"] == species]["PCT"].sum()
            assert total == pytest.approx(100.0, abs=0.5)

    def test_output_has_required_columns(self, sample_df):
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            result = mod.habitat_percentages(sample_df)
        assert set(result.columns) == {"COMMON NAME", "HABITAT_GROUP", "COUNT", "PCT"}

    def test_both_species_present(self, sample_df):
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            result = mod.habitat_percentages(sample_df)
        assert set(result["COMMON NAME"].unique()) == {"Black Vulture", "Turkey Vulture"}


class TestHabitatBySeason:

    def test_water_excluded(self, sample_df):
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            result = mod.habitat_by_season(sample_df)
        assert "Open Water" not in result["HABITAT_GROUP"].values

    def test_percentages_sum_per_species_season(self, sample_df):
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            result = mod.habitat_by_season(sample_df)
        for (species, season), grp in result.groupby(["COMMON NAME", "SEASON"]):
            total = grp["PCT"].sum()
            assert total == pytest.approx(100.0, abs=0.5)

    def test_seasons_are_ordered(self, sample_df):
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            result = mod.habitat_by_season(sample_df)
        assert result["SEASON"].dtype.name == "category"


class TestSeasonalDetectionRates:

    def test_rates_between_0_and_100(self, sample_df):
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            result = mod.seasonal_detection_rates(sample_df)
        assert (result["DETECTION_RATE"] >= 0).all()
        assert (result["DETECTION_RATE"] <= 100).all()

    def test_output_columns(self, sample_df):
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            result = mod.seasonal_detection_rates(sample_df)
        required = {"COMMON NAME", "MONTH", "DETECTED_CHECKLISTS",
                     "TOTAL_CHECKLISTS", "DETECTION_RATE", "MONTH_NAME", "SEASON"}
        assert required.issubset(set(result.columns))

    def test_detection_rate_calculation(self, sample_df):
        """Each month has 1 checklist total, detection rate should be 100%."""
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            result = mod.seasonal_detection_rates(sample_df)
        assert (result["DETECTION_RATE"] == 100.0).all()


class TestCountyCoverage:

    def test_returns_two_dataframes(self, sample_df):
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            coverage, underbirded = mod.county_coverage(sample_df)
        assert isinstance(coverage, pd.DataFrame)
        assert isinstance(underbirded, pd.DataFrame)

    def test_all_counties_in_coverage(self, sample_df):
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            coverage, _ = mod.county_coverage(sample_df)
        input_counties = set(sample_df[["STATE", "COUNTY"]].apply(tuple, axis=1))
        result_counties = set(coverage[["STATE", "COUNTY"]].apply(tuple, axis=1))
        assert input_counties == result_counties

    def test_underbirded_threshold(self, sample_df):
        """With 12 rows spread across many counties, all should be underbirded."""
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            coverage, underbirded = mod.county_coverage(sample_df)
        assert len(underbirded) == len(coverage)

    def test_detection_rate_column(self, sample_df):
        mod = _load_analysis_module()
        with patch.object(mod, "_save", return_value="fake_path"):
            coverage, _ = mod.county_coverage(sample_df)
        assert "DETECTION_RATE" in coverage.columns
        assert (coverage["DETECTION_RATE"] >= 0).all()


class TestLandscapeChangeAnalysis:
    """
    Tests landscape_change_analysis(). This function returns None and
    saves 4 CSVs via _save. We capture what _save was called with.
    """

    def _run_with_capture(self, mod, df):
        """Run landscape_change_analysis and capture all saved DataFrames."""
        saved = {}
        def fake_save(df, name):
            saved[name] = df.copy()
            return f"fake/{name}"
        with patch.object(mod, "_save", side_effect=fake_save):
            mod.landscape_change_analysis(df)
        return saved

    def test_summary_counts_match(self, sample_df):
        mod = _load_analysis_module()
        saved = self._run_with_capture(mod, sample_df)
        summary = saved["landscape_change.csv"]
        assert len(summary) == 2
        assert summary["COUNT"].sum() == len(sample_df)

    def test_summary_percentages_sum_to_100(self, sample_df):
        mod = _load_analysis_module()
        saved = self._run_with_capture(mod, sample_df)
        summary = saved["landscape_change.csv"]
        assert summary["PCT"].sum() == pytest.approx(100.0, abs=0.1)

    def test_change_by_species_percentages(self, sample_df):
        mod = _load_analysis_module()
        saved = self._run_with_capture(mod, sample_df)
        by_species = saved["change_by_species.csv"]
        for species in by_species["COMMON NAME"].unique():
            total = by_species[by_species["COMMON NAME"] == species]["PCT"].sum()
            assert total == pytest.approx(100.0, abs=0.5)

    def test_change_by_species_has_both_species(self, sample_df):
        mod = _load_analysis_module()
        saved = self._run_with_capture(mod, sample_df)
        by_species = saved["change_by_species.csv"]
        assert set(by_species["COMMON NAME"].unique()) == {"Black Vulture", "Turkey Vulture"}

    def test_change_by_habitat_only_changed_rows(self, sample_df):
        """change_by_habitat should only include rows where LANDSCAPE_CHANGE is True."""
        mod = _load_analysis_module()
        saved = self._run_with_capture(mod, sample_df)
        by_habitat = saved["change_by_habitat.csv"]
        changed_count = int(sample_df["LANDSCAPE_CHANGE"].sum())
        assert by_habitat["COUNT"].sum() == changed_count

    def test_change_by_habitat_no_stable(self, sample_df):
        """change_by_habitat should not have 'Stable' as a CHANGE_TYPE."""
        mod = _load_analysis_module()
        saved = self._run_with_capture(mod, sample_df)
        by_habitat = saved["change_by_habitat.csv"]
        assert "Stable" not in by_habitat["CHANGE_TYPE"].values

    def test_all_stable_input(self):
        """When no rows have LANDSCAPE_CHANGE=True, change_by_habitat should not be saved."""
        mod = _load_analysis_module()
        all_stable_df = pd.DataFrame({
            "COMMON NAME": ["Black Vulture", "Turkey Vulture"],
            "LANDSCAPE_CHANGE": [False, False],
            "CHANGE_TYPE": ["Stable", "Stable"],
            "HABITAT_GROUP": ["Deciduous Forest", "Grassland"],
            "SEASON": ["Spring", "Summer"],
        })
        saved = self._run_with_capture(mod, all_stable_df)
        assert "landscape_change.csv" in saved
        assert "change_by_habitat.csv" not in saved
        summary = saved["landscape_change.csv"]
        changed_row = summary[summary["STATUS"] == "Changed Landscape"]
        assert int(changed_row["COUNT"].iloc[0]) == 0

    def test_saves_correct_number_of_csvs(self, sample_df):
        """With change data present, should save 4 CSVs."""
        mod = _load_analysis_module()
        saved = self._run_with_capture(mod, sample_df)
        expected = {"landscape_change.csv", "change_by_species.csv",
                    "change_by_habitat.csv"}
        assert set(saved.keys()) == expected