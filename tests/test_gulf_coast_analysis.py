"""
test_gulf_coast.py — Tests for Gulf Coast Vulture Analysis
-----------------------------------------------------------------
Tests the actual functions in gulf_coast_analysis.py by calling
them with sample data and mocking I/O (raster lookups, CSV saves,
figure saves).

Run from project root:  pytest tests/test_gulf_coast.py -v
"""

import importlib
import importlib.util
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# PATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


def _load_gulf_module():
    """Import gulf_coast_analysis.py from the project root."""
    spec = importlib.util.spec_from_file_location(
        "gulf_coast_analysis",
        os.path.join(PROJECT_ROOT, "gulf_coast_analysis.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# set up
@pytest.fixture
def gulf_df() -> pd.DataFrame:
    """
    Sample Gulf sightings DataFrame — 10 rows within the Gulf bounding box,
    mix of species, months, years, and pre-assigned CCAP codes.
    """
    return pd.DataFrame({
        "GLOBAL UNIQUE IDENTIFIER": [f"URN:G:{i}" for i in range(10)],
        "COMMON NAME": [
            "Black Vulture", "Turkey Vulture", "Black Vulture", "Turkey Vulture",
            "Black Vulture", "Turkey Vulture", "Black Vulture", "Turkey Vulture",
            "Black Vulture", "Turkey Vulture",
        ],
        "LATITUDE":  [29.3, 29.5, 26.1, 29.8, 27.8, 29.3, 28.5, 30.0, 26.5, 29.1],
        "LONGITUDE": [-94.8, -93.5, -97.2, -90.5, -96.5, -94.8, -97.0, -89.5, -97.5, -92.0],
        "OBSERVATION DATE": pd.to_datetime([
            "2018-01-15", "2018-06-10", "2020-03-20", "2020-09-05",
            "2022-04-12", "2022-07-22", "2023-11-30", "2023-12-25",
            "2019-08-09", "2021-10-31",
        ]),
        "STATE": ["Texas"] * 5 + ["Louisiana"] * 2 + ["Louisiana", "Texas", "Louisiana"],
        "COUNTY": [
            "Galveston", "Cameron", "Cameron", "Orleans", "Nueces",
            "Galveston", "Nueces", "St. Tammany", "Cameron", "Terrebonne",
        ],
        "SAMPLING EVENT IDENTIFIER": [f"SG{i}" for i in range(10)],
        "ALL SPECIES REPORTED": [1] * 10,
        "MONTH": [1, 6, 3, 9, 4, 7, 11, 12, 8, 10],
        "YEAR": [2018, 2018, 2020, 2020, 2022, 2022, 2023, 2023, 2019, 2021],
        "SEASON": [
            "Winter", "Summer", "Spring", "Fall", "Spring",
            "Summer", "Fall", "Winter", "Summer", "Fall",
        ],
        "NLCD_CODE": [11, 11, 21, 11, 11, 11, 23, 11, 11, 11],
        "HABITAT": ["Open Water"] * 10,
        "HABITAT_GROUP": ["Open Water"] * 10,
    })


@pytest.fixture
def add_gulf_df(gulf_df) -> pd.DataFrame:
    """Gulf df with C-CAP columns already added."""
    df = gulf_df.copy()
    df["CCAP_CODE"] = [19, 18, 5, 21, 19, 18, 4, 15, 19, 21]
    df["CCAP_HABITAT"] = df["CCAP_CODE"].map(_load_gulf_module().CCAP_LABELS)
    df["CCAP_GROUP"] = df["CCAP_HABITAT"].map(_load_gulf_module().CCAP_GROUPS)
    return df


# TESTS: load_gulf_sightings
class TestLoadGulfSightings:

    def test_filters_to_bounding_box(self, tmp_path):
        """Sightings outside the Gulf bbox should be excluded."""
        mod = _load_gulf_module()
        df = pd.DataFrame({
            "COMMON NAME": ["Black Vulture", "Turkey Vulture", "Black Vulture"],
            "LATITUDE":  [29.5, 35.0, 27.0],    # in bbox, too far north, in bbox
            "LONGITUDE": [-94.0, -95.0, -96.0],
            "OBSERVATION DATE": pd.to_datetime(["2023-01-01"] * 3),
            "SAMPLING EVENT IDENTIFIER": ["S1", "S2", "S3"],
        })
        csv_path = str(tmp_path / "test_sightings.csv")
        df.to_csv(csv_path, index=False)

        result = mod.load_gulf_sightings(csv_path)
        assert len(result) == 2  # row at lat 35.0 excluded

    def test_excludes_west_of_bbox(self, tmp_path):
        """Sightings west of the Gulf bbox (inland TX) should be excluded."""
        mod = _load_gulf_module()
        df = pd.DataFrame({
            "COMMON NAME": ["Black Vulture", "Black Vulture"],
            "LATITUDE":  [29.5, 29.5],
            "LONGITUDE": [-94.0, -105.0],  # in bbox, too far west
            "OBSERVATION DATE": pd.to_datetime(["2023-01-01"] * 2),
            "SAMPLING EVENT IDENTIFIER": ["S1", "S2"],
        })
        csv_path = str(tmp_path / "test_sightings.csv")
        df.to_csv(csv_path, index=False)

        result = mod.load_gulf_sightings(csv_path)
        assert len(result) == 1


# TESTS: add_ccap
class TestAddCCAP:

    def test_adds_ccap_columns(self, gulf_df):
        """add_ccap should add CCAP_CODE, CCAP_HABITAT, CCAP_GROUP."""
        mod = _load_gulf_module()
        # Mock the raster lookup to return known codes
        ccap_codes = np.array([19, 18, 5, 21, 19, 18, 4, 15, 19, 21], dtype=np.int16)

        with patch.object(mod, "ccap_raster_lookup", return_value=ccap_codes):
            result = mod.add_ccap(gulf_df)

        assert "CCAP_CODE" in result.columns
        assert "CCAP_HABITAT" in result.columns
        assert "CCAP_GROUP" in result.columns

    def test_maps_codes_to_correct_labels(self, gulf_df):
        """Code 19 should map to Unconsolidated Shore / Beach/Tidal Flat."""
        mod = _load_gulf_module()
        ccap_codes = np.array([19] * len(gulf_df), dtype=np.int16)

        with patch.object(mod, "ccap_raster_lookup", return_value=ccap_codes):
            result = mod.add_ccap(gulf_df)

        assert (result["CCAP_HABITAT"] == "Unconsolidated Shore").all()
        assert (result["CCAP_GROUP"] == "Beach/Tidal Flat").all()

    def test_estuarine_mapped_to_salt_marsh(self, gulf_df):
        """Estuarine wetland codes should group to Salt Marsh/Estuarine."""
        mod = _load_gulf_module()
        ccap_codes = np.array([18] * len(gulf_df), dtype=np.int16)

        with patch.object(mod, "ccap_raster_lookup", return_value=ccap_codes):
            result = mod.add_ccap(gulf_df)

        assert (result["CCAP_GROUP"] == "Salt Marsh/Estuarine").all()

    def test_unknown_code_handled(self, gulf_df):
        """Out-of-bounds codes should map to Unknown/Other."""
        mod = _load_gulf_module()
        ccap_codes = np.array([-1] * len(gulf_df), dtype=np.int16)

        with patch.object(mod, "ccap_raster_lookup", return_value=ccap_codes):
            result = mod.add_ccap(gulf_df)

        assert (result["CCAP_HABITAT"] == "Unknown").all()
        assert (result["CCAP_GROUP"] == "Other").all()

    def test_input_not_mutated(self, gulf_df):
        """add_ccap should .copy() and not modify the input."""
        mod = _load_gulf_module()
        original_cols = set(gulf_df.columns)
        ccap_codes = np.array([21] * len(gulf_df), dtype=np.int16)

        with patch.object(mod, "ccap_raster_lookup", return_value=ccap_codes):
            mod.add_ccap(gulf_df)

        assert set(gulf_df.columns) == original_cols


# TESTS: gulf_density_analysis
class TestGulfDensityAnalysis:

    def test_habitat_breakdown_output(self, add_gulf_df):
        """Should produce a habitat breakdown CSV with correct columns."""
        mod = _load_gulf_module()
        with patch.object(mod, "_save_csv", return_value="fake") as mock_csv, \
             patch.object(mod, "_save_fig", return_value="fake"):
            result = mod.gulf_density_analysis(add_gulf_df)

        assert set(result.columns) == {"COMMON NAME", "CCAP_GROUP", "COUNT", "PCT"}

    def test_percentages_sum_per_species(self, add_gulf_df):
        """Habitat percentages should sum to ~100% per species."""
        mod = _load_gulf_module()
        with patch.object(mod, "_save_csv", return_value="fake"), \
             patch.object(mod, "_save_fig", return_value="fake"):
            result = mod.gulf_density_analysis(add_gulf_df)

        for species in result["COMMON NAME"].unique():
            total = result[result["COMMON NAME"] == species]["PCT"].sum()
            assert total == pytest.approx(100.0, abs=0.5)

    def test_both_species_present(self, add_gulf_df):
        """Both species should appear in the output."""
        mod = _load_gulf_module()
        with patch.object(mod, "_save_csv", return_value="fake"), \
             patch.object(mod, "_save_fig", return_value="fake"):
            result = mod.gulf_density_analysis(add_gulf_df)

        assert set(result["COMMON NAME"].unique()) == {"Black Vulture", "Turkey Vulture"}


# TESTS: seasonal_analysis
class TestSeasonalAnalysis:

    def test_output_columns(self, gulf_df):
        """Should have the expected columns."""
        mod = _load_gulf_module()
        with patch.object(mod, "_save_csv", return_value="fake"), \
             patch.object(mod, "_save_fig", return_value="fake"):
            result = mod.seasonal_analysis(gulf_df)

        required = {"COMMON NAME", "MONTH", "DETECTED_CHECKLISTS",
                     "TOTAL_CHECKLISTS", "DETECTION_RATE", "MONTH_NAME"}
        assert required.issubset(set(result.columns))


# TESTS: yearly_trend_analysis
class TestYearlyTrendAnalysis:

    def test_rates_between_0_and_100(self, gulf_df):
        """Detection rates should be between 0 and 100."""
        mod = _load_gulf_module()
        with patch.object(mod, "_save_csv", return_value="fake"), \
             patch.object(mod, "_save_fig", return_value="fake"):
            result = mod.yearly_trend_analysis(gulf_df)

        assert (result["DETECTION_RATE"] >= 0).all()
        assert (result["DETECTION_RATE"] <= 100).all()

    def test_output_columns(self, gulf_df):
        """Should have the expected columns."""
        mod = _load_gulf_module()
        with patch.object(mod, "_save_csv", return_value="fake"), \
             patch.object(mod, "_save_fig", return_value="fake"):
            result = mod.yearly_trend_analysis(gulf_df)

        required = {"COMMON NAME", "YEAR", "DETECTED_CHECKLISTS",
                     "TOTAL_CHECKLISTS", "DETECTION_RATE"}
        assert required.issubset(set(result.columns))

    def test_years_span_data_range(self, gulf_df):
        """Output years should match what's in the input data."""
        mod = _load_gulf_module()
        with patch.object(mod, "_save_csv", return_value="fake"), \
             patch.object(mod, "_save_fig", return_value="fake"):
            result = mod.yearly_trend_analysis(gulf_df)

        input_years = set(gulf_df["OBSERVATION DATE"].dt.year.unique())
        output_years = set(result["YEAR"].unique())
        assert output_years == input_years

    def test_detection_rate_calculation(self, gulf_df):
        """
        In sample data, each year has unique checklists. 2018 has 2 checklists,
        1 per species, so each species should have 50% detection rate for 2018.
        """
        mod = _load_gulf_module()
        with patch.object(mod, "_save_csv", return_value="fake"), \
             patch.object(mod, "_save_fig", return_value="fake"):
            result = mod.yearly_trend_analysis(gulf_df)

        year_2018 = result[result["YEAR"] == 2018]
        # 2018 has 2 total checklists, each species detected once
        for _, row in year_2018.iterrows():
            assert row["DETECTED_CHECKLISTS"] == 1
            assert row["TOTAL_CHECKLISTS"] == 2
            assert row["DETECTION_RATE"] == 50.0

    def test_both_species_present(self, gulf_df):
        """Both species should appear in the yearly output."""
        mod = _load_gulf_module()
        with patch.object(mod, "_save_csv", return_value="fake"), \
             patch.object(mod, "_save_fig", return_value="fake"):
            result = mod.yearly_trend_analysis(gulf_df)

        assert set(result["COMMON NAME"].unique()) == {"Black Vulture", "Turkey Vulture"}