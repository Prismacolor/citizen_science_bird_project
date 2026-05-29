# Citizen Science Bird Project

## What This Is

Two vulture species — Black Vulture and Turkey Vulture — share the same scavenging job across Texas, Arkansas, Oklahoma, and Louisiana. But they don't use the landscape the same way. This project uses millions of birdwatcher observations from [eBird](https://ebird.org) to figure out where each species hangs out, what kind of habitat they prefer, and whether the landscapes they depend on are changing.

We also discovered something unexpected: a huge number of vulture sightings fall in areas classified as "open water." So we built a second analysis focused on the Gulf Coast to find out what's really going on — and it turns out vultures are using salt marshes, tidal flats, and beaches that standard land cover maps don't capture well.

## Key Findings

**Habitat preferences differ.** Black vultures are more likely to be found in forests in the eastern half of the study region. Turkey Vultures range more widely across open and arid landscapes. They are also seen more in "open water."

**Migratory trends** Analysis of the Gulf Coast land cover data shows a clear migration pattern for turkey vultures during the summer.

**Landscapes are shifting.** Hundreds of thousands of vulture sightings overlap with areas flagged for urban expansion, cropland conversion, and wetland changes.

**Turkey Vulture coastal detection is declining.** Year-over-year effort-corrected rates show a downward trend for Turkey Vultures on the Gulf Coast, even as birder participation increases.

## Data Sources

- **eBird** — Citizen science bird observations ([ebird.org](https://ebird.org))
- **NLCD Land Cover 2024** — 30m habitat classification from USGS ([mrlc.gov](https://www.mrlc.gov))
- **NLCD Strata Map** — Landscape change detection layer from USGS
- **NOAA C-CAP** — 30m coastal land cover from NOAA ([coast.noaa.gov](https://coast.noaa.gov/digitalcoast/data/ccapregional.html))

---

## Project Structure

```
citizen_science_bird_project/
├── data_extraction.py          # Script 1: eBird loading + dual NLCD raster join
├── sightings_analysis.py       # Script 2: habitat, seasonal, county, change analysis
├── visualizations.py           # Script 3: 9 publication-quality charts
├── gulf_coast_analysis.py      # Gulf coastal study (density, seasonal, yearly trends)
├── data/
│   ├── ebird_data_files/       # Raw eBird .txt exports
│   ├── nlcd_landcover_data/    # Annual NLCD Land Cover 2024 TIF
│   ├── nlcd_strata_data/       # NLCD strata change detection TIF
│   └── noaa_ccap_data/         # NOAA C-CAP coastal land cover TIF
├── analysis_results/           
│   └── gulf_coastal/           # Output CSVs from gulf analysis
                                # Output CSVs from main analysis
├── figures/                    
│   └── gulf_coastal/           # Output PNGs from gulf analysis
                                # Output PNGs from main analysis
├── tests/
│   ├── test_bird_project.py    # Tests for main analysis
│   └── test_gulf_coast.py      # Tests for gulf analysis
├── requirements.txt
└── README.md
```

## Setup

**Python 3.12+** required.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Download Raster Data

The eBird text files go in `data/ebird_data_files/`. The three raster files must be downloaded separately:

| File | Source | Destination |
|------|--------|-------------|
| Annual NLCD Land Cover 2024 | [MRLC](https://www.mrlc.gov/data) | `data/nlcd_landcover_data/` |
| NLCD Strata Map | [MRLC](https://www.mrlc.gov/data) | `data/nlcd_strata_data/` |
| C-CAP Regional 2016 | [NOAA](https://chs.coast.noaa.gov/htdata/raster1/landcover/bulkdownload/30m_lc/) | `data/noaa_ccap_data/` |

## Running

Scripts run in order. Each one depends on the output of the previous script.

```bash
# Step 1: Extract eBird data + dual raster join
python data_extraction.py

# Step 2: Statistical analysis (produces CSVs)
python sightings_analysis.py

# Step 3: Visualizations (produces PNGs)
python visualizations.py

# Gulf coastal study (independent of Scripts 2-3, needs Script 1 output)
python gulf_coast_analysis.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Requirements

```
pandas, numpy, dask, rasterio, pyproj, matplotlib, seaborn, tqdm, pytest
```

See `requirements.txt` for pinned versions.

## Limitations

- **Observer bias.** eBird data reflects where birders go, not where birds are. Urban parks and known hotspots are overrepresented. Rural and remote areas may be undersampled.
- **Effort correction helps but isn't perfect.** Detection rate (vulture checklists / total checklists) controls for birder volume but not for differences in observer skill, time of day, or survey method.
- **NLCD coastal gap.** The NLCD raster classifies barrier islands and shorelines as "Open Water." The C-CAP raster fills this gap for the Gulf Coast but only covers through 2016 and only for a certain distance from the shore.
- **Correlation ≠ causation.** Year-over-year trends may reflect habitat changes, population shifts, or changes in birder behavior. This analysis identifies patterns worth investigating, but does not attempt to draw hard conclusions.