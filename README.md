# NYC: Does Construction Activity Travel with Street Danger?

Data Engineering and Visualization, Milestone 1 (GIU, Winter 2026). Team of three.

An interactive Dash website and a Jupyter notebook that integrate **DOB NOW: Build job filings** (housing & development) with **NYPD Motor Vehicle Collisions** (transportation) at **ZIP code x month** level, to ask whether the geography and timing of construction overlap with crash harm.

- **Live site:** https://nyc-housing-transport.onrender.com/
- **Housing dataset (H5):** https://data.cityofnewyork.us/d/w9ak-ipjd  (`w9ak-ipjd`)
- **Transportation dataset:** https://data.cityofnewyork.us/d/h9gi-nx95  (`h9gi-nx95`, off-catalog, feasibility check in the notebook)

## Repository layout
| File | Purpose |
|---|---|
| `nyc_housing_transport.ipynb` | Full workflow: overview, EDA, pre-integration cleaning, integration, post-integration cleaning, story |
| `pipeline.py` | Same cleaning/integration code as the notebook (the notebook is generated from it); `python pipeline.py` rebuilds the data |
| `app.py` | Dash website (dropdown filters, search mode, Generate Report, 8 research-question charts, story section) |
| `data/integrated_zip_month.csv` | Integrated dataset read by the website (created by the notebook/pipeline; **commit it**) |
| `assets/style.css` | Website styling |
| `requirements.txt`, `Procfile`, `render.yaml` | Deployment |

## Setup (run in this order)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Build the integrated dataset (downloads from NYC Open Data, cached in data/raw/; a few minutes)
python pipeline.py          # or run every cell of nyc_housing_transport.ipynb

# 2. Run the website
python app.py               # open http://localhost:8050
```
The website needs `data/integrated_zip_month.csv`. Without it the page shows a "Data file not found" message.

## Using the website
1. Pick filters: **Borough**, **Year**, **Housing metric**, **Crash metric** (leave borough/year empty for all).
2. Or type into **search mode**, e.g. `Brooklyn 2022 new buildings pedestrians`, `Queens 2021-2023 demolitions fatal`. Recognised: boroughs, years/ranges, construction types, crash types. Search sets the dropdowns and refreshes the report.
3. Press **Generate Report**. All KPI cards, the eight charts and the auto-written findings update from the current filters.

## Deployment on Render (free)
1. Push this folder to a GitHub repository, **including** `data/integrated_zip_month.csv`.
2. On https://render.com choose **New + > Web Service**, connect the repository.
3. Runtime Python; build command `pip install -r requirements.txt`; start command `gunicorn app:server` (`render.yaml` already contains these).
4. Deploy, open the URL, and test dropdowns, search and Generate Report before submitting. The free tier sleeps when idle, so the first load can take ~30-60 s.

## Integration summary
Both datasets are aggregated to ZIP x month (no row-level join). ZIP universe = ZIPs in both datasets, full ZIP x month grid, left joins, join-introduced NaNs become true zeros. Details and alternatives considered are in the notebook.

## Team contributions
| Member | ID | Contributions |
|---|---|---|
| Mourad Gehad | 16002563 | Q1 (ZIP-level scatter), Q2 (borough ratio bars) and Q8 (construction-growth box plot); ran the notebook on NYC Open Data; set up the GitHub repository; deployed and tested the website on Render; wrote the README and notebook documentation |
| Kareem Mahmoud | 16009517 | Q3 (seasonality line chart), Q4 (hot-spot map) and Q7 (lead/lag correlation); interpreted findings; tested website filters, search mode and Generate Report |
| Mohamed Khalil | 13001416 | Q5 (filing-type heatmap) and Q6 (pedestrian/cyclist injury-share chart), added to both the notebook and the website; interpretation of these findings |

## Limitations
Filings are paperwork events, not finished construction; DOB NOW omits some job types; no population or traffic-volume normalisation; COVID-19 distorts 2020; ZIP-level aggregation hides street-level effects; association is not causation.
