"""
NYC: Does construction activity travel with street danger?
Interactive Dash website. Reads data/integrated_zip_month.csv (created by pipeline.py).

Run locally:  python app.py          Deploy: gunicorn app:server  (see README)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update

DATA_PATH = Path(__file__).parent / "data" / "integrated_zip_month.csv"

# ----------------------------------------------------------------------------------------
# Metric catalogues: what each dropdown option means in the integrated table
# ----------------------------------------------------------------------------------------
HOUSING_OPTIONS = {
    "filings_total": "All construction filings",
    "filings_nb": "New-building filings",
    "filings_alt": "Alteration filings",
    "filings_dm": "Demolition filings",
    "new_units": "New dwelling units (new buildings)",
}
CRASH_OPTIONS = {
    "crashes": "All crashes",
    "injured": "People injured",
    "killed": "People killed",
    "ped_injured": "Pedestrians injured",
    "cyc_injured": "Cyclists injured",
    "mot_injured": "Motorists injured",
}
BOROUGHS = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
BOROUGH_COLORS = {"BRONX": "#e4572e", "BROOKLYN": "#2e86ab", "MANHATTAN": "#7b2d8e",
                  "QUEENS": "#3a9d5d", "STATEN ISLAND": "#f2a93b"}
PRIMARY, SECONDARY = "#1f4e79", "#d1495b"


# ----------------------------------------------------------------------------------------
# Search mode: turn free text such as "Brooklyn 2022 new buildings pedestrians" into filters
# ----------------------------------------------------------------------------------------
BOROUGH_ALIASES = {
    "BRONX": ["bronx"], "BROOKLYN": ["brooklyn", "bklyn"], "MANHATTAN": ["manhattan"],
    "QUEENS": ["queens"], "STATEN ISLAND": ["staten island", "staten"],
}
HOUSING_KEYWORDS = [  # order matters: first match wins
    ("filings_dm", ["demolition", "demolitions", "demolish", "teardown"]),
    ("new_units", ["units", "dwelling units", "apartments"]),
    ("filings_nb", ["new building", "new buildings", "new construction", "construction of new"]),
    ("filings_alt", ["alteration", "alterations", "renovation", "renovations", "remodel"]),
    ("filings_total", ["all filings", "all construction", "filings", "permits"]),
]
CRASH_KEYWORDS = [
    ("killed", ["fatal", "killed", "deaths", "deadly"]),
    ("ped_injured", ["pedestrian", "pedestrians", "walkers"]),
    ("cyc_injured", ["cyclist", "cyclists", "bike", "bikes", "bicycle"]),
    ("mot_injured", ["motorist", "motorists", "drivers", "driver"]),
    ("injured", ["injury", "injuries", "injured", "hurt"]),
    ("crashes", ["crash", "crashes", "collision", "collisions", "accidents"]),
]


def parse_search_query(text, available_years):
    """Extract filters from a free-text query.

    Returns a dict that only contains the filters the text actually mentions, e.g.
    {'boroughs': ['BROOKLYN'], 'years': [2022], 'housing': 'filings_nb'}.
    """
    cleaned = re.sub(r"[^a-z0-9 -]", " ", (text or "").lower())
    query = f" {cleaned} "
    found = {}

    boroughs = [b for b, aliases in BOROUGH_ALIASES.items()
                if any(f" {alias} " in query for alias in aliases)]
    if boroughs:
        found["boroughs"] = boroughs

    years = set()
    for start, end in re.findall(r"\b(20\d{2})\s*-\s*(20\d{2})\b", query):  # ranges: 2021-2023
        years.update(range(int(start), int(end) + 1))
    years.update(int(y) for y in re.findall(r"\b(20\d{2})\b", query))
    years &= set(available_years)
    if years:
        found["years"] = sorted(years)

    for key, keywords in HOUSING_KEYWORDS:
        if any(f" {kw} " in query for kw in keywords):
            found["housing"] = key
            break
    for key, keywords in CRASH_KEYWORDS:
        if any(f" {kw} " in query for kw in keywords):
            found["crash"] = key
            break
    return found


# ----------------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------------
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, dtype={"zip": str}, parse_dates=["month"])
    return df


DF = load_data()
YEARS = sorted(DF["year"].unique().tolist()) if DF is not None else []


def filter_data(df, boroughs, years):
    """Empty selection = no restriction."""
    out = df
    if boroughs:
        out = out[out["borough"].isin(boroughs)]
    if years:
        out = out[out["year"].isin(years)]
    return out


def spearman(x, y):
    """Spearman rank correlation (robust to the heavy right-skew of count data)."""
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(x.rank().corr(y.rank()))


def fmt_rho(rho):
    return "n/a" if pd.isna(rho) else f"{rho:+.2f}"


# ----------------------------------------------------------------------------------------
# Figures. Each answers one research question.
# ----------------------------------------------------------------------------------------
def empty_figure(message="No data for this selection - widen the filters"):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 16, "color": "#666"})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=420, template="plotly_white")
    return fig


def zip_totals(data, h_col, c_col):
    """One row per ZIP: totals over the filtered period (the aggregation used by Q1 and Q4)."""
    totals = data.groupby("zip").agg(
        borough=("borough", "first"), zip_lat=("zip_lat", "first"), zip_lon=("zip_lon", "first"),
        housing=(h_col, "sum"), crash=(c_col, "sum"), crashes=("crashes", "sum")).reset_index()
    return totals


def fig_q1_scatter(data, h_col, c_col):
    """Q1 - Do ZIPs with more construction activity also have more crash harm?"""
    totals = zip_totals(data, h_col, c_col)
    if totals.empty:
        return empty_figure()
    rho = spearman(totals["housing"], totals["crash"])
    fig = px.scatter(
        totals, x="housing", y="crash", size="crashes", color="borough", hover_name="zip",
        color_discrete_map=BOROUGH_COLORS, size_max=28,
        labels={"housing": HOUSING_OPTIONS[h_col], "crash": CRASH_OPTIONS[c_col],
                "crashes": "All crashes", "borough": "Borough"})
    if len(totals) >= 3 and totals["housing"].nunique() > 1:  # least-squares trend line
        slope, intercept = np.polyfit(totals["housing"], totals["crash"], 1)
        xs = np.array([totals["housing"].min(), totals["housing"].max()])
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Linear trend",
                                 line={"color": "#444", "dash": "dash"}))
    fig.update_layout(title=f"Each dot is a ZIP code (Spearman rho = {fmt_rho(rho)})",
                      template="plotly_white", height=460, legend_title_text="")
    return fig


def fig_q2_ratio_bars(data, h_col, c_col):
    """Q2 - Which boroughs carry the most crash harm per unit of construction activity?"""
    grouped = data.groupby(["borough", "year"])[[h_col, c_col]].sum().reset_index()
    grouped = grouped[grouped[h_col] > 0]
    if grouped.empty:
        return empty_figure("No construction activity in this selection")
    grouped["ratio"] = grouped[c_col] / grouped[h_col] * 100
    grouped["year"] = grouped["year"].astype(str)
    fig = px.bar(grouped, x="borough", y="ratio", color="year", barmode="group",
                 category_orders={"borough": BOROUGHS},
                 labels={"ratio": f"{CRASH_OPTIONS[c_col]} per 100 x {HOUSING_OPTIONS[h_col].lower()}",
                         "borough": "", "year": "Year"},
                 color_discrete_sequence=px.colors.sequential.Blues_r[:7])
    fig.update_layout(title="Crash harm per 100 units of construction activity",
                      template="plotly_white", height=460)
    return fig


def fig_q3_seasonality(data, h_col, c_col):
    """Q3 - Do construction filings and crashes move together through the year?"""
    monthly = data.groupby("month")[[h_col, c_col]].sum().reset_index()
    if monthly.empty:
        return empty_figure()
    rho = spearman(monthly[h_col], monthly[c_col])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly[h_col], name=HOUSING_OPTIONS[h_col],
                             line={"color": PRIMARY, "width": 3}))
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly[c_col], name=CRASH_OPTIONS[c_col],
                             line={"color": SECONDARY, "width": 3}, yaxis="y2"))
    fig.update_layout(
        title=f"Monthly totals, two scales (Spearman rho = {fmt_rho(rho)})",
        template="plotly_white", height=460, hovermode="x unified",
        yaxis={"title": HOUSING_OPTIONS[h_col], "color": PRIMARY},
        yaxis2={"title": CRASH_OPTIONS[c_col], "color": SECONDARY, "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": -0.2})
    return fig


def fig_q4_map(data, h_col, c_col):
    """Q4 - Where do construction hot spots and crash hot spots overlap? (cross-domain map)"""
    totals = zip_totals(data, h_col, c_col)
    if totals.empty:
        return empty_figure()
    h_cut, c_cut = totals["housing"].quantile(0.75), totals["crash"].quantile(0.75)
    totals["hotspot"] = np.where((totals["housing"] >= h_cut) & (totals["crash"] >= c_cut),
                                 "High construction AND high crash harm", "Other ZIPs")
    totals["size_value"] = totals["housing"].clip(lower=0) + 1  # +1 so zero-construction ZIPs stay visible
    map_fn = getattr(px, "scatter_map", None) or px.scatter_mapbox
    style_kw = {"map_style": "carto-positron"} if hasattr(px, "scatter_map") else {"mapbox_style": "carto-positron"}
    fig = map_fn(
        totals, lat="zip_lat", lon="zip_lon", size="size_value", color="crash", size_max=30,
        hover_name="zip", hover_data={"borough": True, "housing": True, "crash": True,
                                      "size_value": False, "zip_lat": False, "zip_lon": False},
        color_continuous_scale="YlOrRd", zoom=9.2, center={"lat": 40.72, "lon": -73.95},
        labels={"crash": CRASH_OPTIONS[c_col], "housing": HOUSING_OPTIONS[h_col]}, **style_kw)
    hot = totals[totals["hotspot"].str.startswith("High")]
    if not hot.empty:  # ring the overlap ZIPs
        ring_kw = dict(lat=hot["zip_lat"], lon=hot["zip_lon"], mode="markers", name="Overlap hot spot",
                       text=hot["zip"], hoverinfo="text",
                       marker={"size": 40, "color": "#111", "opacity": 0.25})
        ring_trace = go.Scattermap if hasattr(go, "Scattermap") else go.Scattermapbox
        fig.add_trace(ring_trace(**ring_kw))
    fig.update_layout(title="Bubble size = construction activity, colour = crash harm "
                            "(dark halo = top quartile on both)",
                      height=520, margin={"l": 0, "r": 0, "t": 50, "b": 0})
    return fig


def lead_lag_correlations(data, h_col, c_col, max_lag=6):
    """Pooled within-ZIP correlation between construction in month t and crash harm in month t+k.

    Each ZIP's two series are z-scored first, so the result reflects 'above/below this ZIP's
    own normal' rather than big-ZIP-versus-small-ZIP differences.
    """
    h = data.pivot_table(index="month", columns="zip", values=h_col, aggfunc="sum")
    c = data.pivot_table(index="month", columns="zip", values=c_col, aggfunc="sum")
    zscore = lambda frame: (frame - frame.mean()) / frame.std(ddof=0).replace(0, np.nan)
    hz, cz = zscore(h), zscore(c)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = cz.shift(-lag)  # crash value `lag` months after the construction month
        pair = pd.DataFrame({"h": hz.to_numpy().ravel(), "c": shifted.to_numpy().ravel()}).dropna()
        rows.append({"lag": lag, "corr": pair["h"].corr(pair["c"]) if len(pair) > 10 else np.nan})
    return pd.DataFrame(rows)


def fig_q5_lead_lag(data, h_col, c_col):
    """Q5 (bonus) - Does a construction spike come before or after a crash spike?"""
    if data["zip"].nunique() < 2 or data["month"].nunique() < 12:
        return empty_figure("Need at least 2 ZIPs and 12 months for a lead/lag analysis")
    lag_df = lead_lag_correlations(data, h_col, c_col)
    fig = px.bar(lag_df, x="lag", y="corr",
                 labels={"lag": "Months between construction month and crash month "
                                "(positive = crashes measured later)", "corr": "Correlation"},
                 color="corr", color_continuous_scale="RdBu_r", range_color=[-0.5, 0.5])
    fig.update_layout(title="Within-ZIP lead/lag correlation (association only, not cause)",
                      template="plotly_white", height=420, coloraxis_showscale=False)
    return fig


# ----------------------------------------------------------------------------------------
# Report assembly (plain function so it is easy to test; the Dash callback just calls it)
# ----------------------------------------------------------------------------------------
def kpi_card(title, value):
    return html.Div([html.Div(value, className="kpi-value"), html.Div(title, className="kpi-title")],
                    className="kpi-card")


def build_report(boroughs, years, h_col, c_col):
    h_col = h_col if h_col in HOUSING_OPTIONS else "filings_total"
    c_col = c_col if c_col in CRASH_OPTIONS else "crashes"
    data = filter_data(DF, boroughs, years)
    if data.empty:
        msg = html.Div("No rows match these filters.", className="finding")
        return [msg] + [empty_figure()] * 5 + [msg]

    totals = zip_totals(data, h_col, c_col)
    rho = spearman(totals["housing"], totals["crash"])
    kpis = [
        kpi_card("ZIP codes", f"{data['zip'].nunique():,}"),
        kpi_card(HOUSING_OPTIONS[h_col], f"{int(data[h_col].sum()):,}"),
        kpi_card(CRASH_OPTIONS[c_col], f"{int(data[c_col].sum()):,}"),
        kpi_card("ZIP-level Spearman rho", fmt_rho(rho)),
    ]

    # Auto-written findings: computed from the current selection, so they are never stale
    strength = ("no clear" if pd.isna(rho) or abs(rho) < 0.2 else
                "a weak" if abs(rho) < 0.4 else "a moderate" if abs(rho) < 0.7 else "a strong")
    direction = "" if strength == "no clear" else (" positive" if rho > 0 else " negative")
    ratio = (data.groupby("borough")[[h_col, c_col]].sum().query(f"`{h_col}` > 0"))
    ratio_text = ""
    if not ratio.empty:
        ratio["r"] = ratio[c_col] / ratio[h_col]
        top, bottom = ratio["r"].idxmax(), ratio["r"].idxmin()
        ratio_text = (f" Per unit of construction activity, {top.title()} has the most "
                      f"{CRASH_OPTIONS[c_col].lower()} and {bottom.title()} the fewest.")
    findings = html.Div([
        html.B("Findings for this selection: "),
        f"Across {len(totals)} ZIP codes there is {strength}{direction} association "
        f"(rho = {fmt_rho(rho)}) between {HOUSING_OPTIONS[h_col].lower()} and "
        f"{CRASH_OPTIONS[c_col].lower()}.{ratio_text} "
        "Remember: busier ZIPs have more of everything (people, cars, buildings), so association "
        "here is not evidence that construction causes crashes."], className="finding")

    return [html.Div(kpis, className="kpi-row"),
            fig_q1_scatter(data, h_col, c_col), fig_q2_ratio_bars(data, h_col, c_col),
            fig_q3_seasonality(data, h_col, c_col), fig_q4_map(data, h_col, c_col),
            fig_q5_lead_lag(data, h_col, c_col), findings]


# ----------------------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------------------
STORY = html.Div([
    html.H2("The story: does construction travel with street danger?"),
    html.Div([
        html.Div([html.H4("The question"),
                  html.P("Where New York builds, demolishes and renovates, do streets also become more "
                         "dangerous? Planners approve construction ZIP by ZIP, while traffic-safety teams "
                         "respond crash by crash. We put the two side by side to see whether the "
                         "geography of construction and the geography of crash harm overlap.")]),
        html.Div([html.H4("The data"),
                  html.P("DOB NOW: Build job filings (housing & development) and NYPD Motor Vehicle "
                         "Collisions (transportation), both from NYC Open Data, Jan 2020 - Sep 2026. "
                         "They share no key, so both were aggregated to ZIP code x month and joined on "
                         "that pair. Duplicate filings and post-approval amendments were removed, missing "
                         "ZIPs were repaired from the nearest located record, and months with no record "
                         "became true zeros.")]),
        html.Div([html.H4("What to look for"),
                  html.P("Choose a housing and a crash metric, then press Generate Report. The scatter and "
                         "map show where the two domains overlap, the bars compare boroughs, and the "
                         "line and lead/lag charts show timing. The findings box under the charts is "
                         "computed from whatever you selected.")]),
        html.Div([html.H4("Limits"),
                  html.P("Filings are paperwork events, not completed construction, and DOB NOW does not "
                         "cover every job type. We have no population, traffic-volume or street-length "
                         "data, so busy ZIPs naturally score high on both sides. COVID-19 distorts 2020. "
                         "Correlation is not causation: nothing here shows construction causes crashes.")]),
        html.Div([html.H4("Why it matters"),
                  html.P("Planners can flag ZIPs with heavy construction and high crash harm for "
                         "construction-period street-safety measures (protected crossings, truck routing, "
                         "sidewalk sheds). Safety teams can anticipate where activity is rising, and "
                         "advocates get a transparent starting point for neighbourhood conversations.")]),
    ], className="story-grid"),
], className="card")


def build_layout():
    if DF is None:
        return html.Div([html.H2("Data file not found"),
                         html.P(f"Expected {DATA_PATH}. Run `python pipeline.py` (or the notebook) "
                                "first to build it, then restart the app.")], className="card")
    return html.Div([
        html.Div([html.H1("NYC: Construction Activity vs. Street Danger"),
                  html.P("Housing & development filings x motor-vehicle crashes, by ZIP code and month")],
                 className="header"),
        STORY,
        html.Div([
            html.H3("Explore"),
            html.Div([
                html.Div([html.Label("Borough"),
                          dcc.Dropdown(id="borough", options=[{"label": b.title(), "value": b} for b in BOROUGHS],
                                       multi=True, placeholder="All boroughs")], className="control"),
                html.Div([html.Label("Year"),
                          dcc.Dropdown(id="year", options=[{"label": str(y), "value": y} for y in YEARS],
                                       multi=True, placeholder="All years")], className="control"),
                html.Div([html.Label("Housing metric"),
                          dcc.Dropdown(id="housing", options=[{"label": v, "value": k} for k, v in HOUSING_OPTIONS.items()],
                                       value="filings_nb", clearable=False)], className="control"),
                html.Div([html.Label("Crash metric"),
                          dcc.Dropdown(id="crash", options=[{"label": v, "value": k} for k, v in CRASH_OPTIONS.items()],
                                       value="injured", clearable=False)], className="control"),
            ], className="controls"),
            html.Div([
                dcc.Input(id="search-text", type="text", debounce=False, n_submit=0,
                          placeholder='Search mode, e.g. "Brooklyn 2022 new buildings pedestrians"',
                          className="search-box"),
                html.Button("Apply search", id="search-btn", n_clicks=0, className="btn secondary"),
                html.Button("Generate Report", id="generate-btn", n_clicks=0, className="btn primary"),
            ], className="action-row"),
            html.Div(id="search-msg", className="search-msg"),
        ], className="card"),
        dcc.Store(id="search-applied"),
        dcc.Loading(html.Div(id="kpis"), type="dot"),
        html.Div([
            html.Div([html.H4("Q1. Do ZIPs with more construction also see more crash harm?"),
                      dcc.Graph(id="fig-q1")], className="card half"),
            html.Div([html.H4("Q2. Which borough has the most crash harm per unit of construction?"),
                      dcc.Graph(id="fig-q2")], className="card half"),
        ], className="row"),
        html.Div([
            html.Div([html.H4("Q3. Do construction and crashes follow the same seasonal rhythm?"),
                      dcc.Graph(id="fig-q3")], className="card half"),
            html.Div([html.H4("Q4. Where do construction and crash hot spots overlap?"),
                      dcc.Graph(id="fig-q4")], className="card half"),
        ], className="row"),
        html.Div([html.H4("Q5 (bonus). Does construction come before or after crash spikes within a ZIP?"),
                  dcc.Graph(id="fig-q5")], className="card"),
        html.Div(id="findings", className="card"),
        html.Div("Sources: NYC Open Data - DOB NOW: Build Job Application Filings (w9ak-ipjd); "
                 "Motor Vehicle Collisions - Crashes (h9gi-nx95).", className="footer"),
    ], className="page")


app = Dash(__name__, title="NYC Construction vs Street Danger", suppress_callback_exceptions=True)
server = app.server  # exposed for gunicorn
app.layout = build_layout


# ----------------------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------------------
def apply_search(text, current_boroughs, current_years):
    """Return values for the 4 dropdowns + a human-readable message."""
    found = parse_search_query(text, YEARS)
    if not found:
        return no_update, no_update, no_update, no_update, (
            "Nothing recognised. Try a borough, a year (or 2021-2023), a construction type "
            "(new buildings, alterations, demolitions) or a crash type (pedestrian, cyclist, fatal).")
    parts = []
    if "boroughs" in found:
        parts.append("borough: " + ", ".join(b.title() for b in found["boroughs"]))
    if "years" in found:
        parts.append("year: " + ", ".join(map(str, found["years"])))
    if "housing" in found:
        parts.append("housing: " + HOUSING_OPTIONS[found["housing"]])
    if "crash" in found:
        parts.append("crash: " + CRASH_OPTIONS[found["crash"]])
    return (found.get("boroughs", no_update), found.get("years", no_update),
            found.get("housing", no_update), found.get("crash", no_update),
            "Filters applied from search - " + "; ".join(parts))


@app.callback(
    Output("borough", "value"), Output("year", "value"), Output("housing", "value"),
    Output("crash", "value"), Output("search-msg", "children"), Output("search-applied", "data"),
    Input("search-btn", "n_clicks"), Input("search-text", "n_submit"),
    State("search-text", "value"), State("borough", "value"), State("year", "value"),
    prevent_initial_call=True)
def on_search(_clicks, _submits, text, current_boroughs, current_years):
    boroughs, years, housing, crash, message = apply_search(text, current_boroughs, current_years)
    changed = message.startswith("Filters applied")
    # Writing to the store re-triggers the report, so search results appear without an extra click
    return boroughs, years, housing, crash, message, (_clicks or 0) + (_submits or 0) if changed else no_update


@app.callback(
    Output("kpis", "children"), Output("fig-q1", "figure"), Output("fig-q2", "figure"),
    Output("fig-q3", "figure"), Output("fig-q4", "figure"), Output("fig-q5", "figure"),
    Output("findings", "children"),
    Input("generate-btn", "n_clicks"), Input("search-applied", "data"),  # only these two trigger a refresh
    State("borough", "value"), State("year", "value"), State("housing", "value"), State("crash", "value"))
def on_generate(_clicks, _search_token, boroughs, years, housing, crash):
    return build_report(boroughs, years, housing, crash)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
