"""
E-Commerce Intelligence & Bulk Order Classifier
------------------------------------------------
Run with:  streamlit run app.py

Structure
  1. Config & constants
  2. Styling helpers
  3. Data & model loading
  4. Statistics (separability, correlations)
  5. Chart builders
  6. Pages (Overview / Analytics / Classifier)
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, ks_2samp
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="E-Commerce Intelligence & Bulk Classifier",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

BULK, RETAIL = "Bulk / Wholesale", "Individual / Retail"
SEG_COLORS = {BULK: "#39FF14", RETAIL: "#00C6FF"}
SEG_ORDER = {"Segment": [BULK, RETAIL]}

FEATURE_COLS = ["Quantity", "UnitPrice", "Is_UK", "Is_Guest", "Hour", "DayOfWeek", "Month", "Unique_SKUs"]
CORE_FEATURES = ["Quantity", "Unique_SKUs", "UnitPrice", "Hour"]
BINARY_FEATURES = {"Is_UK", "Is_Guest"}
DISCRETE_FEATURES = BINARY_FEATURES | {"Hour", "DayOfWeek", "Month"}
HEAVY_TAIL = {"Quantity", "UnitPrice", "Unique_SKUs", "TotalRevenue"}

MODEL_PATH = "best_rf_pipeline.joblib"
KS_HIGH, KS_MODERATE = 0.35, 0.20

DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

POWER_COLORS = {"High": "#39FF14", "Moderate": "#FFC857", "Low": "#8892A6"}
POWER_LABEL = {"High": "🟢 High", "Moderate": "🟡 Moderate", "Low": "⚪ Low"}

FEATURE_DESC = {
    "Quantity": "Total units on the invoice. Bulk orders are defined by large volumes (≥ 50 units).",
    "Unique_SKUs": "Distinct product codes in the basket. Wholesale buyers restock a wide assortment.",
    "UnitPrice": "Average unit price (£) across the invoice lines.",
    "Hour": "Hour of the day the invoice was created (0-23).",
    "DayOfWeek": "Day of the week (0 = Monday … 6 = Sunday).",
    "Month": "Calendar month of the order (captures Q4 restocking seasonality).",
    "Is_UK": "1 if the order ships to the United Kingdom, otherwise 0.",
    "Is_Guest": "1 if the order has no registered customer ID (guest checkout).",
}

METRIC_GUIDE = """
| Metric | What it tells you | Scale |
|---|---|---|
| **KS statistic** | Largest gap between the two cumulative curves (Bulk vs Retail) | 0 = identical → 1 = fully separated |
| **JS divergence** | Symmetric, bounded difference between the two distributions | 0 → 1 |
| **KL divergence** | Asymmetric, *unbounded* difference (can look huge when ranges barely overlap) | 0 → ∞ |
| **Cohen's d** | Difference in means in standard-deviation units (sign = which segment is higher) | \\|d\\| 0.2 small · 0.5 medium · 0.8 large |
| **Mutual info** | How much knowing the feature reduces uncertainty about the segment | 0 → higher is better |
| **p-value** | With thousands of orders it is almost always ≈ 0 — **rely on effect size, not p** | — |
"""


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1557683316-973673baf926?auto=format&fit=crop&w=2048&q=80");
            background-size: cover; background-position: center; background-attachment: fixed;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stCaptionContainer"] { color: #FFFFFF; }
        [data-testid="stSidebar"] {
            background-color: rgba(0,0,0,0.85) !important;
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255,255,255,0.15);
        }
        .glass-container, [data-testid="stMetric"] {
            background-color: rgba(0,0,0,0.75);
            backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 14px; padding: 20px 24px; margin-bottom: 16px;
        }
        [data-testid="stMetric"] { padding: 14px 18px; }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color: #FFFFFF !important; }
        .glass-container h3 { margin-top: 0; }
        .stTabs [data-baseweb="tab"] { font-weight: 600; }
        .stDataFrame { border-radius: 10px; }
        div.stButton > button, div.stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
            color: #FFFFFF; border: none; border-radius: 8px;
            padding: 0.6rem 1.5rem; font-weight: bold; transition: all 0.25s ease;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            transform: scale(1.02); box-shadow: 0 0 15px rgba(0,198,255,0.6);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(html: str, border: str | None = None) -> None:
    style = f' style="border: 2px solid {border};"' if border else ""
    st.markdown(f'<div class="glass-container"{style}>{html}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.2)", borderwidth=1),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.25)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.25)")
    if height:
        fig.update_layout(height=height)
    return fig


def show(fig: go.Figure, height: int | None = None) -> None:
    st.plotly_chart(style_fig(fig, height), use_container_width=True)


@st.cache_data(show_spinner="Loading data…")
def load_ecommerce_data() -> tuple[pd.DataFrame, str]:
    """Returns (invoice-level dataframe, data source label)."""
    try:
        try:
            raw = pd.read_excel("Online Retail.xlsx", dtype={"InvoiceNo": str, "StockCode": str})
        except FileNotFoundError:
            raw = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1", dtype={"InvoiceNo": str, "StockCode": str})

        raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
        raw["UnitPrice"] = pd.to_numeric(raw["UnitPrice"], errors="coerce")
        raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
        raw = raw.dropna(subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"])

        inv_customer = raw.groupby("InvoiceNo")["CustomerID"].transform("first")
        raw["CustomerID"] = raw["CustomerID"].fillna(inv_customer)
        raw["Is_Guest"] = raw["CustomerID"].isna().astype(int)

        sales = raw[
            (raw["UnitPrice"] > 0)
            & (~raw["InvoiceNo"].str.startswith("C", na=False))
            & (raw["Quantity"] > 0)
        ].copy()
        sales["LineRevenue"] = sales["Quantity"] * sales["UnitPrice"]
        sales["Is_UK"] = (sales["Country"] == "United Kingdom").astype(int)
        sales["Hour"] = sales["InvoiceDate"].dt.hour
        sales["DayOfWeek"] = sales["InvoiceDate"].dt.dayofweek
        sales["Month"] = sales["InvoiceDate"].dt.month

        df = sales.groupby("InvoiceNo").agg(
            Quantity=("Quantity", "sum"),
            UnitPrice=("UnitPrice", "mean"),
            Is_UK=("Is_UK", "first"),
            Is_Guest=("Is_Guest", "first"),
            Hour=("Hour", "first"),
            DayOfWeek=("DayOfWeek", "first"),
            Month=("Month", "first"),
            Unique_SKUs=("StockCode", "nunique"),
            TotalRevenue=("LineRevenue", "sum"),
        ).reset_index()
        df["Is_Bulk"] = np.where((df["Quantity"] >= 50) | (df["TotalRevenue"] >= 500), 1, 0)
        source = "Online Retail dataset"

    except FileNotFoundError:
        rng = np.random.default_rng(42)
        n_retail, n_bulk = 14500, 4000
        retail = pd.DataFrame({
            "Quantity": rng.integers(1, 35, n_retail),
            "UnitPrice": rng.uniform(1.5, 25.0, n_retail),
            "Is_UK": rng.choice([1, 0], size=n_retail, p=[0.90, 0.10]),
            "Is_Guest": rng.choice([0, 1], size=n_retail, p=[0.85, 0.15]),
            "Hour": rng.integers(8, 20, n_retail),
            "DayOfWeek": rng.integers(0, 7, n_retail),
            "Month": rng.choice(range(1, 13), size=n_retail),
            "Unique_SKUs": rng.integers(1, 15, n_retail),
            "Is_Bulk": 0,
        })
        bulk = pd.DataFrame({
            "Quantity": rng.integers(50, 600, n_bulk),
            "UnitPrice": rng.uniform(2.0, 45.0, n_bulk),
            "Is_UK": rng.choice([1, 0], size=n_bulk, p=[0.75, 0.25]),
            "Is_Guest": rng.choice([0, 1], size=n_bulk, p=[0.95, 0.05]),
            "Hour": rng.integers(9, 17, n_bulk),
            "DayOfWeek": rng.integers(0, 5, n_bulk),
            "Month": rng.choice([9, 10, 11, 12], size=n_bulk),
            "Unique_SKUs": rng.integers(10, 80, n_bulk),
            "Is_Bulk": 1,
        })
        df = pd.concat([retail, bulk], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        df["TotalRevenue"] = df["Quantity"] * df["UnitPrice"]
        source = "Synthetic demo data"

    df["Segment"] = df["Is_Bulk"].map({1: BULK, 0: RETAIL})
    return df, source


@st.cache_resource(show_spinner="Loading model…")
def load_model(_df: pd.DataFrame):
    try:
        return joblib.load(MODEL_PATH), f"Loaded {MODEL_PATH}"
    except FileNotFoundError:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ])
        pipe.fit(_df[FEATURE_COLS], _df["Is_Bulk"])
        return pipe, "Trained on the fly (no .joblib found)"


def get_feature_importances(model, names: list[str]) -> dict[str, float]:
    steps = list(model.named_steps.values())[::-1] if hasattr(model, "named_steps") else [model]
    for step in steps:
        if hasattr(step, "feature_importances_"):
            return dict(zip(names, step.feature_importances_))
    return {f: 1.0 / len(names) for f in names}


def _clean(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()


def _divergences(b: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    """Returns (Jensen-Shannon divergence, KL divergence B||R) on a shared 30-bin histogram."""
    both = np.concatenate([b, r])
    if both.min() == both.max():
        return 0.0, 0.0
    bins = np.histogram_bin_edges(both, bins=30)
    p, _ = np.histogram(b, bins=bins)
    q, _ = np.histogram(r, bins=bins)
    p = (p + 1e-10) / (p + 1e-10).sum()
    q = (q + 1e-10) / (q + 1e-10).sum()
    return float(jensenshannon(p, q, base=2.0) ** 2), float(entropy(p, q))


def _power(ks: float) -> str:
    return "High" if ks > KS_HIGH else ("Moderate" if ks > KS_MODERATE else "Low")


@st.cache_data(show_spinner=False)
def compute_separability(_df: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    bulk, retail = _df[_df["Is_Bulk"] == 1], _df[_df["Is_Bulk"] == 0]

    sample = _df.sample(min(20000, len(_df)), random_state=42)
    X = sample[list(features)].apply(pd.to_numeric, errors="coerce").fillna(0)
    mi = mutual_info_classif(
        X, sample["Is_Bulk"], discrete_features=[f in DISCRETE_FEATURES for f in features], random_state=42
    )
    mi_map = dict(zip(features, mi))

    rows = []
    for feat in features:
        b, r = _clean(bulk[feat]), _clean(retail[feat])
        if len(b) < 2 or len(r) < 2:
            continue
        ks, p = ks_2samp(b, r)
        js, kl = _divergences(b, r)
        is_bin = feat in BINARY_FEATURES

        typ_b = float(b.mean() * 100) if is_bin else float(np.median(b))
        typ_r = float(r.mean() * 100) if is_bin else float(np.median(r))
        pooled = np.sqrt(((len(b) - 1) * b.var(ddof=1) + (len(r) - 1) * r.var(ddof=1)) / (len(b) + len(r) - 2))
        d = float((b.mean() - r.mean()) / pooled) if pooled > 0 else 0.0
        ratio = float(b.mean() / r.mean()) if abs(r.mean()) > 1e-9 else np.nan
        rows.append({
            "Feature": feat,
            "Power": _power(ks),
            "Basis": "% of orders" if is_bin else "Median",
            "Bulk": round(typ_b, 2),
            "Retail": round(typ_r, 2),
            "Ratio (B/R)": round(ratio, 2) if pd.notna(ratio) else np.nan,
            "KS": round(float(ks), 4),
            "JS": round(js, 4),
            "KL": round(kl, 4),
            "Cohen d": round(d, 3),
            "Mutual Info": round(float(mi_map[feat]), 4),
            "p-value": max(float(p), 1e-300),
        })

    cols = ["Rank", "Feature", "Power", "Basis", "Bulk", "Retail", "Ratio (B/R)",
            "KS", "JS", "KL", "Cohen d", "Mutual Info", "p-value"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows).sort_values("KS", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", out.index + 1)
    return out


@st.cache_data(show_spinner=False)
def compute_correlations(_df: pd.DataFrame, method: str) -> pd.DataFrame:
    cols = FEATURE_COLS + (["TotalRevenue"] if "TotalRevenue" in _df.columns else []) + ["Is_Bulk"]
    return _df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method).round(2)


def top_feature_pairs(corr: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    feats = [c for c in corr.columns if c not in ("Is_Bulk", "TotalRevenue")]
    rows = [(a, b, corr.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:]]
    pairs = pd.DataFrame(rows, columns=["Feature A", "Feature B", "Correlation"])
    pairs["Strength"] = pd.cut(pairs["Correlation"].abs(), [0, 0.3, 0.5, 1.0001],
                               labels=["Weak", "Moderate", "Strong"], include_lowest=True)
    return pairs.reindex(pairs["Correlation"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)


def segment_comparison_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        b = pd.Series(_clean(df.loc[df["Is_Bulk"] == 1, feat]))
        r = pd.Series(_clean(df.loc[df["Is_Bulk"] == 0, feat]))
        if b.empty or r.empty:
            continue
        rows.append({
            "Feature": feat,
            "Bulk Mean": round(b.mean(), 2), "Retail Mean": round(r.mean(), 2),
            "Bulk IQR": round(b.quantile(0.75) - b.quantile(0.25), 2),
            "Retail IQR": round(r.quantile(0.75) - r.quantile(0.25), 2),
        })
    return pd.DataFrame(rows)


def build_takeaways(sep: pd.DataFrame) -> list[str]:
    if sep.empty:
        return []
    top, low = sep.iloc[0], sep.iloc[-1]
    n_high = int((sep["Power"] == "High").sum())
    notes = [
        f"**{top['Feature']}** is the strongest separator (KS = {top['KS']:.2f}): "
        f"{top['Basis'].lower()} is **{top['Bulk']:,.1f}** for Bulk vs **{top['Retail']:,.1f}** for Retail.",
        f"**{n_high} of {len(sep)}** features have high separability (KS > {KS_HIGH}); "
        f"the rest add little on their own.",
        f"**{low['Feature']}** is the weakest signal (KS = {low['KS']:.2f}) — the two segments behave almost alike on it.",
    ]
    q = sep[sep["Feature"] == "Quantity"]
    if not q.empty and q["KS"].iloc[0] >= 0.98:
        notes.append(
            "⚠️ **Quantity has KS ≈ 1.0 by construction**: `Is_Bulk` is defined as Quantity ≥ 50 *or* revenue ≥ £500, "
            "so the label leaks into this feature. Treat the model's accuracy as an upper bound; "
            "retrain without `Quantity` (and revenue-derived signals) to test real-world generalisation."
        )
    return notes


def fig_ks_ranking(sep: pd.DataFrame) -> go.Figure:
    d = sep.sort_values("KS")
    fig = px.bar(d, x="KS", y="Feature", orientation="h", color="Power",
                 color_discrete_map=POWER_COLORS, category_orders={"Power": ["High", "Moderate", "Low"]},
                 text=d["KS"].map("{:.2f}".format), title="Separability ranking (KS statistic)")
    fig.update_traces(textposition="outside")
    for x, label in [(KS_MODERATE, "Moderate"), (KS_HIGH, "High")]:
        fig.add_vline(x=x, line_dash="dot", line_color="rgba(255,255,255,0.5)",
                      annotation_text=label, annotation_position="top")
    fig.update_xaxes(range=[0, 1.1], title="KS statistic (0 = identical, 1 = fully separated)")
    fig.update_yaxes(title=None)
    return fig


def fig_metric_heatmap(sep: pd.DataFrame) -> go.Figure:
    m = sep.set_index("Feature")[["KS", "JS", "Cohen d", "Mutual Info", "KL"]].copy()
    m["Cohen d"] = m["Cohen d"].abs()
    m = m.rename(columns={"Cohen d": "|Cohen d|"})
    scaled = m / m.max().replace(0, 1)
    fig = px.imshow(scaled, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis",
                    zmin=0, zmax=1, title="Do all metrics agree? (each column scaled 0–1)")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(side="top")
    return fig


def fig_effect_size(sep: pd.DataFrame) -> go.Figure:
    d = sep.sort_values("Cohen d").copy()
    d["Direction"] = np.where(d["Cohen d"] >= 0, "Higher in Bulk", "Higher in Retail")
    fig = px.bar(d, x="Cohen d", y="Feature", orientation="h", color="Direction",
                 color_discrete_map={"Higher in Bulk": SEG_COLORS[BULK], "Higher in Retail": SEG_COLORS[RETAIL]},
                 title="Effect size & direction (Cohen's d)")
    for x in (-0.8, 0.8):
        fig.add_vline(x=x, line_dash="dot", line_color="rgba(255,255,255,0.4)")
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.6)")
    fig.update_yaxes(title=None)
    fig.update_xaxes(title="← Retail higher | Bulk higher →  (dotted = large effect ±0.8)")
    return fig


def fig_ratio(sep: pd.DataFrame) -> go.Figure:
    d = sep.dropna(subset=["Ratio (B/R)"]).copy()
    d = d[d["Ratio (B/R)"] > 0].sort_values("Ratio (B/R)")
    d["Direction"] = np.where(d["Ratio (B/R)"] >= 1, "Bulk higher", "Retail higher")
    fig = px.bar(d, x="Ratio (B/R)", y="Feature", orientation="h", log_x=True, color="Direction",
                 color_discrete_map={"Bulk higher": SEG_COLORS[BULK], "Retail higher": SEG_COLORS[RETAIL]},
                 text=d["Ratio (B/R)"].map("{:.2f}x".format),
                 title="How many times bigger is the Bulk mean? (log scale)")
    fig.add_vline(x=1, line_color="rgba(255,255,255,0.6)")
    fig.update_yaxes(title=None)
    return fig


def fig_distribution(df: pd.DataFrame, feat: str, kind: str) -> go.Figure:
    if feat in BINARY_FEATURES:
        t = (df.groupby("Segment")[feat].mean() * 100).reset_index(name="Share (%)")
        return px.bar(t, x="Segment", y="Share (%)", color="Segment", color_discrete_map=SEG_COLORS,
                      title=f"Share of orders with {feat} = 1")

    plot_df = df[df[feat] <= df[feat].quantile(0.97)] if feat in HEAVY_TAIL else df
    if kind == "Violin":
        return px.violin(plot_df, x="Segment", y=feat, color="Segment", box=True, points=False,
                         color_discrete_map=SEG_COLORS, title=f"{feat} by segment (violin)")
    if kind == "KDE":
        b = _clean(plot_df.loc[plot_df["Is_Bulk"] == 1, feat])
        r = _clean(plot_df.loc[plot_df["Is_Bulk"] == 0, feat])
        try:
            if len(b) > 1 and len(r) > 1 and b.std() > 0 and r.std() > 0:
                fig = ff.create_distplot([b, r], [BULK, RETAIL], colors=[SEG_COLORS[BULK], SEG_COLORS[RETAIL]],
                                         show_hist=False, show_rug=False)
                fig.update_layout(title=f"{feat} density (KDE)", xaxis_title=feat, yaxis_title="Density")
                return fig
        except Exception:
            pass
    return px.histogram(plot_df, x=feat, color="Segment", barmode="overlay", opacity=0.65, nbins=40,
                        histnorm="probability density", color_discrete_map=SEG_COLORS,
                        title=f"{feat} distribution")


def fig_ecdf(df: pd.DataFrame, feat: str) -> go.Figure:
    """Cumulative curves with the maximum gap (= KS statistic) drawn explicitly."""
    b = np.sort(_clean(df.loc[df["Is_Bulk"] == 1, feat]))
    r = np.sort(_clean(df.loc[df["Is_Bulk"] == 0, feat]))
    both = np.concatenate([b, r])
    vals = np.unique(both)
    grid = vals if len(vals) <= 2000 else np.unique(np.quantile(both, np.linspace(0, 1, 2000)))
    fb = np.searchsorted(b, grid, side="right") / len(b)
    fr = np.searchsorted(r, grid, side="right") / len(r)
    gap = np.abs(fb - fr)
    i = int(gap.argmax())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=fb, mode="lines", name=BULK,
                             line=dict(color=SEG_COLORS[BULK], shape="hv")))
    fig.add_trace(go.Scatter(x=grid, y=fr, mode="lines", name=RETAIL,
                             line=dict(color=SEG_COLORS[RETAIL], shape="hv")))
    fig.add_trace(go.Scatter(x=[grid[i], grid[i]], y=[fb[i], fr[i]], mode="lines+markers", name="Max gap (KS)",
                             line=dict(color="#FFFFFF", dash="dash")))
    fig.add_annotation(x=grid[i], y=(fb[i] + fr[i]) / 2, text=f"KS = {gap[i]:.3f}", showarrow=True,
                       ax=60, ay=0, font=dict(color="#FFFFFF"), bgcolor="rgba(0,0,0,0.6)")
    if feat in HEAVY_TAIL:
        fig.update_xaxes(range=[float(grid[0]), float(np.quantile(both, 0.99))])
    fig.update_layout(title=f"Cumulative curves — the KS gap for {feat}",
                      xaxis_title=feat, yaxis_title="Cumulative share of orders")
    return fig


def fig_corr_heatmap(corr: pd.DataFrame, method: str) -> go.Figure:
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    fig = px.imshow(corr.mask(mask), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title=f"Feature correlation ({method.title()})")
    return fig


def fig_target_corr(corr: pd.DataFrame) -> go.Figure:
    s = corr["Is_Bulk"].drop("Is_Bulk")
    d = s.reindex(s.abs().sort_values().index).reset_index()
    d.columns = ["Feature", "Correlation"]
    d["Direction"] = np.where(d["Correlation"] >= 0, "Pushes toward Bulk", "Pushes toward Retail")
    fig = px.bar(d, x="Correlation", y="Feature", orientation="h", color="Direction",
                 color_discrete_map={"Pushes toward Bulk": SEG_COLORS[BULK], "Pushes toward Retail": SEG_COLORS[RETAIL]},
                 title="Correlation with the Bulk label")
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(title=None)
    return fig


def fig_relationship(df: pd.DataFrame, x: str, y: str, log: bool) -> go.Figure:
    s = df.dropna(subset=[x, y])
    if log:
        s = s[(s[x] > 0) & (s[y] > 0)]
    s = s.sample(n=min(3000, len(s)), random_state=42)
    return px.scatter(s, x=x, y=y, color="Segment", color_discrete_map=SEG_COLORS, opacity=0.6,
                      log_x=log, log_y=log, marginal_x="box", marginal_y="box",
                      title=f"{x} vs {y}")


def fig_parallel(df: pd.DataFrame, dims: list[str]) -> go.Figure:
    s = df.sample(n=min(1500, len(df)), random_state=42).copy()
    for c in ("Quantity", "UnitPrice", "Unique_SKUs"):
        if c in dims:
            s[c] = s[c].clip(upper=s[c].quantile(0.98))
    fig = px.parallel_coordinates(
        s, dimensions=dims, color="Is_Bulk",
        color_continuous_scale=[[0, SEG_COLORS[RETAIL]], [1, SEG_COLORS[BULK]]],
        title="Parallel coordinates — each line is one order (green = Bulk, blue = Retail)",
    )
    fig.update_traces(labelfont=dict(color="#FFFFFF"), tickfont=dict(color="#FFFFFF"),
                      rangefont=dict(color="#FFFFFF"))
    fig.update_coloraxes(showscale=False)
    return fig


def share_table(df: pd.DataFrame, col: str, labels: dict) -> pd.DataFrame:
    t = df.groupby(["Segment", col]).size().reset_index(name="Orders")
    t["Share (%)"] = (t["Orders"] / t.groupby("Segment")["Orders"].transform("sum") * 100).round(1)
    t["Label"] = t[col].map(labels)
    return t.sort_values(col)


def fig_bulk_heatmap(df: pd.DataFrame) -> go.Figure:
    pv = (df.pivot_table(index="DayOfWeek", columns="Hour", values="Is_Bulk", aggfunc="mean") * 100)
    pv = pv.reindex(range(7)).rename(index=DAY_NAMES)
    calm_scale = [
        [0.0, "#24445C"],
        [0.3, "#2F7A87"],
        [0.6, "#6FB7A4"],
        [0.85, "#C9DDA8"],
        [1.0, "#F3E2A9"],
    ]
    fig = px.imshow(
        pv, aspect="auto", color_continuous_scale=calm_scale, text_auto=".0f",
        labels=dict(x="Hour of day", y="Day", color="% Bulk"),
        title="When do Bulk orders happen? (% of orders that are Bulk)",
    )
    fig.update_traces(xgap=2, ygap=2,
                      hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Bulk share: %{z:.1f}%<extra></extra>")
    fig.update_xaxes(dtick=1, side="bottom")
    fig.update_coloraxes(colorbar=dict(ticksuffix="%", title=dict(text="% Bulk")))
    return fig


def fig_pca_boundary(df: pd.DataFrame, model) -> go.Figure | None:
    valid = df.dropna(subset=FEATURE_COLS)
    if len(valid) < 5:
        return None
    sample = valid.sample(n=min(2500, len(valid)), random_state=42).copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(sample[FEATURE_COLS])
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(Xs)
    sample["PC1"], sample["PC2"] = pcs[:, 0], pcs[:, 1]

    pad_x = max((sample["PC1"].max() - sample["PC1"].min()) * 0.08, 0.5)
    pad_y = max((sample["PC2"].max() - sample["PC2"].min()) * 0.08, 0.5)
    gx = np.linspace(sample["PC1"].min() - pad_x, sample["PC1"].max() + pad_x, 60)
    gy = np.linspace(sample["PC2"].min() - pad_y, sample["PC2"].max() + pad_y, 60)
    xx, yy = np.meshgrid(gx, gy)
    grid = scaler.inverse_transform(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    zz = np.clip(model.predict_proba(pd.DataFrame(grid, columns=FEATURE_COLS))[:, 1], 0, 1).reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=gx, y=gy, z=zz, hoverinfo="skip", name="P(Bulk)",
        colorscale=[[0, "rgba(0,198,255,0.35)"], [0.5, "rgba(255,255,255,0.10)"], [1, "rgba(57,255,20,0.35)"]],
        contours=dict(start=0, end=1, size=0.1, showlines=True),
        colorbar=dict(title=dict(text="P(Bulk)", font=dict(color="#FFFFFF")), tickfont=dict(color="#FFFFFF")),
    ))
    for seg in (BULK, RETAIL):
        sub = sample[sample["Segment"] == seg]
        fig.add_trace(go.Scatter(x=sub["PC1"], y=sub["PC2"], mode="markers", name=seg,
                                 marker=dict(color=SEG_COLORS[seg], size=6, opacity=0.75,
                                             line=dict(width=0.5, color="#000000"))))
    v = pca.explained_variance_ratio_ * 100
    fig.update_layout(title=f"PCA projection with model decision surface (explained variance {v[0]:.1f}% + {v[1]:.1f}%)",
                      xaxis_title=f"PC1 ({v[0]:.1f}%)", yaxis_title=f"PC2 ({v[1]:.1f}%)")
    return fig


def fig_gauge(value: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 44}},
        title={"text": "Prediction Confidence", "font": {"color": "#FFFFFF", "size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#FFFFFF", "tickfont": {"color": "#FFFFFF"}},
            "bar": {"color": color}, "bgcolor": "rgba(255,255,255,0.1)",
            "borderwidth": 2, "bordercolor": "rgba(255,255,255,0.3)",
            "steps": [{"range": [0, 50], "color": "rgba(255,255,255,0.05)"},
                      {"range": [50, 80], "color": "rgba(255,255,255,0.12)"},
                      {"range": [80, 100], "color": "rgba(255,255,255,0.20)"}],
        },
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def render_overview(df: pd.DataFrame) -> None:
    st.title("🏪 E-Commerce Data Pipeline & Buyer Segmentation")

    rev_share = df.loc[df["Is_Bulk"] == 1, "TotalRevenue"].sum() / df["TotalRevenue"].sum() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Clean Invoices", f"{len(df):,}", "Net of cancellations")
    c2.metric("Bulk Order Ratio", f"{df['Is_Bulk'].mean() * 100:.1f}%", "Share of invoices")
    c3.metric("Average Order Value", f"£{df['TotalRevenue'].mean():,.2f}", "Net AOV")
    c4.metric("Bulk Revenue Share", f"{rev_share:.1f}%", "Share of revenue")

    left, right = st.columns(2)
    with left:
        counts = df["Segment"].value_counts().reset_index()
        counts.columns = ["Segment", "Orders"]
        show(px.pie(counts, names="Segment", values="Orders", hole=0.55, color="Segment",
                    color_discrete_map=SEG_COLORS, title="Share of orders"), 340)
    with right:
        rev = df.groupby("Segment")["TotalRevenue"].sum().reset_index()
        show(px.pie(rev, names="Segment", values="TotalRevenue", hole=0.55, color="Segment",
                    color_discrete_map=SEG_COLORS, title="Share of revenue"), 340)

    card("""
        <h3>⚙️ End-to-End Data Pipeline</h3>
        <p>Raw retail logs contain cancellation invoices (prefix <code>C</code>), zero-price adjustments and guest
        checkouts. The pipeline removes cancellations, keeps guest orders via invoice-level grouping, and engineers
        market and time signals (<code>Is_UK</code>, <code>Hour</code>, <code>DayOfWeek</code>, <code>Month</code>).</p>
        <p>A <b>Random Forest</b> then classifies each order as <b>Bulk / Wholesale (B2B)</b> or
        <b>Individual / Retail (B2C)</b> to support inventory allocation and targeted discounting.</p>
    """)

    a, b = st.columns(2)
    with a:
        card("""
            <h3 style="color:#39FF14;">🟢 Bulk / Wholesale (B2B signal)</h3>
            <ul>
              <li><b>High volume:</b> ≥ 50 units or ≥ £500 basket value.</li>
              <li><b>Wide assortment:</b> dozens of distinct SKUs per invoice.</li>
              <li><b>Business hours:</b> weekdays 09:00–17:00, peaking in Q4.</li>
            </ul>""")
    with b:
        card("""
            <h3 style="color:#00C6FF;">🔵 Individual / Retail (B2C segment)</h3>
            <ul>
              <li><b>Small baskets:</b> only a few units per checkout.</li>
              <li><b>More guest checkouts</b> (<code>Is_Guest = 1</code>).</li>
              <li><b>Evenings &amp; weekends</b> with strong UK concentration.</li>
            </ul>""")


def tab_separability(df: pd.DataFrame, sep: pd.DataFrame, features: list[str]) -> None:
    with st.expander("ℹ️ How to read these metrics"):
        st.markdown(METRIC_GUIDE)

    notes = build_takeaways(sep)
    if notes:
        st.markdown("### 💡 Key takeaways")
        for n in notes:
            st.markdown(f"- {n}")

    table = sep.assign(Power=sep["Power"].map(POWER_LABEL))
    st.dataframe(
        table, hide_index=True, use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Power": st.column_config.TextColumn("Separability", width="small"),
            "Basis": st.column_config.TextColumn(help="Median for numeric features, % of orders for binary flags"),
            "Bulk": st.column_config.NumberColumn("Bulk", format="%.2f"),
            "Retail": st.column_config.NumberColumn("Retail", format="%.2f"),
            "Ratio (B/R)": st.column_config.NumberColumn(format="%.2fx"),
            "KS": st.column_config.ProgressColumn("KS", min_value=0.0, max_value=1.0, format="%.3f"),
            "JS": st.column_config.ProgressColumn("JS", min_value=0.0, max_value=1.0, format="%.3f"),
            "KL": st.column_config.NumberColumn("KL (B‖R)", format="%.2f"),
            "Cohen d": st.column_config.NumberColumn("Cohen's d", format="%+.2f"),
            "Mutual Info": st.column_config.NumberColumn("Mutual info", format="%.3f"),
            "p-value": st.column_config.NumberColumn(format="%.1e"),
        },
    )
    st.download_button("⬇️ Download table (CSV)", sep.to_csv(index=False).encode("utf-8"),
                       "feature_separability.csv", "text/csv")

    c1, c2 = st.columns(2)
    with c1:
        show(fig_ks_ranking(sep), 380)
    with c2:
        show(fig_metric_heatmap(sep), 380)
    c3, c4 = st.columns(2)
    with c3:
        show(fig_effect_size(sep), 380)
    with c4:
        show(fig_ratio(sep), 380)

    with st.expander("📋 Mean & IQR comparison"):
        st.dataframe(segment_comparison_table(df, features), hide_index=True, use_container_width=True)


def tab_relationships(df: pd.DataFrame) -> None:
    method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True,
                      help="Spearman is more robust to the heavy tails in Quantity and UnitPrice.")
    corr = compute_correlations(df, method)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        show(fig_corr_heatmap(corr, method), 460)
    with c2:
        show(fig_target_corr(corr), 280)
        st.markdown("**Most correlated feature pairs** (watch for redundancy)")
        st.dataframe(top_feature_pairs(corr), hide_index=True, use_container_width=True,
                     column_config={"Correlation": st.column_config.NumberColumn(format="%+.2f")})

    st.markdown("### 🔍 Relationship explorer")
    numeric = [c for c in ["Quantity", "UnitPrice", "Unique_SKUs", "TotalRevenue", "Hour"] if c in df.columns]
    e1, e2, e3 = st.columns([1, 1, 1])
    x = e1.selectbox("X axis", numeric, index=0)
    y = e2.selectbox("Y axis", numeric, index=min(2, len(numeric) - 1))
    log = e3.checkbox("Log scale", value=True)
    show(fig_relationship(df, x, y, log), 520)

    show(fig_parallel(df, ["Quantity", "Unique_SKUs", "UnitPrice", "Hour", "DayOfWeek", "Month"]), 460)

    with st.expander("🧩 Pairwise scatter matrix"):
        try:
            s = df.dropna(subset=CORE_FEATURES).sample(n=min(1200, len(df)), random_state=42).copy()
            for c in ("Quantity", "UnitPrice"):
                s[c] = np.clip(s[c], s[c].quantile(0.01), s[c].quantile(0.98))
            fig = px.scatter_matrix(s, dimensions=CORE_FEATURES, color="Segment", opacity=0.65,
                                    color_discrete_map=SEG_COLORS, title="Pairwise scatter matrix")
            fig.update_traces(diagonal_visible=False, marker=dict(size=4))
            show(fig, 680)
        except Exception as exc:
            st.warning(f"Unable to display scatter matrix: {exc}")


def tab_distributions(df: pd.DataFrame, sep_all: pd.DataFrame, importances: dict) -> None:
    c1, c2 = st.columns([1, 1])
    feat = c1.selectbox("Feature", FEATURE_COLS, index=0)
    kind = c2.radio("View", ["Violin", "KDE", "Histogram"], horizontal=True)
    st.caption(FEATURE_DESC[feat])

    row = sep_all[sep_all["Feature"] == feat]
    if not row.empty:
        r = row.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("KS statistic", f"{r['KS']:.3f}", POWER_LABEL[r["Power"]], delta_color="off")
        m2.metric(f"Bulk ({r['Basis'].lower()})", f"{r['Bulk']:,.2f}")
        m3.metric(f"Retail ({r['Basis'].lower()})", f"{r['Retail']:,.2f}")
        m4.metric("RF importance", f"{importances.get(feat, 0) * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        show(fig_distribution(df, feat, kind), 420)
    with right:
        show(fig_ecdf(df, feat), 420)
    st.caption("Right: the dashed white line is the largest vertical gap between the two cumulative curves — "
               "that gap *is* the KS statistic.")


def tab_time_market(df: pd.DataFrame) -> None:
    show(fig_bulk_heatmap(df), 380)

    c1, c2 = st.columns(2)
    with c1:
        t = share_table(df, "DayOfWeek", DAY_NAMES)
        show(px.line(t, x="Label", y="Share (%)", color="Segment", markers=True, category_orders=SEG_ORDER,
                     color_discrete_map=SEG_COLORS, title="Weekly pattern (% of each segment's orders)"), 340)
    with c2:
        t = share_table(df, "Month", MONTH_NAMES)
        show(px.bar(t, x="Label", y="Share (%)", color="Segment", barmode="group", category_orders=SEG_ORDER,
                    color_discrete_map=SEG_COLORS, title="Monthly seasonality (% of each segment's orders)"), 340)

    c3, c4 = st.columns(2)
    with c3:
        flags = (df.groupby("Segment")[["Is_UK", "Is_Guest"]].mean() * 100).reset_index()
        flags = flags.melt(id_vars="Segment", var_name="Flag", value_name="Share (%)")
        flags["Flag"] = flags["Flag"].map({"Is_UK": "UK orders", "Is_Guest": "Guest checkouts"})
        show(px.bar(flags, x="Flag", y="Share (%)", color="Segment", barmode="group",
                    color_discrete_map=SEG_COLORS, title="Market & checkout profile (%)"), 340)
    with c4:
        feat = st.selectbox("Hourly mean of", ["Quantity", "Unique_SKUs", "UnitPrice", "TotalRevenue"])
        hr = df.groupby(["Segment", "Hour"])[feat].mean().reset_index().sort_values("Hour")
        show(px.line(hr, x="Hour", y=feat, color="Segment", markers=True,
                     color_discrete_map=SEG_COLORS, title=f"Hourly mean of {feat}"), 300)


def tab_model(df: pd.DataFrame, model, sep_all: pd.DataFrame, importances: dict) -> None:
    imp = pd.DataFrame({"Feature": FEATURE_COLS,
                        "Importance (%)": [round(importances.get(f, 0) * 100, 2) for f in FEATURE_COLS]})
    c1, c2 = st.columns(2)
    with c1:
        show(px.bar(imp.sort_values("Importance (%)"), x="Importance (%)", y="Feature", orientation="h",
                    color="Importance (%)", color_continuous_scale="Blues",
                    title="Random Forest feature importance").update_layout(coloraxis_showscale=False), 400)
    with c2:
        m = sep_all.merge(imp, on="Feature")
        fig = px.scatter(m, x="KS", y="Importance (%)", text="Feature", color="Power",
                         color_discrete_map=POWER_COLORS, title="Statistics vs. model: do they agree?")
        fig.update_traces(textposition="top center", marker=dict(size=14))
        fig.update_xaxes(range=[0, 1.1])
        show(fig, 400)
    st.caption("Top-right = strong on both the statistical test and in the model. "
               "High KS with low importance usually means the feature is redundant with another one.")

    with st.spinner("Computing decision surface…"):
        fig = fig_pca_boundary(df, model)
    if fig is None:
        st.warning("Not enough valid rows for a PCA projection.")
    else:
        show(fig, 600)
        st.caption("2-D PCA projection of the 8-D feature space; the background is an approximation of the model's "
                   "P(Bulk) obtained by mapping the 2-D grid back to feature space.")


def render_analytics(df: pd.DataFrame, model) -> None:
    st.title("📊 Strategic Analytics & Feature Separability")

    scope = st.radio("Feature scope", ["Core features (4)", "All model features (8)"], horizontal=True)
    features = CORE_FEATURES if scope.startswith("Core") else FEATURE_COLS

    sep = compute_separability(df, tuple(features))
    sep_all = compute_separability(df, tuple(FEATURE_COLS))
    importances = get_feature_importances(model, FEATURE_COLS)

    t1, t2, t3, t4, t5 = st.tabs(["📐 Separability", "🔗 Relationships", "📈 Distributions",
                                  "🕒 Time & Market", "🧠 Model"])
    with t1:
        tab_separability(df, sep, features)
    with t2:
        tab_relationships(df)
    with t3:
        tab_distributions(df, sep_all, importances)
    with t4:
        tab_time_market(df)
    with t5:
        tab_model(df, model, sep_all, importances)


def render_classifier(model) -> None:
    st.title("🤖 Real-Time Order Classifier")
    card("<p>Enter the basket parameters to classify an incoming checkout as "
         "<b>Bulk / Wholesale (B2B)</b> or <b>Individual / Retail (B2C)</b>.</p>")

    with st.form("order_input_form"):
        left, right = st.columns(2)
        with left:
            quantity = st.number_input("Total Quantity (units)", min_value=1, value=75, step=1)
            unit_price = st.number_input("Average Unit Price (£)", min_value=0.01, value=12.50, format="%.2f")
            unique_skus = st.number_input("Unique SKUs in basket", min_value=1, value=18, step=1)
            is_uk = st.selectbox("Market (Is_UK)", [1, 0],
                                 format_func=lambda x: "United Kingdom (1)" if x == 1 else "International (0)")
        with right:
            is_guest = st.selectbox("Checkout type (Is_Guest)", [0, 1],
                                    format_func=lambda x: "Registered account (0)" if x == 0 else "Guest checkout (1)")
            hour = st.slider("Order hour (24h)", 0, 23, 14)
            day = st.selectbox("Day of week", list(range(7)), index=2,
                               format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday",
                                                      "Friday", "Saturday", "Sunday"][x])
            month = st.slider("Month", 1, 12, 11)
        submitted = st.form_submit_button("Analyze Order", use_container_width=True)

    if not submitted:
        return

    X = pd.DataFrame([{
        "Quantity": quantity, "UnitPrice": unit_price, "Is_UK": is_uk, "Is_Guest": is_guest,
        "Hour": hour, "DayOfWeek": day, "Month": month, "Unique_SKUs": unique_skus,
    }])[FEATURE_COLS]
    probs = model.predict_proba(X)[0]
    is_bulk = int(model.predict(X)[0]) == 1
    confidence = (probs[1] if is_bulk else probs[0]) * 100
    color = SEG_COLORS[BULK] if is_bulk else SEG_COLORS[RETAIL]
    title = "BULK / WHOLESALE (B2B)" if is_bulk else "INDIVIDUAL / RETAIL (B2C)"
    revenue = quantity * unit_price

    c1, c2 = st.columns(2)
    with c1:
        card(f"""
            <div style="text-align:center; padding: 24px 0;">
              <h4>Order Classification Verdict</h4>
              <h1 style="color:{color}; font-size:2.2rem; margin:15px 0;">{title}</h1>
              <p style="font-size:1.1rem;">Estimated basket value: <b>£{revenue:,.2f}</b><br>
              Bulk probability: <b>{probs[1] * 100:.2f}%</b> &nbsp;|&nbsp;
              Retail probability: <b>{probs[0] * 100:.2f}%</b></p>
            </div>""", border=color)
        rule_bulk = quantity >= 50 or revenue >= 500
        agree = rule_bulk == is_bulk
        st.caption(f"Business rule (≥ 50 units or ≥ £500) says **{'Bulk' if rule_bulk else 'Retail'}** — "
                   f"{'consistent with' if agree else 'differs from'} the model.")
    with c2:
        show(fig_gauge(confidence, color), 320)


def main() -> None:
    inject_css()
    df, data_source = load_ecommerce_data()
    model, model_source = load_model(df)

    st.sidebar.title("🛒 E-Commerce BI")
    st.sidebar.caption("Strategic retail analytics & B2B/B2C order classification")
    page = st.sidebar.radio("Navigation", ["🏪 Executive Overview", "📊 Strategic Analytics", "🤖 Order Classifier"])
    st.sidebar.divider()
    st.sidebar.caption(f"**Data:** {data_source} · {len(df):,} invoices")
    st.sidebar.caption(f"**Model:** {model_source}")

    if page.startswith("🏪"):
        render_overview(df)
    elif page.startswith("📊"):
        render_analytics(df, model)
    else:
        render_classifier(model)


main()
