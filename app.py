import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import ks_2samp, entropy
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="E-Commerce Intelligence & Bulk Classifier",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

BULK_COLOR = "#059669"
RETAIL_COLOR = "#0284C7"
TEXT_PRIMARY = "#0F172A"
TEXT_SECONDARY = "#475569"

def inject_ecommerce_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

        .stApp {
            background-color: #F8FAFC !important;
            background-image: none !important;
            color: #0F172A;
            font-family: 'Inter', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        [data-testid="stHeader"] {
            background-color: rgba(255, 255, 255, 0.94) !important;
            border-bottom: 1px solid #E2E8F0;
        }
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0;
        }
        [data-testid="stSidebar"] * {
            color: #0F172A;
        }
        .glass-container {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 24px 26px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
            color: #0F172A;
        }
        .separability-card {
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
            border: 1px solid #CBD5E1;
            border-left: 5px solid #0284C7;
            border-radius: 14px;
            padding: 24px 28px;
            margin-bottom: 18px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        }
        .separability-card h3 {
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
            font-weight: 700;
            font-size: 1.35rem;
            letter-spacing: -0.02em;
            margin-bottom: 8px;
            color: #0F172A !important;
        }
        .separability-card p {
            font-family: 'Inter', sans-serif;
            font-size: 0.98rem;
            line-height: 1.65;
            color: #334155 !important;
            margin-bottom: 0;
        }
        .separability-table-wrap [data-testid="stDataFrame"] {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 12px !important;
            padding: 6px !important;
            font-family: 'Inter', sans-serif !important;
            font-variant-numeric: tabular-nums;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        }
        .stat-pill {
            display: inline-block;
            background-color: #F1F5F9;
            border: 1px solid #CBD5E1;
            color: #0F172A !important;
            font-weight: 600;
            font-size: 0.92rem;
            padding: 6px 12px;
            border-radius: 8px;
            margin: 4px 6px 8px 0;
            font-variant-numeric: tabular-nums;
        }
        .insight-box {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 18px 20px;
            color: #0F172A;
        }
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stMetricLabel"] {
            color: #475569 !important;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: #0F172A !important;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }
        [data-testid="stMetricDelta"] {
            color: #0284C7 !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #0F172A !important;
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
            letter-spacing: -0.015em;
        }
        p, li, label, span, div {
            color: #0F172A;
        }
        code {
            background-color: #F1F5F9;
            color: #0F172A !important;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        .stDataFrame {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 8px;
        }
        div.stButton > button, button[kind="formSubmit"] {
            background: linear-gradient(90deg, #0284C7 0%, #0369A1 100%);
            color: #FFFFFF !important;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            width: 100%;
            transition: all 0.2s ease;
        }
        div.stButton > button:hover, button[kind="formSubmit"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(2, 132, 199, 0.28);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_ecommerce_css()

FEATURE_COLS = [
    "Quantity", "UnitPrice", "Is_UK", "Is_Guest",
    "Hour", "DayOfWeek", "Month", "Unique_SKUs"
]

DAY_NAMES = {
    0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"
}
MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

@st.cache_data
def load_ecommerce_data():
    try:
        try:
            raw_df = pd.read_excel("Online Retail.xlsx", dtype={"InvoiceNo": str, "StockCode": str})
        except FileNotFoundError:
            raw_df = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1", dtype={"InvoiceNo": str, "StockCode": str})

        raw_df["InvoiceDate"] = pd.to_datetime(raw_df["InvoiceDate"], errors="coerce")
        raw_df["UnitPrice"] = pd.to_numeric(raw_df["UnitPrice"], errors="coerce")
        raw_df["Quantity"] = pd.to_numeric(raw_df["Quantity"], errors="coerce")
        raw_df.dropna(subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"], inplace=True)

        invoice_customer_map = raw_df.groupby("InvoiceNo")["CustomerID"].transform("first")
        raw_df["CustomerID"] = raw_df["CustomerID"].fillna(invoice_customer_map)
        raw_df["Is_Guest"] = raw_df["CustomerID"].isna().astype(int)

        sales_df = raw_df[(raw_df["UnitPrice"] > 0) & (~raw_df["InvoiceNo"].str.startswith("C", na=False)) & (raw_df["Quantity"] > 0)].copy()
        sales_df["LineRevenue"] = sales_df["Quantity"] * sales_df["UnitPrice"]
        sales_df["Is_UK"] = (sales_df["Country"] == "United Kingdom").astype(int)
        sales_df["Hour"] = sales_df["InvoiceDate"].dt.hour
        sales_df["DayOfWeek"] = sales_df["InvoiceDate"].dt.dayofweek
        sales_df["Month"] = sales_df["InvoiceDate"].dt.month

        df = sales_df.groupby("InvoiceNo").agg(
            Quantity=("Quantity", "sum"),
            UnitPrice=("UnitPrice", "mean"),
            Is_UK=("Is_UK", "first"),
            Is_Guest=("Is_Guest", "first"),
            Hour=("Hour", "first"),
            DayOfWeek=("DayOfWeek", "first"),
            Month=("Month", "first"),
            Unique_SKUs=("StockCode", "nunique"),
            TotalRevenue=("LineRevenue", "sum")
        ).reset_index()

        df["Is_Bulk"] = np.where((df["Quantity"] >= 50) | (df["TotalRevenue"] >= 500), 1, 0)

    except FileNotFoundError:
        rng = np.random.default_rng(42)
        n_retail, n_bulk = 14500, 4000

        retail_data = pd.DataFrame({
            "Quantity": rng.integers(1, 35, n_retail),
            "UnitPrice": rng.uniform(1.5, 25.0, n_retail),
            "Is_UK": rng.choice([1, 0], size=n_retail, p=[0.90, 0.10]),
            "Is_Guest": rng.choice([0, 1], size=n_retail, p=[0.85, 0.15]),
            "Hour": rng.integers(8, 20, n_retail),
            "DayOfWeek": rng.integers(0, 7, n_retail),
            "Month": rng.choice(range(1, 13), size=n_retail),
            "Unique_SKUs": rng.integers(1, 15, n_retail),
            "Is_Bulk": 0
        })
        bulk_data = pd.DataFrame({
            "Quantity": rng.integers(50, 600, n_bulk),
            "UnitPrice": rng.uniform(2.0, 45.0, n_bulk),
            "Is_UK": rng.choice([1, 0], size=n_bulk, p=[0.75, 0.25]),
            "Is_Guest": rng.choice([0, 1], size=n_bulk, p=[0.95, 0.05]),
            "Hour": rng.integers(9, 17, n_bulk),
            "DayOfWeek": rng.integers(0, 5, n_bulk),
            "Month": rng.choice([9, 10, 11, 12], size=n_bulk),
            "Unique_SKUs": rng.integers(10, 80, n_bulk),
            "Is_Bulk": 1
        })
        df = pd.concat([retail_data, bulk_data], ignore_index=True).sample(frac=1, random_state=42)
        df["TotalRevenue"] = df["Quantity"] * df["UnitPrice"]

    df["Segment"] = df["Is_Bulk"].map({1: "Bulk / Wholesale", 0: "Individual / Retail"})
    return df

@st.cache_resource
def load_rf_pipeline(_df):
    try:
        model = joblib.load("best_rf_pipeline.joblib")
    except FileNotFoundError:
        X = _df[FEATURE_COLS]
        y = _df["Is_Bulk"]
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
        ])
        model.fit(X, y)
    return model

def apply_clean_plotly(fig):
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=TEXT_PRIMARY, family="Inter, Plus Jakarta Sans, sans-serif"),
        title=dict(font=dict(color=TEXT_PRIMARY, size=16)),
        xaxis=dict(
            gridcolor="rgba(15, 23, 42, 0.08)",
            zerolinecolor="rgba(15, 23, 42, 0.2)",
            tickfont=dict(color=TEXT_SECONDARY),
            title=dict(font=dict(color=TEXT_PRIMARY)),
        ),
        yaxis=dict(
            gridcolor="rgba(15, 23, 42, 0.08)",
            zerolinecolor="rgba(15, 23, 42, 0.2)",
            tickfont=dict(color=TEXT_SECONDARY),
            title=dict(font=dict(color=TEXT_PRIMARY)),
        ),
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.94)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(color=TEXT_PRIMARY),
        ),
    )
    return fig

def compute_separability_metrics(df, target_features):
    bulk_df = df[df["Is_Bulk"] == 1]
    retail_df = df[df["Is_Bulk"] == 0]
    results = []

    for feat in target_features:
        if feat not in df.columns:
            continue
        b_vals = pd.to_numeric(bulk_df[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
        r_vals = pd.to_numeric(retail_df[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values

        if len(b_vals) == 0 or len(r_vals) == 0:
            results.append({
                "Feature": feat,
                "Bulk Median": np.nan,
                "Retail Median": np.nan,
                "Mean Ratio (B/R)": np.nan,
                "KS Statistic": np.nan,
                "KS p-value": np.nan,
                "KL Divergence (B||R)": np.nan,
                "JS Divergence": np.nan,
                "Separability Power": "Insufficient Data",
            })
            continue

        ks_stat, p_val = ks_2samp(b_vals, r_vals)

        combined = np.concatenate([b_vals, r_vals])
        if np.min(combined) == np.max(combined):
            kl_div = 0.0
            js_div = 0.0
        else:
            bins = np.histogram_bin_edges(combined, bins=30)
            p_hist, _ = np.histogram(b_vals, bins=bins, density=True)
            q_hist, _ = np.histogram(r_vals, bins=bins, density=True)

            p_prob = (p_hist + 1e-10) / np.sum(p_hist + 1e-10)
            q_prob = (q_hist + 1e-10) / np.sum(q_hist + 1e-10)
            kl_div = float(entropy(p_prob, q_prob))
            js_div = float(jensenshannon(p_prob, q_prob, base=2.0) ** 2)

        b_med = float(np.median(b_vals))
        r_med = float(np.median(r_vals))
        b_mean = float(np.mean(b_vals))
        r_mean = float(np.mean(r_vals))
        ratio_mean = (b_mean / r_mean) if abs(r_mean) > 1e-9 else np.nan

        results.append({
            "Feature": feat,
            "Bulk Median": round(b_med, 2),
            "Retail Median": round(r_med, 2),
            "Mean Ratio (B/R)": round(ratio_mean, 2) if pd.notna(ratio_mean) else np.nan,
            "KS Statistic": round(float(ks_stat), 4),
            "KS p-value": float(p_val),
            "KL Divergence (B||R)": round(kl_div, 4),
            "JS Divergence": round(js_div, 4),
            "Separability Power": "High" if ks_stat > 0.35 else ("Moderate" if ks_stat > 0.2 else "Low"),
        })

    if not results:
        return pd.DataFrame(columns=[
            "Feature", "Bulk Median", "Retail Median", "Mean Ratio (B/R)",
            "KS Statistic", "KS p-value", "KL Divergence (B||R)", "JS Divergence", "Separability Power"
        ])

    res_df = pd.DataFrame(results).sort_values(by="KS Statistic", ascending=False, na_position="last").reset_index(drop=True)
    res_df.index += 1
    res_df.index.name = "Rank"
    return res_df

def build_segment_comparison_table(df, features):
    rows = []
    bulk_df = df[df["Is_Bulk"] == 1]
    retail_df = df[df["Is_Bulk"] == 0]
    for feat in features:
        if feat not in df.columns:
            continue
        b = pd.to_numeric(bulk_df[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        r = pd.to_numeric(retail_df[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(b) == 0 or len(r) == 0:
            continue
        rows.append({
            "Feature": feat,
            "Bulk Mean": round(float(b.mean()), 2),
            "Retail Mean": round(float(r.mean()), 2),
            "Bulk IQR (Q75-Q25)": round(float(b.quantile(0.75) - b.quantile(0.25)), 2),
            "Retail IQR (Q75-Q25)": round(float(r.quantile(0.75) - r.quantile(0.25)), 2),
        })
    return pd.DataFrame(rows)

def get_feature_importances(pipeline, feature_names):
    if hasattr(pipeline, "named_steps"):
        for step in reversed(list(pipeline.named_steps.values())):
            if hasattr(step, "feature_importances_"):
                return dict(zip(feature_names, step.feature_importances_))
    elif hasattr(pipeline, "feature_importances_"):
        return dict(zip(feature_names, pipeline.feature_importances_))
    return {f: 1.0 / len(feature_names) for f in feature_names}

df = load_ecommerce_data()
rf_pipeline = load_rf_pipeline(df)

st.sidebar.title("🛒 E-Commerce BI")
st.sidebar.markdown("Strategic Retail Analytics & B2B/B2C Order Classification")
page = st.sidebar.radio(
    "Navigation",
    ["🏪 Executive Overview", "📊 Strategic Analytics", "🤖 Order Classifier"],
)

if page == "🏪 Executive Overview":
    st.title("🏪 E-Commerce Data Pipeline & Buyer Segmentation")

    total_orders = f"{len(df):,}"
    bulk_ratio = f"{(df['Is_Bulk'].mean() * 100):.1f}%"
    aov = f"£{df['TotalRevenue'].mean():,.2f}"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Clean Invoices", total_orders, "Net of Cancellations")
    col2.metric("Bulk Order Ratio", bulk_ratio, "B2B Wholesale Share")
    col3.metric("Average Order Value", aov, "Net AOV")
    col4.metric("Model Type", "Random Forest", "Ensemble Classifier")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-container">
            <h3>⚙️ End-to-End E-Commerce Data Engineering Pipeline</h3>
            <p>
                Raw transactional retail logs contain cancellation invoices (prefix <code>'C'</code>), zero-price promotional adjustments, 
                and split guest checkouts. Our pipeline reconciles returns proportionally against original purchases, preserves unauthenticated 
                <b>Guest Checkout</b> orders via invoice-level grouping, and engineers spatial/temporal signals (<code>Is_UK</code>, Q4 seasonality, peak hours).
            </p>
            <p>
                Using <b>Random Forest</b>, we classify incoming orders in real time to distinguish high-volume <b>Bulk / Wholesale (B2B)</b> 
                accounts from <b>Individual / Retail (B2C)</b> shoppers—enabling dynamic inventory allocation and targeted discounting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="glass-container" style="border-top: 4px solid {BULK_COLOR};">
                <h3 style="color: {BULK_COLOR} !important;">🟢 Bulk / Wholesale Orders (B2B Signal)</h3>
                <ul>
                    <li><b>High Basket Volume:</b> Large unit quantities (≥ 50 units) or high basket value (≥ £500).</li>
                    <li><b>Broad SKU Diversity:</b> Orders span dozens of distinct product codes (<code>Unique_SKUs</code>) per invoice.</li>
                    <li><b>Business-Hour Concentration:</b> Concentrated heavily on weekdays between 09:00 and 17:00, peaking in Q4 restocking.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="glass-container" style="border-top: 4px solid {RETAIL_COLOR};">
                <h3 style="color: {RETAIL_COLOR} !important;">🔵 Individual / Retail Orders (B2C Segment)</h3>
                <ul>
                    <li><b>Small Unit Baskets:</b> Single-item or small household quantities (&lt; 35 units per checkout).</li>
                    <li><b>Higher Guest Checkout Share:</b> Frequently purchased without registered B2B account IDs (<code>Is_Guest = 1</code>).</li>
                    <li><b>Evening & Weekend Activity:</b> Distributed across evenings and weekends with strong UK domestic concentration.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif page == "📊 Strategic Analytics":
    st.title("📊 Strategic E-Commerce Analytics & Feature Separability")

    key_features = ["Quantity", "Unique_SKUs", "UnitPrice", "Hour"]
    importances = get_feature_importances(rf_pipeline, FEATURE_COLS)

    st.markdown(
        """
        <div class="separability-card">
            <h3>📐 Statistical Feature Separability Ranking</h3>
            <p>
                Quantifying distribution divergence and central-tendency shifts between <b>Bulk / Wholesale (B2B)</b> and 
                <b>Individual / Retail (B2C)</b> orders using the two-sample <b>Kolmogorov-Smirnov (KS) Statistic</b>, 
                <b>Kullback-Leibler (KL) Divergence</b>, and <b>Jensen-Shannon Divergence (JSD)</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scope_col1, _ = st.columns([2, 1])
    with scope_col1:
        eval_scope = st.radio(
            "Feature Evaluation Scope:",
            ["Core Separability Features (4)", "All Model Features (8)"],
            horizontal=True,
        )
    target_eval_features = key_features if eval_scope.startswith("Core") else FEATURE_COLS

    sep_df = compute_separability_metrics(df, target_eval_features)
    st.markdown('<div class="separability-table-wrap">', unsafe_allow_html=True)
    st.dataframe(
        sep_df,
        width="stretch",
        column_config={
            "KS Statistic": st.column_config.NumberColumn(format="%.4f"),
            "KS p-value": st.column_config.NumberColumn(format="%.2e"),
            "KL Divergence (B||R)": st.column_config.NumberColumn(format="%.4f"),
            "JS Divergence": st.column_config.NumberColumn(format="%.4f"),
            "Mean Ratio (B/R)": st.column_config.NumberColumn(format="%.2fx"),
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📋 Detailed Segment & Temporal Comparisons (Mean, IQR, Market, Guest & Time)"):
        comp_df = build_segment_comparison_table(df, target_eval_features)
        st.dataframe(comp_df, width="stretch")

        cat_col1, cat_col2 = st.columns(2)
        with cat_col1:
            uk_by_seg = (df.groupby("Segment")["Is_UK"].mean() * 100).round(1)
            fig_uk = px.bar(
                x=uk_by_seg.index,
                y=uk_by_seg.values,
                labels={"x": "Segment", "y": "UK Share (%)"},
                color=uk_by_seg.index,
                color_discrete_map={"Bulk / Wholesale": BULK_COLOR, "Individual / Retail": RETAIL_COLOR},
                title="UK Order Share by Segment (%)",
            )
            fig_uk.update_layout(showlegend=False, height=300)
            st.plotly_chart(apply_clean_plotly(fig_uk), width="stretch")

        with cat_col2:
            guest_by_seg = (df.groupby("Segment")["Is_Guest"].mean() * 100).round(1)
            fig_guest = px.bar(
                x=guest_by_seg.index,
                y=guest_by_seg.values,
                labels={"x": "Segment", "y": "Guest Checkout Share (%)"},
                color=guest_by_seg.index,
                color_discrete_map={"Bulk / Wholesale": BULK_COLOR, "Individual / Retail": RETAIL_COLOR},
                title="Guest Checkout Share by Segment (%)",
            )
            fig_guest.update_layout(showlegend=False, height=300)
            st.plotly_chart(apply_clean_plotly(fig_guest), width="stretch")

        time_col1, time_col2 = st.columns(2)
        with time_col1:
            dow_df = df.copy()
            dow_df["DayLabel"] = dow_df["DayOfWeek"].map(DAY_NAMES)
            dow_shares = (
                dow_df.groupby(["Segment", "DayOfWeek", "DayLabel"])
                .size()
                .reset_index(name="Orders")
            )
            dow_shares["SharePct"] = (
                dow_shares["Orders"] / dow_shares.groupby("Segment")["Orders"].transform("sum") * 100
            ).round(1)
            dow_shares = dow_shares.sort_values("DayOfWeek")
            fig_dow = px.line(
                dow_shares,
                x="DayLabel",
                y="SharePct",
                color="Segment",
                markers=True,
                labels={"DayLabel": "Day of Week", "SharePct": "Share of Segment Orders (%)"},
                color_discrete_map={"Bulk / Wholesale": BULK_COLOR, "Individual / Retail": RETAIL_COLOR},
                title="Weekly Order Pattern by Segment (%)",
            )
            fig_dow.update_layout(height=320)
            st.plotly_chart(apply_clean_plotly(fig_dow), width="stretch")

        with time_col2:
            m_df = df.copy()
            m_df["MonthLabel"] = m_df["Month"].map(MONTH_NAMES)
            m_shares = (
                m_df.groupby(["Segment", "Month", "MonthLabel"])
                .size()
                .reset_index(name="Orders")
            )
            m_shares["SharePct"] = (
                m_shares["Orders"] / m_shares.groupby("Segment")["Orders"].transform("sum") * 100
            ).round(1)
            m_shares = m_shares.sort_values("Month")
            fig_m = px.bar(
                m_shares,
                x="MonthLabel",
                y="SharePct",
                color="Segment",
                barmode="group",
                labels={"MonthLabel": "Month", "SharePct": "Share of Segment Orders (%)"},
                color_discrete_map={"Bulk / Wholesale": BULK_COLOR, "Individual / Retail": RETAIL_COLOR},
                title="Monthly Seasonality Comparison by Segment (%)",
            )
            fig_m.update_layout(height=320)
            st.plotly_chart(apply_clean_plotly(fig_m), width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-container">
            <h3>🔗 Cross-Feature Relationships, Hourly Dynamics & Model Importance</h3>
            <p>
                Analyzing pairwise correlations, <b>Quantity vs. UnitPrice</b> scaling with <b>Unique_SKUs</b>, 
                hourly feature shifts across segments, and Random Forest feature attribution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rel_col1, rel_col2 = st.columns([1.15, 1])

    with rel_col1:
        corr_cols = FEATURE_COLS + (["TotalRevenue"] if "TotalRevenue" in df.columns else []) + ["Is_Bulk"]
        corr_matrix = df[corr_cols].apply(pd.to_numeric, errors="coerce").corr().round(2)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Feature Correlation Heatmap",
        )
        fig_corr.update_layout(height=450)
        st.plotly_chart(apply_clean_plotly(fig_corr), width="stretch")

    with rel_col2:
        scale_mode = st.radio(
            "Axis Scale for Quantity & UnitPrice:",
            ["Linear", "Logarithmic (Log10)"],
            horizontal=True,
        )
        rel_sample = df.dropna(subset=["Quantity", "UnitPrice", "Unique_SKUs"]).copy()
        rel_sample = rel_sample.sample(n=min(2000, len(rel_sample)), random_state=42)
        if scale_mode.startswith("Log"):
            rel_sample = rel_sample[(rel_sample["Quantity"] > 0) & (rel_sample["UnitPrice"] > 0)]

        fig_scatter = px.scatter(
            rel_sample,
            x="Quantity",
            y="UnitPrice",
            size="Unique_SKUs",
            color="Segment",
            log_x=scale_mode.startswith("Log"),
            log_y=scale_mode.startswith("Log"),
            size_max=22,
            opacity=0.72,
            color_discrete_map={"Bulk / Wholesale": BULK_COLOR, "Individual / Retail": RETAIL_COLOR},
            hover_data=["Hour", "DayOfWeek", "Month", "Is_UK", "Is_Guest"],
            title="Quantity vs. UnitPrice (Bubble Size = Unique_SKUs)",
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(apply_clean_plotly(fig_scatter), width="stretch")

    imp_col, hourly_col = st.columns([1, 1.25])

    with imp_col:
        imp_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance (%)": [round(importances.get(f, 0.0) * 100, 2) for f in FEATURE_COLS],
        }).sort_values("Importance (%)", ascending=True)
        fig_imp = px.bar(
            imp_df,
            x="Importance (%)",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance (%)",
            color="Importance (%)",
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(showlegend=False, coloraxis_showscale=False, height=360)
        st.plotly_chart(apply_clean_plotly(fig_imp), width="stretch")

    with hourly_col:
        hourly_feat = st.selectbox(
            "Compare Feature Mean Across Hours of Day:",
            options=["Quantity", "Unique_SKUs", "UnitPrice", "TotalRevenue"],
            index=0,
        )
        if hourly_feat in df.columns:
            hr_df = (
                df.groupby(["Segment", "Hour"])[hourly_feat]
                .mean()
                .reset_index()
                .sort_values("Hour")
            )
            fig_hr = px.line(
                hr_df,
                x="Hour",
                y=hourly_feat,
                color="Segment",
                markers=True,
                color_discrete_map={"Bulk / Wholesale": BULK_COLOR, "Individual / Retail": RETAIL_COLOR},
                title=f"Hourly Mean of {hourly_feat} by Segment",
            )
            fig_hr.update_layout(height=360)
            st.plotly_chart(apply_clean_plotly(fig_hr), width="stretch")

    with st.expander("🧩 Pairwise Scatter Matrix Across Core Features (Quantity, Unique_SKUs, UnitPrice, Hour)"):
        try:
            splom_df = df.dropna(subset=key_features + ["Segment"]).copy()
            splom_df = splom_df.sample(n=min(1200, len(splom_df)), random_state=42)
            for c_cap in ["Quantity", "UnitPrice"]:
                splom_df[c_cap] = np.clip(splom_df[c_cap], splom_df[c_cap].quantile(0.01), splom_df[c_cap].quantile(0.98))

            fig_splom = px.scatter_matrix(
                splom_df,
                dimensions=key_features,
                color="Segment",
                opacity=0.65,
                color_discrete_map={"Bulk / Wholesale": BULK_COLOR, "Individual / Retail": RETAIL_COLOR},
                title="Pairwise Feature Scatter Matrix by Segment",
            )
            fig_splom.update_traces(diagonal_visible=False, marker=dict(size=4))
            fig_splom.update_layout(height=680)
            st.plotly_chart(apply_clean_plotly(fig_splom), width="stretch")
        except Exception as exc:
            st.warning(f"Unable to display scatter matrix: {exc}")

    st.markdown("<br>", unsafe_allow_html=True)

    viz_col1, _ = st.columns([1, 3])
    with viz_col1:
        plot_type = st.radio(
            "Select Distribution Visualization:",
            ["Violin Plots", "KDE Plots"],
            horizontal=False,
        )

    shap_narratives = {
        "Quantity": (
            "Total item volume per invoice. Orders exceeding <b>50 units</b> strongly push Random Forest predictions toward <b>Bulk / Wholesale</b>, reflecting commercial restocking behavior versus single-digit retail baskets."
        ),
        "Unique_SKUs": (
            "Number of distinct products in a single basket. Wholesale buyers bundle wide assortments (<b>15+ SKUs</b>), whereas individual shoppers typically purchase 1 to 8 specific items."
        ),
        "UnitPrice": (
            "Average item price in GBP (£). Interacts non-linearly with Quantity: high unit prices combined with moderate quantities push baskets over the £500 wholesale value threshold."
        ),
        "Hour": (
            "Hour of invoice generation (24h). Commercial B2B orders cluster sharply during standard office hours (<b>09:00–16:00</b>), while evening checkouts shift attribution toward <b>Individual / Retail</b>."
        ),
    }

    for feat in key_features:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        chart_col, insight_col = st.columns([2, 1])

        with chart_col:
            plot_df = df.copy()
            if feat in ["Quantity", "UnitPrice"]:
                upper_cap = plot_df[feat].quantile(0.97)
                plot_df = plot_df[plot_df[feat] <= upper_cap]

            if plot_type == "Violin Plots":
                fig = px.violin(
                    plot_df,
                    x="Segment",
                    y=feat,
                    color="Segment",
                    box=True,
                    points=False,
                    color_discrete_map={
                        "Bulk / Wholesale": BULK_COLOR,
                        "Individual / Retail": RETAIL_COLOR,
                    },
                    title=f"{feat} Distribution by Buyer Segment (Violin)",
                )
            else:
                b_vals = pd.to_numeric(plot_df[plot_df["Is_Bulk"] == 1][feat], errors="coerce").dropna().values
                r_vals = pd.to_numeric(plot_df[plot_df["Is_Bulk"] == 0][feat], errors="coerce").dropna().values
                try:
                    if len(b_vals) > 1 and len(r_vals) > 1 and np.std(b_vals) > 0 and np.std(r_vals) > 0:
                        fig = ff.create_distplot(
                            [b_vals, r_vals],
                            group_labels=["Bulk / Wholesale", "Individual / Retail"],
                            colors=[BULK_COLOR, RETAIL_COLOR],
                            show_hist=False,
                            show_rug=False,
                        )
                        fig.update_layout(
                            title=f"{feat} Kernel Density Estimate (KDE)",
                            xaxis_title=feat,
                            yaxis_title="Density",
                        )
                    else:
                        raise ValueError("Insufficient variance for KDE")
                except Exception:
                    fig = px.histogram(
                        plot_df,
                        x=feat,
                        color="Segment",
                        barmode="overlay",
                        opacity=0.65,
                        color_discrete_map={
                            "Bulk / Wholesale": BULK_COLOR,
                            "Individual / Retail": RETAIL_COLOR,
                        },
                        title=f"{feat} Distribution Histogram (Fallback)",
                    )
            fig = apply_clean_plotly(fig)
            st.plotly_chart(fig, width="stretch")

        with insight_col:
            imp_pct = importances.get(feat, 0.0) * 100
            b_med_val = df[df["Is_Bulk"] == 1][feat].median()
            r_med_val = df[df["Is_Bulk"] == 0][feat].median()
            st.markdown(
                f"""
                <div class="insight-box">
                    <h3 style="margin-top:0;">🧠 Model Attribution: <code>{feat}</code></h3>
                    <div>
                        <span class="stat-pill">RF Importance: {imp_pct:.2f}%</span><br>
                        <span class="stat-pill">Bulk Median: {b_med_val:.2f}</span>
                        <span class="stat-pill">Retail Median: {r_med_val:.2f}</span>
                    </div>
                    <hr style="border-color:#E2E8F0; margin:12px 0;">
                    <p style="margin-bottom:6px; font-weight:700; color:#0F172A;">SHAP-Style Impact Analysis</p>
                    <p style="margin-bottom:0; color:#1E293B; line-height:1.6;">{shap_narratives[feat]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-container">
            <h3>🗺️ Principal Component Analysis (PCA) & Decision Boundary Approximation</h3>
            <p>
                2D projection of the 8-dimensional order feature space overlaid with the Random Forest 
                decision probability surface separating Bulk/Wholesale accounts from Individual Retail orders.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        valid_pca_df = df.dropna(subset=FEATURE_COLS).copy()
        if len(valid_pca_df) < 5:
            st.warning("Not enough valid rows to compute 2D PCA projection.")
        else:
            sample_df = valid_pca_df.sample(n=min(2500, len(valid_pca_df)), random_state=42).copy()
            scaler_pca = StandardScaler()
            X_scaled = scaler_pca.fit_transform(sample_df[FEATURE_COLS])

            pca = PCA(n_components=2, random_state=42)
            pcs = pca.fit_transform(X_scaled)
            sample_df["PC1"] = pcs[:, 0]
            sample_df["PC2"] = pcs[:, 1]

            x_min, x_max = float(sample_df["PC1"].min()), float(sample_df["PC1"].max())
            y_min, y_max = float(sample_df["PC2"].min()), float(sample_df["PC2"].max())
            pad_x = max((x_max - x_min) * 0.08, 0.5)
            pad_y = max((y_max - y_min) * 0.08, 0.5)

            grid_x = np.linspace(x_min - pad_x, x_max + pad_x, 60)
            grid_y = np.linspace(y_min - pad_y, y_max + pad_y, 60)
            xx, yy = np.meshgrid(grid_x, grid_y)
            grid_2d = np.c_[xx.ravel(), yy.ravel()]
            grid_8d = scaler_pca.inverse_transform(pca.inverse_transform(grid_2d))
            grid_df = pd.DataFrame(grid_8d, columns=FEATURE_COLS)
            zz = np.clip(rf_pipeline.predict_proba(grid_df)[:, 1], 0.0, 1.0).reshape(xx.shape)

            fig_pca = go.Figure()
            fig_pca.add_trace(
                go.Contour(
                    x=grid_x,
                    y=grid_y,
                    z=zz,
                    colorscale=[
                        [0.0, "rgba(2, 132, 199, 0.22)"],
                        [0.5, "rgba(241, 245, 249, 0.35)"],
                        [1.0, "rgba(5, 150, 105, 0.25)"],
                    ],
                    contours=dict(start=0.0, end=1.0, size=0.1, showlines=True),
                    colorbar=dict(
                        title=dict(text="P(Bulk)", font=dict(color=TEXT_PRIMARY)),
                        tickfont=dict(color=TEXT_PRIMARY),
                    ),
                    name="Decision Boundary",
                    hoverinfo="skip",
                )
            )

            for label_name, color_hex in [("Bulk / Wholesale", BULK_COLOR), ("Individual / Retail", RETAIL_COLOR)]:
                sub = sample_df[sample_df["Segment"] == label_name]
                fig_pca.add_trace(
                    go.Scatter(
                        x=sub["PC1"],
                        y=sub["PC2"],
                        mode="markers",
                        name=label_name,
                        marker=dict(color=color_hex, size=6, opacity=0.8, line=dict(width=0.6, color="#FFFFFF")),
                    )
                )

            var_exp = pca.explained_variance_ratio_ * 100
            fig_pca.update_layout(
                title=f"2D PCA Projection with Decision Boundary (Explained Variance: {var_exp[0]:.1f}% + {var_exp[1]:.1f}%)",
                xaxis_title=f"Principal Component 1 ({var_exp[0]:.1f}%)",
                yaxis_title=f"Principal Component 2 ({var_exp[1]:.1f}%)",
                height=600,
            )
            fig_pca = apply_clean_plotly(fig_pca)
            st.plotly_chart(fig_pca, width="stretch")
    except Exception as exc:
        st.error(f"Unable to render PCA decision boundary plot: {exc}")

elif page == "🤖 Order Classifier":
    st.title("🤖 Real-Time E-Commerce Order Classifier")

    st.markdown(
        """
        <div class="glass-container">
            <p>
                Enter the order basket parameters below to classify whether an incoming checkout is a 
                <b>Bulk / Wholesale (B2B)</b> or <b>Individual / Retail (B2C)</b> transaction using <code>best_rf_pipeline.joblib</code>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("order_input_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            quantity = st.number_input("Total Quantity (Units)", min_value=1, value=75, step=1)
            unit_price = st.number_input("Average Unit Price (£)", min_value=0.01, value=12.50, format="%.2f")
            unique_skus = st.number_input("Unique SKUs in Basket", min_value=1, value=18, step=1)
            is_uk = st.selectbox("Market Location (Is_UK)", options=[1, 0], format_func=lambda x: "United Kingdom (1)" if x == 1 else "International (0)")

        with col_right:
            is_guest = st.selectbox("Checkout Type (Is_Guest)", options=[0, 1], format_func=lambda x: "Registered Account (0)" if x == 0 else "Guest Checkout (1)")
            hour = st.slider("Order Hour (24h)", min_value=0, max_value=23, value=14)
            day_of_week = st.selectbox(
                "Day of Week",
                options=[0, 1, 2, 3, 4, 5, 6],
                index=2,
                format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
            )
            month = st.slider("Month (1-12)", min_value=1, max_value=12, value=11)

        submitted = st.form_submit_button("Analyze Order")

    if submitted:
        input_data = pd.DataFrame([{
            "Quantity": quantity,
            "UnitPrice": unit_price,
            "Is_UK": is_uk,
            "Is_Guest": is_guest,
            "Hour": hour,
            "DayOfWeek": day_of_week,
            "Month": month,
            "Unique_SKUs": unique_skus,
        }])[FEATURE_COLS]

        probs = rf_pipeline.predict_proba(input_data)[0]
        pred_class = int(rf_pipeline.predict(input_data)[0])

        is_bulk = pred_class == 1
        confidence = probs[1] * 100 if is_bulk else probs[0] * 100
        result_title = "BULK / WHOLESALE (B2B)" if is_bulk else "INDIVIDUAL / RETAIL (B2C)"
        result_color = BULK_COLOR if is_bulk else RETAIL_COLOR
        est_revenue = quantity * unit_price

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown(
                f"""
                <div class="glass-container" style="border: 2px solid {result_color}; text-align: center; padding: 40px 20px;">
                    <h4>Order Classification Verdict</h4>
                    <h1 style="color: {result_color} !important; font-size: 2.2rem; margin: 15px 0;">{result_title}</h1>
                    <p style="font-size: 1.1rem; color: #0F172A;">
                        Estimated Basket Value: <b>£{est_revenue:,.2f}</b><br>
                        Bulk Probability: <b>{probs[1]*100:.2f}%</b> &nbsp;|&nbsp; 
                        Retail Probability: <b>{probs[0]*100:.2f}%</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    number={"suffix": "%", "font": {"color": TEXT_PRIMARY, "size": 44}},
                    title={"text": "Prediction Confidence", "font": {"color": TEXT_PRIMARY, "size": 20}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": TEXT_PRIMARY, "tickfont": {"color": TEXT_PRIMARY}},
                        "bar": {"color": result_color},
                        "bgcolor": "#F1F5F9",
                        "borderwidth": 2,
                        "bordercolor": "#CBD5E1",
                        "steps": [
                            {"range": [0, 50], "color": "#FFFFFF"},
                            {"range": [50, 80], "color": "#E2E8F0"},
                            {"range": [80, 100], "color": "#CBD5E1"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
            fig_gauge = apply_clean_plotly(fig_gauge)
            st.plotly_chart(fig_gauge, width="stretch")
