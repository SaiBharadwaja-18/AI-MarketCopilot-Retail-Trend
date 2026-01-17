import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import timedelta


# ---------------- Streamlit Config ----------------
st.set_page_config(
    page_title="AI MarketCopilot - Retail Trend Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Global Styling ----------------
st.markdown("""
<style>
/* Global app */
.stApp {
    background: radial-gradient(circle at top left, #eff6ff 0, #ffffff 40%, #f9fafb 100%);
    color: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
}

/* Main container */
div.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Remove default header style */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Titles */
h1, h2, h3, h4 {
    color: #020617 !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
}
h1 {
    font-size: 2.3rem !important;
}
h2 {
    font-size: 1.4rem !important;
}

/* Subtitle */
.app-subtitle {
    font-size: 0.94rem;
    color: #64748b;
    margin-bottom: 1.1rem;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%) !important;
    border: 1px solid #e2e8f0 !important;
    padding: 12px 14px !important;
    border-radius: 18px !important;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06) !important;
}
div[data-testid="stMetricLabel"] > div {
    color: #64748b !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stMetricValue"] > div {
    color: #0f172a !important;
    font-size: 23px !important;
    font-weight: 800 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617 !important;
    border-right: 1px solid #020617 !important;
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f9fafb !important;
}
section[data-testid="stSidebar"] .small-note {
    color: #9ca3af !important;
}
section[data-testid="stSidebar"] .stSlider > div[data-baseweb="slider"] {
    color: #e5e7eb !important;
}

/* Sidebar badge */
.sidebar-pill {
    border-radius: 12px;
    padding: 0.55rem 0.7rem;
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(59,130,246,0.06));
    border: 1px solid rgba(148,163,184,0.45);
    font-size: 0.8rem;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 10px 22px rgba(15,23,42,0.05);
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #475569 !important;
    border-bottom: 1px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #0f172a !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* Section headers inside tabs */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
}
.section-pill {
    font-size: 0.74rem;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    border: 1px solid #bfdbfe;
    background: #eff6ff;
    color: #1d4ed8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Small helper text */
.small-note {
    font-size: 13px;
    color: #64748b;
}

/* Plot containers */
.css-1y4p8pa, .stPlotlyChart, .stImage {
    border-radius: 14px !important;
}

/* Horizontal rule */
hr {
    margin-top: 1rem;
    margin-bottom: 1.1rem;
    border-color: #e5e7eb;
}

/* =========================================================
   ✅ FIX: st.code blocks inside sidebar (Model Artifacts/Dataset)
   ========================================================= */
section[data-testid="stSidebar"] pre {
    background: #0b1220 !important;
    border: 1px solid rgba(148,163,184,0.25) !important;
    color: #e5e7eb !important;
    border-radius: 14px !important;
}

section[data-testid="stSidebar"] pre code {
    color: #e5e7eb !important;
}

section[data-testid="stSidebar"] div[data-testid="stCodeBlock"] {
    border-radius: 14px !important;
    overflow: hidden !important;
}
</style>

""", unsafe_allow_html=True)


# ---------------- Helper Functions ----------------
def calc_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100


@st.cache_data
def load_data_from_project():
    dataset_path = "Online_Retail.xlsx"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Keep Online_Retail.xlsx in same folder as app.py"
        )

    df = pd.read_excel(dataset_path)
    return df


def clean_data(df):
    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Description"] = df["Description"].astype(str)

    df["is_cancelled"] = df["InvoiceNo"].astype(str).str.startswith("C")
    df = df[~df["is_cancelled"]]

    df = df.dropna(subset=["CustomerID", "Description"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    df["Date"] = df["InvoiceDate"].dt.date
    df["Date"] = pd.to_datetime(df["Date"])

    return df


@st.cache_resource
def load_model():
    model_path = os.path.join("models", "xgb_model.pkl")
    feat_path = os.path.join("models", "feature_cols.pkl")
    meta_path = os.path.join("models", "train_meta.pkl")

    for p in [model_path, feat_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    model = joblib.load(model_path)
    feature_cols = joblib.load(feat_path)
    meta = joblib.load(meta_path)
    return model, feature_cols, meta


def make_features(daily_sales_df):
    ds_ml = daily_sales_df.copy()

    ds_ml["dow"] = ds_ml["ds"].dt.dayofweek
    ds_ml["month"] = ds_ml["ds"].dt.month
    ds_ml["day"] = ds_ml["ds"].dt.day
    ds_ml["is_weekend"] = (ds_ml["dow"] >= 5).astype(int)

    for lag in [1, 7, 14, 28]:
        ds_ml[f"lag_{lag}"] = ds_ml["y"].shift(lag)

    for window in [7, 14, 28]:
        ds_ml[f"roll_mean_{window}"] = ds_ml["y"].rolling(window).mean()
        ds_ml[f"roll_std_{window}"] = ds_ml["y"].rolling(window).std()

    ds_ml = ds_ml.dropna().reset_index(drop=True)
    return ds_ml


def build_forecast_inference(df_clean):
    model, feature_cols, meta = load_model()

    daily_sales = df_clean.groupby("Date")["Revenue"].sum().reset_index()
    daily_sales = daily_sales.rename(columns={"Date": "ds", "Revenue": "y"})
    daily_sales["ds"] = pd.to_datetime(daily_sales["ds"])
    daily_sales = daily_sales.sort_values("ds").reset_index(drop=True)

    ds_ml = make_features(daily_sales)

    split = int(len(ds_ml) * 0.8)
    test_ml = ds_ml.iloc[split:].copy()
    y_test = test_ml["y"]

    X_test = test_ml[feature_cols]
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    metrics_raw = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "MAPE": calc_mape(y_test, y_pred)
    }

    eval_cap = np.quantile(y_test, 0.99)
    y_test_eval = np.minimum(y_test, eval_cap)
    y_pred_eval = np.minimum(y_pred, eval_cap)

    metrics_robust = {
        "MAE": mean_absolute_error(y_test_eval, y_pred_eval),
        "RMSE": np.sqrt(mean_squared_error(y_test_eval, y_pred_eval)),
        "R2": r2_score(y_test_eval, y_pred_eval),
        "MAPE": calc_mape(y_test_eval, y_pred_eval)
    }

    return daily_sales, test_ml, y_test, y_pred, metrics_raw, metrics_robust


def decision_support(test_ml, y_test, y_pred, growth_thr=8.0, risk_thr=70.0):
    rec_df = test_ml[["ds"]].copy()
    rec_df["ActualRevenue"] = y_test.values
    rec_df["PredictedRevenue"] = y_pred

    rec_df["Volatility7"] = rec_df["ActualRevenue"].rolling(7).std()
    rec_df["Volatility7"] = rec_df["Volatility7"].fillna(rec_df["Volatility7"].median())

    vmin, vmax = rec_df["Volatility7"].min(), rec_df["Volatility7"].max()
    rec_df["RiskIndex"] = 100 * (rec_df["Volatility7"] - vmin) / (vmax - vmin + 1e-9)

    rec_df["PredGrowthPct"] = rec_df["PredictedRevenue"].pct_change() * 100
    rec_df["PredGrowthPct"] = rec_df["PredGrowthPct"].fillna(0)

    def recommend(row):
        if row["RiskIndex"] >= risk_thr:
            return "HIGH RISK – Stock Carefully"
        elif row["PredGrowthPct"] >= growth_thr:
            return "STOCK UP + PROMOTE"
        elif row["PredGrowthPct"] <= -growth_thr:
            return "REDUCE STOCK"
        else:
            return "NORMAL STOCK"

    rec_df["Recommendation"] = rec_df.apply(recommend, axis=1)
    return rec_df


def rfm_segmentation(df_clean, k=4):
    snapshot_date = df_clean["InvoiceDate"].max() + timedelta(days=1)

    rfm = df_clean.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "Revenue": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    rfm_model = rfm.copy()
    rfm_model["Monetary"] = np.log1p(rfm_model["Monetary"])

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_model[["Recency", "Frequency", "Monetary"]])

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(rfm_scaled)

    persona_map = {
        3: "VIP Champions",
        2: "Loyal Customers",
        0: "Regular Customers",
        1: "Lost / At-Risk"
    }
    action_map = {
        "VIP Champions": "Premium support + early access + exclusive drops",
        "Loyal Customers": "Loyalty rewards + cross-sell bundles + retention offers",
        "Regular Customers": "Personalized promos + seasonal offers",
        "Lost / At-Risk": "Win-back discounts + reactivation campaigns"
    }

    rfm["Persona"] = rfm["Cluster"].map(persona_map).fillna("Other")
    rfm["RecommendedAction"] = rfm["Persona"].map(action_map).fillna("General offers")

    persona_stats = rfm.groupby("Persona").agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        TotalRevenue=("Monetary", "sum")
    ).sort_values("TotalRevenue", ascending=False)

    return rfm, persona_stats


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown(
        '<p class="small-note">Model and dataset are loaded from the application environment.</p>',
        unsafe_allow_html=True
    )

    growth_thr = st.slider("Growth threshold (%)", 5, 15, 8)
    risk_thr = st.slider("Risk threshold (Risk Index)", 50, 90, 70)

    st.markdown("---")
    st.markdown("#### Deployment")
    st.markdown(
        '<div class="sidebar-pill">XGBoost forecasting model served via Streamlit</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("#### Model Artifacts")
    st.code("models/xgb_model.pkl\nmodels/feature_cols.pkl\nmodels/train_meta.pkl")

    st.markdown("#### Dataset")
    st.code("Online_Retail.xlsx")


# ---------------- Header ----------------
st.markdown(
    '<div class="section-header"><span class="section-pill">AI MarketCopilot</span></div>',
    unsafe_allow_html=True
)
st.title("Retail Revenue Trend & Customer Intelligence")

st.markdown(
    '<p class="app-subtitle">'
    'Forecast daily revenue, quantify risk, optimize inventory decisions, and segment customers using RFM and clustering.'
    '</p>',
    unsafe_allow_html=True
)


# ---------------- Load pipeline ----------------
df = load_data_from_project()
df_clean = clean_data(df)

daily_sales, test_ml, y_test, y_pred, metrics_raw, metrics_robust = build_forecast_inference(df_clean)

# ---------------- Top KPIs ----------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows (Raw)", f"{df.shape[0]:,}")
k2.metric("Rows (Cleaned)", f"{df_clean.shape[0]:,}")
k3.metric("Unique Customers", f"{df_clean['CustomerID'].nunique():,}")
k4.metric("Countries", f"{df_clean['Country'].nunique():,}")
k5.metric("Total Revenue", f"£{df_clean['Revenue'].sum():,.0f}")

st.markdown("---")


# ---------------- Tabs ----------------
tabs = st.tabs([
    "📌 Data",
    "📈 Trend & Forecast",
    "🧠 Recommendations",
    "👥 Segmentation"
])


# TAB 1 - Data
with tabs[0]:
    st.markdown(
        '<div class="section-header"><span class="section-pill">Data</span>'
        '<h2>Dataset Overview</h2></div>',
        unsafe_allow_html=True
    )
    st.dataframe(df_clean.head(50), use_container_width=True)


# TAB 2 - Forecasting visuals
with tabs[1]:
    st.markdown(
        '<div class="section-header"><span class="section-pill">Forecast</span>'
        '<h2>Revenue Trend and Model Performance</h2></div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Robust Metrics (Spike‑capped)")
        a, b, c, d = st.columns(4)
        a.metric("MAE", f"{metrics_robust['MAE']:.2f}")
        b.metric("RMSE", f"{metrics_robust['RMSE']:.2f}")
        c.metric("R²", f"{metrics_robust['R2']:.4f}")
        d.metric("MAPE", f"{metrics_robust['MAPE']:.2f}%")

    with c2:
        st.markdown("##### Raw Metrics (Full scale)")
        a, b, c, d = st.columns(4)
        a.metric("MAE", f"{metrics_raw['MAE']:.2f}")
        b.metric("RMSE", f"{metrics_raw['RMSE']:.2f}")
        c.metric("R²", f"{metrics_raw['R2']:.4f}")
        d.metric("MAPE", f"{metrics_raw['MAPE']:.2f}%")

    fig1 = plt.figure(figsize=(14, 4))
    plt.plot(daily_sales["ds"], daily_sales["y"], color="#2563eb")
    plt.title("Daily Revenue Trend")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.grid(alpha=0.15)
    st.pyplot(fig1)

    eval_cap = np.quantile(y_test, 0.99)
    y_test_eval = np.minimum(y_test, eval_cap)
    y_pred_eval = np.minimum(y_pred, eval_cap)

    fig2 = plt.figure(figsize=(14, 4))
    plt.plot(test_ml["ds"], y_test_eval, label="Actual (capped)", color="#0f172a")
    plt.plot(test_ml["ds"], y_pred_eval, label="Predicted (capped)", color="#22c55e")
    plt.title("Actual vs Predicted Revenue (Robust View)")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(alpha=0.15)
    st.pyplot(fig2)


# TAB 3 - Decision Support
with tabs[2]:
    st.markdown(
        '<div class="section-header"><span class="section-pill">Decisions</span>'
        '<h2>Inventory and Promotion Recommendations</h2></div>',
        unsafe_allow_html=True
    )

    rec_df = decision_support(test_ml, y_test, y_pred, growth_thr=growth_thr, risk_thr=risk_thr)

    left, right = st.columns([1, 2])
    with left:
        st.markdown("##### Recommendation Mix")
        st.dataframe(
            rec_df["Recommendation"].value_counts().reset_index()
            .rename(columns={"index": "Recommendation", "Recommendation": "Days"}),
            use_container_width=True
        )

    with right:
        st.markdown("##### Recent Daily Recommendations")
        st.dataframe(rec_df.tail(25), use_container_width=True)

    st.markdown("##### Top Days to Stock Up / Promote")
    best_days = rec_df.sort_values(["PredictedRevenue", "RiskIndex"], ascending=[False, True]).head(10)
    st.dataframe(
        best_days[["ds", "PredictedRevenue", "PredGrowthPct", "RiskIndex", "Recommendation"]],
        use_container_width=True
    )

    fig3 = plt.figure(figsize=(12, 3.5))
    plt.hist(rec_df["RiskIndex"], bins=20, color="#f97316")
    plt.title("Risk Index Distribution")
    plt.xlabel("Risk Index")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.15)
    st.pyplot(fig3)


# TAB 4 - Segmentation
with tabs[3]:
    st.markdown(
        '<div class="section-header"><span class="section-pill">Customers</span>'
        '<h2>Customer Segmentation (RFM + KMeans)</h2></div>',
        unsafe_allow_html=True
    )

    rfm, persona_stats = rfm_segmentation(df_clean, k=4)

    c1, c2 = st.columns([1.35, 1])
    with c1:
        st.markdown("##### Persona Summary")
        st.dataframe(persona_stats, use_container_width=True)
    with c2:
        fig4 = plt.figure(figsize=(10, 4))
        plt.bar(persona_stats.index, persona_stats["TotalRevenue"].values, color="#6366f1")
        plt.title("Revenue Contribution by Persona")
        plt.xlabel("Persona")
        plt.ylabel("Revenue")
        plt.xticks(rotation=20)
        plt.grid(axis="y", alpha=0.15)
        st.pyplot(fig4)

    st.markdown("##### Sample Customer Records")
    st.dataframe(
        rfm[["CustomerID", "Recency", "Frequency", "Monetary", "Persona", "RecommendedAction"]].head(30),
        use_container_width=True
    )