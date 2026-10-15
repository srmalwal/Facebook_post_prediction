import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Facebook Live | K-Means Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancement
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stSidebarUserContent"] {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("Live.csv")
    except FileNotFoundError:
        return None

df_raw = load_dataset()

# Title Header
st.title("📊 K means Clustering")
st.caption("Perform unsupervised machine learning to group Facebook posts based on audience interaction metrics.")

if df_raw is None:
    st.error("⚠️ **Dataset Not Found:** Please ensure `Live.csv` is located in the same directory as `app.py`.")
    st.stop()

# Features used for clustering
FEATURES = [
    "num_reactions", "num_comments", "num_shares",
    "num_likes", "num_loves", "num_wows",
    "num_hahas", "num_sads", "num_angrys"
]

# Check missing columns
missing_cols = [col for col in FEATURES if col not in df_raw.columns]
if missing_cols:
    st.error(f"⚠️ Missing expected feature columns in dataset: `{', '.join(missing_cols)}`")
    st.stop()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Model Parameters")
k_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=8, value=3, step=1)
st.sidebar.markdown("---")
st.sidebar.info("Adjusting **k** updates the KMeans model fit, metric summaries, and PCA visualisations in real-time.")

# ---------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------
X_raw = df_raw[FEATURES].copy()
X_log = np.log1p(X_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

df_processed = df_raw.copy()
df_processed["Cluster"] = [f"Cluster {c}" for c in cluster_labels]

# ---------------------------------------------------------
# Main Application Views
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Overview", 
    "📈 Cluster Profiling", 
    "🗺️ PCA Projection", 
    "🔮 Predict Cluster"
])

# TAB 1: OVERVIEW
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts", f"{len(df_raw):,}")
    col2.metric("Features Evaluated", len(FEATURES))
    col3.metric("Selected Clusters (k)", k_clusters)

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)

    if "status_type" in df_raw.columns:
        st.subheader("Content Type Distribution")
        fig_type = px.bar(
            df_raw["status_type"].value_counts().reset_index(),
            x="status_type", 
            y="count",
            labels={"status_type": "Status Type", "count": "Count"},
            color="status_type",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_type.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_type, use_container_width=True)

# TAB 2: CLUSTER PROFILING
with tab2:
    st.subheader("Cluster Distribution")
    counts = df_processed["Cluster"].value_counts().reset_index()
    counts.columns = ["Cluster", "Post Count"]
    
    fig_dist = px.bar(
        counts, 
        x="Cluster", 
        y="Post Count", 
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_dist.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Mean Engagement Metrics per Cluster")
    cluster_means = df_processed.groupby("Cluster")[FEATURES].mean().reset_index()
    st.dataframe(
        cluster_means.style.highlight_max(axis=0, color="#d1e7dd"), 
        use_container_width=True
    )

# TAB 3: PCA VISUALIZATION
with tab3:
    st.subheader("2D Interactive Principal Component Analysis")
    st.caption("Dimensionality reduction via PCA allows linear projection of high-dimensional engagement metrics into 2D space.")

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(pca_coords, columns=["PCA 1", "PCA 2"])
    pca_df["Cluster"] = df_processed["Cluster"]
    if "status_type" in df_processed.columns:
        pca_df["Type"] = df_processed["status_type"]

    fig_pca = px.scatter(
        pca_df,
        x="PCA 1",
        y="PCA 2",
        color="Cluster",
        hover_data=["Type"] if "status_type" in df_processed.columns else None,
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_pca.update_layout(height=550)
    st.plotly_chart(fig_pca, use_container_width=True)

# TAB 4: PREDICTION INFERENCE
with tab4:
    st.subheader("Classify Sample Post Metrics")
    st.markdown("Enter custom post engagement values below to determine which cluster it maps onto.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            r_react = st.number_input("Reactions", min_value=0, value=150)
            r_comm = st.number_input("Comments", min_value=0, value=12)
            r_share = st.number_input("Shares", min_value=0, value=3)

        with c2:
            r_like = st.number_input("Likes", min_value=0, value=130)
            r_love = st.number_input("Loves", min_value=0, value=15)
            r_wow = st.number_input("Wows", min_value=0, value=2)

        with c3:
            r_haha = st.number_input("Hahas", min_value=0, value=1)
            r_sad = st.number_input("Sads", min_value=0, value=0)
            r_angry = st.number_input("Angrys", min_value=0, value=0)

        submit = st.form_submit_button("Predict Cluster Assignment", use_container_width=True)

    if submit:
        user_data = pd.DataFrame([{
            "num_reactions": r_react, "num_comments": r_comm, "num_shares": r_share,
            "num_likes": r_like, "num_loves": r_love, "num_wows": r_wow,
            "num_hahas": r_haha, "num_sads": r_sad, "num_angrys": r_angry
        }])[FEATURES]

        user_log = np.log1p(user_data)
        user_scaled = scaler.transform(user_log)
        predicted_cluster = kmeans.predict(user_scaled)[0]

        st.success(f"🎯 **Predicted Label:** This post maps to **Cluster {predicted_cluster}**")