import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Credit Risk XAI Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# COLOR PALETTES (Matching Thesis)
# ==========================================
METHOD_COLORS = {
    'SHAP': '#1f77b4',
    'Banzhaf': '#ff7f0e',
    'Myerson': '#2ca02c',
    'Owen-Domain': '#d62728',
    'Owen-Data': '#9467bd',
    'Owen-Model': '#8c564b',
    'R-Myerson': '#17becf'
}

# ==========================================
# SIDEBAR & DATA LOADING
# ==========================================
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("Upload a results CSV generated from the methodology pipeline to visualize the Accuracy-Interpretability trade-off.")

uploaded_file = st.sidebar.file_uploader("Upload Results CSV", type=["csv"])

if uploaded_file is None:
    st.title("📊 Coalition-Aware XAI Dashboard")
    st.markdown("""
    ### Welcome to the Credit Risk Explainability Explorer!
    Please upload a results CSV file from the sidebar to begin.
    
    **Compatible Datasets:**
    * `coursera_loans_results_7methods.csv` (1% Default)
    * `LC66_results_7methods_noleak.csv` (4% Default)
    * `LC10pcdefaultresults.csv` (10% Default)
    * `taiwan_results_7methods_S.csv` (22% Default)
    * `german_results_7methods.csv` (30% Default)
    """)
    st.stop()

# Read the uploaded data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

df = load_data(uploaded_file)

# Dynamic Title based on upload
st.title(f"📊 Dashboard: `{uploaded_file.name}`")
st.markdown("---")

# ==========================================
# MAIN DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Leaderboard", 
    "📈 Method Comparison", 
    "🎯 Pareto Trade-off", 
    "🔥 Heatmaps",
    "🗄️ Raw Data"
])

# ------------------------------------------
# TAB 1: LEADERBOARD
# ------------------------------------------
with tab1:
    st.subheader("Top 10 Configurations by S(α=0.5)")
    st.markdown("This table highlights the best combinations of Model, Sampler, and XAI Method that balance both predictive power (AUC) and Interpretability (I).")
    
    # Sort and format
    top_df = df.sort_values(by="S(α=0.5)", ascending=False).head(10)
    top_df_display = top_df[['Model', 'Sampler', 'Method', 'AUC', 'I', 'S(α=0.5)']].copy()
    
    # Highlight the absolute best row
    st.dataframe(
        top_df_display.style.highlight_max(subset=['S(α=0.5)'], color='lightgreen', axis=0)
              .format({'AUC': "{:.4f}", 'I': "{:.4f}", 'S(α=0.5)': "{:.4f}"}),
        use_container_width=True
    )
    
    # Quick metrics for best config
    best_config = top_df.iloc[0]
    st.success(f"🥇 **Best Overall Configuration:** **{best_config['Method']}** using **{best_config['Model']}** with **{best_config['Sampler']}**.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Top AUC", f"{best_config['AUC']:.4f}")
    col2.metric("Top Interpretability (I)", f"{best_config['I']:.4f}")
    col3.metric("Top S(α=0.5)", f"{best_config['S(α=0.5)']:.4f}")

# ------------------------------------------
# TAB 2: METHOD COMPARISON (Bar Charts)
# ------------------------------------------
with tab2:
    st.subheader("Average Metrics by Explanation Method")
    
    # Group by method
    method_means = df.groupby('Method')[['AUC', 'I', 'S(α=0.5)']].mean().reset_index()
    
    # Sort to ensure consistent colors
    method_means['Method'] = pd.Categorical(method_means['Method'], categories=list(METHOD_COLORS.keys()), ordered=True)
    method_means = method_means.sort_values('Method')

    col1, col2 = st.columns(2)
    
    with col1:
        # S(alpha) Bar Chart
        fig_S = px.bar(
            method_means, x='Method', y='S(α=0.5)', 
            color='Method', color_discrete_map=METHOD_COLORS,
            title="Mean Performance-Interpretability Score S(α=0.5)",
            text_auto='.3f'
        )
        fig_S.update_layout(showlegend=False, xaxis_title="Explanation Method", yaxis_title="Mean S(α=0.5)")
        st.plotly_chart(fig_S, use_container_width=True)

    with col2:
        # I-Score Bar Chart
        fig_I = px.bar(
            method_means, x='Method', y='I', 
            color='Method', color_discrete_map=METHOD_COLORS,
            title="Mean Overall Interpretability (I)",
            text_auto='.3f'
        )
        fig_I.update_layout(showlegend=False, xaxis_title="Explanation Method", yaxis_title="Mean I-Score")
        st.plotly_chart(fig_I, use_container_width=True)

# ------------------------------------------
# TAB 3: PARETO TRADE-OFF
# ------------------------------------------
with tab3:
    st.subheader("Accuracy vs. Interpretability Trade-off")
    st.markdown("Methods positioned in the **top-right corner** represent the optimal Pareto front (high accuracy, high interpretability).")
    
    fig_pareto = px.scatter(
        df, x='AUC', y='I', 
        color='Method', color_discrete_map=METHOD_COLORS,
        symbol='Model',
        hover_data=['Sampler', 'S(α=0.5)'],
        size_max=12,
        title="Pareto Front: AUC vs I-Score"
    )
    fig_pareto.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig_pareto.update_layout(xaxis_title="AUC-ROC (Accuracy)", yaxis_title="I-Score (Interpretability)", height=600)
    st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown("---")
    st.subheader("Group Quality (Q) vs. Interpretability (I) - Owen Variants Only")
    
    owen_df = df[df['Method'].isin(['Owen-Domain', 'Owen-Data', 'Owen-Model'])].copy()
    if not owen_df.empty and 'Q' in owen_df.columns:
        fig_q = px.scatter(
            owen_df, x='Q', y='I', 
            color='Method', color_discrete_map=METHOD_COLORS,
            hover_data=['Model', 'Sampler'],
            title="Q-Score vs I-Score (Evaluating Contextual Grouping)"
        )
        fig_q.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig_q, use_container_width=True)
    else:
        st.info("No valid Q scores found for Owen variants in this dataset.")

# ------------------------------------------
# TAB 4: HEATMAPS
# ------------------------------------------
with tab4:
    st.subheader("Aggregated Performance Heatmaps")
    st.markdown("Analyze how different Models and Sampling techniques interact.")
    
    metric_choice = st.selectbox("Select Metric to Visualize:", ["AUC", "I", "S(α=0.5)", "Stability"])
    
    # Pivot data
    pivot_df = df.pivot_table(values=metric_choice, index='Model', columns='Sampler', aggfunc='mean')
    
    # Dynamic colorscale based on metric
    cmap = "Blues" if metric_choice == "AUC" else "Reds" if metric_choice == "I" else "Purples"
    
    fig_heat = px.imshow(
        pivot_df, 
        text_auto=".3f", 
        aspect="auto",
        color_continuous_scale=cmap,
        title=f"Mean {metric_choice} by Model × Sampler"
    )
    fig_heat.update_layout(xaxis_title="Resampling Strategy", yaxis_title="Classifier Model")
    st.plotly_chart(fig_heat, use_container_width=True)

# ------------------------------------------
# TAB 5: RAW DATA
# ------------------------------------------
with tab5:
    st.subheader("Raw Dataset")
    st.markdown("Explore, filter, or download the raw metrics for your thesis reporting.")
    st.dataframe(df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f"exported_{uploaded_file.name}",
        mime="text/csv",
    )
