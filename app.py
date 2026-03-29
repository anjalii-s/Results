import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Credit XAI Explorer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Font and CSS Styling
st.markdown("""
    <style>
    .main { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    .stMetric { background-color: #f8f9fb; padding: 15px; border-radius: 10px; border: 1px solid #eef2f6; }
    .stPlotlyChart { border-radius: 15px; }
    h1, h2, h3 { color: #1e293b; }
    .report-text { font-size: 1.1rem; color: #475569; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# CONSTANTS & CONFIG
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

DATASET_REGISTRY = {
    "German Credit (30%)": {
        "main": "german_results_7methods.csv",
        "wilcoxon": "german_wilcoxon_cliffs_results.csv",
        "nemenyi": "german_nemenyi_results.csv",
        "corr": "german_auc_I_correlation.csv",
        "label": "High Default Rate (30%)"
    },
    "Taiwan Credit (22.12%)": {
        "main": "taiwan_results_7methods_S.csv",
        "wilcoxon": "taiwan_wilcoxon_cliffs_results.csv",
        "nemenyi": "taiwan_nemenyi_results.csv",
        "corr": "taiwan_auc_I_correlation.csv",
        "label": "Moderate Default Rate (22%)"
    },
    "Lending Club (10%)": {
        "main": "LC10pcdefaultresults.csv",
        "wilcoxon": "lc10_wilcoxon_cliffs_results.csv",
        "nemenyi": "lc10_nemenyi_results.csv",
        "corr": "lc10_auc_I_correlation.csv",
        "label": "Industry Standard (10%)"
    },
    "Lending Club LC66 (4.01%)": {
        "main": "LC66_results_7methods_noleak.csv",
        "wilcoxon": "Lc66_wilcoxon_cliffs_results.csv",
        "nemenyi": "Lc66_nemenyi_results.csv",
        "corr": "Lc66_correlation.csv",
        "label": "Severe Imbalance (4%)"
    },
    "Coursera Loans (1%)": {
        "main": "coursera_loans_results_7methods.csv",
        "wilcoxon": "wilcoxon_cliffs_results_coursera.csv",
        "nemenyi": "nemenyi_results_coursera.csv",
        "corr": "auc_I_correlation_coursera.csv",
        "label": "Extreme Imbalance (1%)"
    }
}

# ==========================================
# UTILITIES
# ==========================================
@st.cache_data
def load_data(path, is_index=False):
    # Handle filename variations
    variations = [path, path.replace('.csv', ' (1).csv'), path.replace('.csv', ' .csv')]
    for v in variations:
        if os.path.exists(v):
            try:
                df = pd.read_csv(v, index_col=0 if is_index else None)
                if 'Sampler' in df.columns:
                    df['Sampler'] = df['Sampler'].fillna('None')
                return df
            except: pass
    return None

def analyze_auc_i_relation(corr_df):
    if corr_df is None: return "Correlation data unavailable."
    rho = corr_df['Spearman_rho'].iloc[0]
    p = corr_df['Spearman_p'].iloc[0]
    
    if p > 0.05:
        return "**Analysis:** The relation between Accuracy (AUC) and Interpretability (I) is not statistically significant. This suggests that achieving higher interpretability does not necessarily come at the cost of model performance in this specific dataset."
    else:
        direction = "negative" if rho < 0 else "positive"
        return f"**Analysis:** There is a significant {direction} correlation (ρ={rho:.2f}) between Accuracy and Interpretability, suggesting a traditional trade-off exists within this dataset's configuration."

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("Navigation")
selection = st.sidebar.selectbox("Choose View:", ["📊 Global Synthesis"] + list(DATASET_REGISTRY.keys()))

# ==========================================
# HOME PAGE / GLOBAL SYNTHESIS
# ==========================================
if selection == "📊 Global Synthesis":
    st.title("Ensemble Learning and Coalition-aware Explainability for Imbalanced Credit Default")
    st.markdown("""
    <div class='report-text'>
    This dashboard visualizes research on Game-Theoretic attribution methods. 
    The goal is to determine which methods remain <b>stable</b> and <b>reliable</b> when 
    predicting rare credit default events across varying levels of class imbalance.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("Stability Profile across Imbalance Scenarios")
    global_results = []
    for name, cfg in DATASET_REGISTRY.items():
        df = load_data(cfg['main'])
        if df is not None:
            summary = df.groupby('Method')['S(α=0.5)'].mean().reset_index()
            summary['Imbalance'] = name
            global_results.append(summary)
            
    if global_results:
        combined = pd.concat(global_results)
        fig = px.line(combined, x='Imbalance', y='S(α=0.5)', color='Method', markers=True,
                     color_discrete_map=METHOD_COLORS, title="S-Score Trend: From Balanced (30%) to Imbalanced (1%)")
        fig.update_layout(xaxis_title="Dataset (Sorted by Imbalance)", yaxis_title="Overall Score S(α=0.5)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("The proposed **R-Myerson** method (Cyan) demonstrates superior stability particularly as the default rate decreases.")

# ==========================================
# DATASET PAGE
# ==========================================
else:
    cfg = DATASET_REGISTRY[selection]
    st.header(f"{selection} — {cfg['label']}")
    
    main_df = load_data(cfg['main'])
    wil_df = load_data(cfg['wilcoxon'])
    nem_df = load_data(cfg['nemenyi'], is_index=True)
    corr_df = load_data(cfg['corr'])
    
    if main_df is None:
        st.error(f"Missing data file: `{cfg['main']}`. Please ensure it is in the repository.")
    else:
        tabs = st.tabs(["🎯 Trade-off & Performance", "🔬 Statistical Significance", "🗄️ Raw Data"])
        
        with tabs[0]:
            col1, col2 = st.columns([1.5, 1])
            with col1:
                st.subheader("Accuracy vs. Interpretability (Pareto Front)")
                fig_p = px.scatter(main_df, x='AUC', y='I', color='Method', symbol='Model',
                                 hover_data=['Sampler', 'S(α=0.5)'], color_discrete_map=METHOD_COLORS)
                fig_p.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
                fig_p.update_layout(height=500)
                st.plotly_chart(fig_p, use_container_width=True)
                st.markdown(analyze_auc_i_relation(corr_df))

            with col2:
                st.subheader("Leaderboard (Top 5)")
                top5 = main_df.sort_values('S(α=0.5)', ascending=False).head(5)
                st.table(top5[['Model', 'Sampler', 'Method', 'S(α=0.5)']].style.format({'S(α=0.5)': '{:.4f}'}))
                
                # Distribution of scores
                fig_bar = px.box(main_df, x='Method', y='S(α=0.5)', color='Method', 
                                color_discrete_map=METHOD_COLORS, title="Score Distribution")
                fig_bar.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_bar, use_container_width=True)

        with tabs[1]:
            st.subheader("Hypothesis Testing")
            s_col1, s_col2 = st.columns([1, 1.2])
            
            with s_col1:
                st.markdown("**Wilcoxon Pairwise Comparison**")
                if wil_df is not None:
                    # Specific column reduction as requested
                    display_wil = wil_df[['Method1', 'Method2', 'Significant', 'Effect_size']].copy()
                    def color_effect(val):
                        if str(val).lower() == 'large': return 'background-color: #d1fae5; color: #065f46;'
                        if str(val).lower() == 'medium': return 'background-color: #fef3c7; color: #92400e;'
                        return ''
                    st.dataframe(display_wil.style.applymap(color_effect, subset=['Effect_size']), height=400, use_container_width=True)
                else: st.warning("Wilcoxon results not found.")

            with s_col2:
                st.markdown("**Nemenyi Post-hoc Significance (p-values)**")
                if nem_df is not None:
                    fig_nem = px.imshow(nem_df, text_auto=".3f", color_continuous_scale='RdYlGn_r', zmin=0, zmax=0.1)
                    fig_nem.update_layout(height=450, margin=dict(t=0))
                    st.plotly_chart(fig_nem, use_container_width=True)
                    st.caption("**Interpretation:** Cells with values < 0.05 (Red/Orange) indicate that the two methods are statistically different from each other.")
                else: st.warning("Nemenyi heatmap data not found.")

        with tabs[2]:
            st.dataframe(main_df, use_container_width=True)
