import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="XAI Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# THEME & COLORS
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
# DATASET REGISTRY
# ==========================================
DATASET_CONFIG = {
    "🇩🇪 German Credit (30%)": {
        "main": "german_results_7methods.csv",
        "wilcoxon": "german_wilcoxon_cliffs_results.csv",
        "nemenyi": "german_nemenyi_results.csv",
        "corr": "german_auc_I_correlation.csv",
        "desc": "High default rate (30%). Used to test XAI stability in balanced scenarios."
    },
    "🇹🇼 Taiwan Credit (22.12%)": {
        "main": "taiwan_results_7methods_S.csv",
        "wilcoxon": "taiwan_wilcoxon_cliffs_results.csv",
        "nemenyi": "taiwan_nemenyi_results.csv",
        "corr": "taiwan_auc_I_correlation.csv",
        "desc": "Moderate imbalance. Focus on contextual feature dependencies."
    },
    "🏦 Lending Club (10%)": {
        "main": "LC10pcdefaultresults.csv",
        "wilcoxon": "lc10_wilcoxon_cliffs_results.csv",
        "nemenyi": "lc10_nemenyi_results.csv",
        "corr": "lc10_auc_I_correlation.csv",
        "desc": "Standard industry default rate. Benchmarking ensemble performance."
    },
    "🏦 Lending Club LC66 (4.01%)": {
        "main": "LC66_results_7methods_noleak.csv",
        "wilcoxon": "Lc66_wilcoxon_cliffs_results.csv",
        "nemenyi": "Lc66_nemenyi_results.csv",
        "corr": "Lc66_correlation.csv",
        "desc": "Severe imbalance. Testing robustness to rare event prediction."
    },
    "🎓 Coursera Loans (1%)": {
        "main": "coursera_loans_results_7methods.csv",
        "wilcoxon": "coursera_wilcoxon_cliffs_results.csv",
        "nemenyi": "coursera_nemenyi_results.csv",
        "corr": "coursera_auc_I_correlation.csv",
        "desc": "Extreme imbalance (1%). Evaluating XAI reliability at the edge."
    }
}

# ==========================================
# UTILITIES
# ==========================================
@st.cache_data
def load_csv(path, index_col=None):
    # Try multiple variations of the filename to prevent errors
    variations = [path, path.replace('.csv', ' (1).csv'), path.replace('.csv', ' .csv')]
    for v in variations:
        if os.path.exists(v):
            try:
                return pd.read_csv(v, index_col=index_col)
            except: pass
    return None

def style_df(df):
    if df is None: return None
    def color_size(val):
        if str(val).lower() == 'large': return 'background-color: rgba(44, 160, 44, 0.2); font-weight: bold;'
        if str(val).lower() == 'medium': return 'background-color: rgba(255, 127, 14, 0.2);'
        return ''
    return df.style.applymap(color_size, subset=['Effect_size']) if 'Effect_size' in df.columns else df

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://img.icons8.com/fluency/96/analytics.png", width=80)
st.sidebar.title("Thesis Explorer")
selection = st.sidebar.radio("Go to:", ["🏠 Home & Global Synthesis"] + list(DATASET_CONFIG.keys()))

# ==========================================
# TAB 1: HOME & GLOBAL SYNTHESIS
# ==========================================
if selection == "🏠 Home & Global Synthesis":
    st.title("🏦 Ensemble Learning & XAI in Credit Risk")
    st.markdown("""
    ### Thesis Overview
    **Research Title:** *Analysis of the Accuracy and Interpretability Trade-Off in Imbalanced Loan Default Prediction.*
    
    This dashboard provides an interactive look at how **Cooperative Game Theory** (Shapley, Banzhaf, Myerson, and Owen) 
    can provide more stable and regulator-aligned explanations for complex credit models (RF, XGBoost, LightGBM).
    """)
    
    st.divider()
    st.subheader("🌐 Global Performance Summary: R-Myerson vs. Others")
    
    # Aggregate S(alpha) across all datasets for a global comparison
    global_data = []
    for name, cfg in DATASET_CONFIG.items():
        df = load_csv(cfg['main'])
        if df is not None:
            means = df.groupby('Method')['S(α=0.5)'].mean().reset_index()
            means['Dataset'] = name.split(' ')[0] # Just the flag/name
            global_data.append(means)
            
    if global_data:
        full_global = pd.concat(global_data)
        fig_global = px.line(full_global, x='Dataset', y='S(α=0.5)', color='Method',
                            markers=True, color_discrete_map=METHOD_COLORS,
                            title="Global Stability Profile (S-Score) across Imbalance Levels")
        fig_global.update_layout(yaxis_title="Mean Performance-Interpretability Score")
        st.plotly_chart(fig_global, use_container_width=True)
        st.info("💡 **Observation:** Notice how R-Myerson (Cyan) maintains high stability even as the default rate drops to 1%.")

# ==========================================
# DATASET SPECIFIC PAGES
# ==========================================
else:
    cfg = DATASET_CONFIG[selection]
    st.title(f"{selection}")
    st.caption(cfg['desc'])
    
    # Load Data
    main_df = load_csv(cfg['main'])
    wilcoxon_df = load_csv(cfg['wilcoxon'])
    nemenyi_df = load_csv(cfg['nemenyi'], index_col=0)
    corr_df = load_csv(cfg['corr'])
    
    if main_df is None:
        st.error(f"Missing primary results file: `{cfg['main']}`")
    else:
        # Layout
        tab_perf, tab_stat, tab_raw = st.tabs(["🎯 Trade-off Analysis", "🔬 Statistical Significance", "🗄️ Raw Data"])
        
        with tab_perf:
            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.subheader("Accuracy-Interpretability Pareto Front")
                fig_p = px.scatter(main_df, x='AUC', y='I', color='Method', symbol='Model',
                                 hover_data=['Sampler', 'S(α=0.5)'], color_discrete_map=METHOD_COLORS)
                fig_p.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
                st.plotly_chart(fig_p, use_container_width=True)
            
            with col2:
                st.subheader("Leaderboard: Top 5 Configurations")
                top5 = main_df.sort_values('S(α=0.5)', ascending=False).head(5)
                st.table(top5[['Model', 'Sampler', 'Method', 'S(α=0.5)']].style.format({'S(α=0.5)': '{:.4f}'}))
                
                # Metric display
                best = top5.iloc[0]
                st.metric("Highest Achieved S(α=0.5)", f"{best['S(α=0.5)']:.4f}", f"via {best['Method']}")

            st.divider()
            st.subheader("Mean Score (S) by Explainer")
            m_means = main_df.groupby('Method')['S(α=0.5)'].mean().sort_values(ascending=False).reset_index()
            fig_b = px.bar(m_means, x='Method', y='S(α=0.5)', color='Method', color_discrete_map=METHOD_COLORS, text_auto='.3f')
            st.plotly_chart(fig_b, use_container_width=True)

        with tab_stat:
            s_col1, s_col2 = st.columns([1, 1])
            
            with s_col1:
                st.markdown("#### Wilcoxon & Cliff's Delta Effect Size")
                if wilcoxon_df is not None:
                    st.dataframe(style_df(wilcoxon_df), use_container_width=True, height=400)
                    st.caption("Green highlight indicates a 'Large' mitigation effect size.")
                else: st.warning("Wilcoxon results not found.")
                
                st.markdown("#### Correlation: AUC vs Interpretability")
                if corr_df is not None:
                    c1, c2 = st.columns(2)
                    c1.metric("Spearman ρ", f"{corr_df['Spearman_rho'].iloc[0]:.3f}", help="Close to 0 means Accuracy and Interpretability are independent.")
                    c2.metric("Kendall τ", f"{corr_df['Kendall_tau'].iloc[0]:.3f}")
                else: st.warning("Correlation results not found.")

            with s_col2:
                st.markdown("#### Nemenyi Post-hoc Significance (p-values)")
                if nemenyi_df is not None:
                    fig_n = px.imshow(nemenyi_df, text_auto=".3f", color_continuous_scale='RdYlGn_r', zmin=0, zmax=0.1)
                    fig_n.update_layout(height=450, margin=dict(t=20))
                    st.plotly_chart(fig_n, use_container_width=True)
                    st.markdown("<small>🟥 Cells with <b>p < 0.05</b> indicate statistically significant differences.</small>", unsafe_allow_html=True)
                else: st.warning("Nemenyi results not found.")

        with tab_raw:
            st.dataframe(main_df, use_container_width=True)
            st.download_button("Download CSV", main_df.to_csv(index=False), f"{selection}_raw.csv", "text/csv")
