import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Credit Risk XAI Explorer",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional UI/UX Custom CSS
st.markdown("""
    <style>
    .main { font-family: 'Inter', -apple-system, sans-serif; background-color: #fcfcfc;}
    h1, h2, h3 { color: #0f172a; font-weight: 700; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .leaderboard-card { background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: transform 0.2s;}
    .leaderboard-card:hover { transform: translateY(-5px); }
    .rank-icon { font-size: 2.5rem; margin-bottom: 10px; }
    .insight-box { background-color: #f0f9ff; border-left: 4px solid #0ea5e9; padding: 15px; border-radius: 4px; margin: 15px 0; color: #0f172a;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
METHOD_COLORS = {
    'SHAP': '#3b82f6',         # Blue
    'Banzhaf': '#f59e0b',      # Amber
    'Myerson': '#22c55e',      # Green
    'Owen-Domain': '#ef4444',  # Red
    'Owen-Data': '#8b5cf6',    # Purple
    'Owen-Model': '#64748b',   # Slate
    'R-Myerson': '#06b6d4'     # Cyan (Highlighted)
}

DATASET_REGISTRY = {
    "German Credit": {
        "main": "german_results_7methods.csv",
        "wilcoxon": "german_wilcoxon_cliffs_results.csv",
        "nemenyi": "german_nemenyi_results.csv",
        "corr": "german_auc_I_correlation.csv",
        "label": "High Default Rate (30%)",
        "imbalance_rate": 30.0
    },
    "Taiwan Credit": {
        "main": "taiwan_results_7methods_S.csv",
        "wilcoxon": "taiwan_wilcoxon_cliffs_results.csv",
        "nemenyi": "taiwan_nemenyi_results.csv",
        "corr": "taiwan_auc_I_correlation.csv",
        "label": "Moderate Default Rate (22.12%)",
        "imbalance_rate": 22.12
    },
    "Lending Club 10%": {
        "main": "LC10pcdefaultresults.csv",
        "wilcoxon": "lc10_wilcoxon_cliffs_results.csv",
        "nemenyi": "lc10_nemenyi_results.csv",
        "corr": "lc10_auc_I_correlation.csv",
        "label": "Industry Standard (10%)",
        "imbalance_rate": 10.0
    },
    "Lending Club LC66": {
        "main": "LC66_results_7methods_noleak.csv",
        "wilcoxon": "Lc66_wilcoxon_cliffs_results .csv", # matching your uploaded name with space
        "nemenyi": "Lc66_nemenyi_results.csv",
        "corr": "Lc66_correlation.csv",
        "label": "Severe Imbalance (4.01%)",
        "imbalance_rate": 4.01
    },
    "Coursera Loans": {
        "main": "coursera_loans_results_7methods.csv",
        "wilcoxon": "wilcoxon_cliffs_results_coursera.csv",
        "nemenyi": "nemenyi_results_coursera.csv",
        "corr": "auc_I_correlation_coursera.csv",
        "label": "Extreme Imbalance (1%)",
        "imbalance_rate": 1.0
    }
}

# ==========================================
# UTILITIES
# ==========================================
@st.cache_data
def load_data(path, is_index=False):
    # Robust file loading handling common naming inconsistencies
    variations = [path, path.replace(' .csv', '.csv'), path.replace('.csv', ' (1).csv')]
    for v in variations:
        if os.path.exists(v):
            try:
                df = pd.read_csv(v, index_col=0 if is_index else None)
                # Clean up Sampler column ('nan' -> 'None')
                if 'Sampler' in df.columns:
                    df['Sampler'] = df['Sampler'].fillna('None').replace('nan', 'None')
                return df
            except: pass
    return None

def color_consensus(val):
    if val == 'Yes (True Diff)':
        return 'background-color: #dcfce7; color: #166534; font-weight: bold;'
    return 'color: #94a3b8;'

def color_effect(val):
    val_str = str(val).lower()
    if val_str == 'large': return 'color: #059669; font-weight: bold;'
    if val_str == 'medium': return 'color: #d97706; font-weight: bold;'
    return 'color: #64748b;'

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("### 🧭 Navigation")
views = ["🌐 Global Synthesis"] + list(DATASET_REGISTRY.keys())
selection = st.sidebar.radio("Select View:", views)

st.sidebar.markdown("---")
st.sidebar.markdown("<small><b>Methodology Pipeline</b><br>Evaluation of Game-Theoretic explainers across varying levels of class imbalance.</small>", unsafe_allow_html=True)

# ==========================================
# VIEW: GLOBAL SYNTHESIS
# ==========================================
if selection == "🌐 Global Synthesis":
    st.title("Ensemble Learning and Coalition-Aware Explainability for Imbalanced Credit Default")
    
    st.markdown("""
    <div class='insight-box'>
    <b>Executive Summary:</b> This dashboard compares seven feature attribution methods across five datasets to demystify the Accuracy-Interpretability trade-off. We introduce <b>R-Myerson</b> as a robust solution that redistributes graph-constrained attributions to maintain stability even in extreme (1%) imbalance scenarios.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Cross-Dataset Performance Trajectory")
    
    # Compile global dataset
    global_results = []
    for name, cfg in DATASET_REGISTRY.items():
        df = load_data(cfg['main'])
        if df is not None:
            summary = df.groupby('Method')['S(α=0.5)'].mean().reset_index()
            summary['Imbalance'] = cfg['imbalance_rate']
            summary['Dataset'] = name
            global_results.append(summary)
            
    if global_results:
        combined = pd.concat(global_results)
        combined = combined.sort_values('Imbalance', ascending=False) # 30% down to 1%
        
        # Interactive Line/Scatter Area Plot
        fig = px.area(combined, x='Imbalance', y='S(α=0.5)', color='Method', 
                      color_discrete_map=METHOD_COLORS, line_group='Method',
                      title="Evolution of Explainer Stability (S-Score) as Default Rate Decreases",
                      markers=True)
        
        fig.update_layout(
            xaxis_title="Default Rate / Imbalance (%) → (Decreasing)", 
            yaxis_title="Overall S(α=0.5) Score",
            xaxis=dict(autorange="reversed"), # 30% on left, 1% on right
            template="plotly_white",
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("### Overall Dominance")
    st.markdown("Across all datasets tested, the combination of **SMOTETomek** resampling with the **R-Myerson** explainer consistently achieves the optimal Pareto frontier, preventing the typical degradation seen in SHAP when minority classes fall below 5%.")

# ==========================================
# VIEW: SPECIFIC DATASET
# ==========================================
else:
    cfg = DATASET_REGISTRY[selection]
    st.title(f"{selection}")
    st.markdown(f"**Dataset Profile:** {cfg['label']}")
    
    # Load Data
    main_df = load_data(cfg['main'])
    wil_df = load_data(cfg['wilcoxon'])
    nem_df = load_data(cfg['nemenyi'], is_index=True)
    corr_df = load_data(cfg['corr'])
    
    if main_df is None:
        st.error(f"Data missing for {selection}. Please ensure `{cfg['main']}` is uploaded.")
        st.stop()

    # --- LEADERBOARD (Top 3 Podium) ---
    st.markdown("### 🏆 Top 3 Configurations")
    top3 = main_df.sort_values('S(α=0.5)', ascending=False).head(3).reset_index(drop=True)
    
    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i in range(len(top3)):
        with cols[i]:
            st.markdown(f"""
            <div class='leaderboard-card'>
                <div class='rank-icon'>{medals[i]}</div>
                <h4 style='margin-bottom:0px; color:#0f172a;'>{top3.loc[i, 'Method']}</h4>
                <p style='color:#64748b; font-size:0.9rem; margin-top:0px;'>{top3.loc[i, 'Model']} + {top3.loc[i, 'Sampler']}</p>
                <h2 style='color:#0ea5e9; margin:10px 0;'>{top3.loc[i, 'S(α=0.5)']:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- TABS ---
    tabs = st.tabs(["🎯 Accuracy vs Interpretability", "🧩 Q vs I (Group Quality)", "🔬 Statistical Significance"])
    
    # TAB 1: ACCURACY VS INTERPRETABILITY
    with tabs[0]:
        col_scatter, col_bar = st.columns([1.5, 1])
        
        with col_scatter:
            fig_p = px.scatter(main_df, x='AUC', y='I', color='Method', symbol='Model',
                             hover_data=['Sampler'], color_discrete_map=METHOD_COLORS,
                             title="Pareto Front (AUC vs. I-Score)",
                             animation_frame="Sampler" if len(main_df['Sampler'].unique()) > 1 else None) # Adds animation to see effect of samplers
            
            fig_p.update_traces(marker=dict(size=14, line=dict(width=1, color='white')), opacity=0.85)
            fig_p.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig_p, use_container_width=True)

        with col_bar:
            m_means = main_df.groupby('Method')['S(α=0.5)'].mean().sort_values(ascending=True).reset_index()
            fig_b = px.bar(m_means, x='S(α=0.5)', y='Method', orientation='h', 
                           color='Method', color_discrete_map=METHOD_COLORS, text_auto='.3f',
                           title="Average S-Score by Method")
            fig_b.update_layout(showlegend=False, template="plotly_white", height=450, yaxis_title="")
            st.plotly_chart(fig_b, use_container_width=True)
            
        # Analysis Text for AUC vs I
        if corr_df is not None:
            rho = corr_df['Spearman_rho'].iloc[0]
            p = corr_df['Spearman_p'].iloc[0]
            st.markdown(f"""
            <div class='insight-box'>
            <b>Relation Analysis:</b> The Spearman correlation between Accuracy (AUC) and Interpretability (I) is <b>ρ = {rho:.3f}</b> (p-value: {p:.3f}). 
            {'This indicates a significant trade-off.' if p < 0.05 else 'The high p-value indicates that accuracy and interpretability behave independently here; we can improve explainability without sacrificing predictive power!'}
            </div>
            """, unsafe_allow_html=True)

    # TAB 2: Q vs I (Owen Variants)
    with tabs[1]:
        st.markdown("### Evaluating Coalition Groupings")
        st.markdown("For Owen value variants (Domain, Data, Model), the Group Quality (Q) measures the structural integrity of the feature coalitions. A positive relationship here validates our feature clustering methodology.")
        
        owen_df = main_df[main_df['Method'].isin(['Owen-Domain', 'Owen-Data', 'Owen-Model'])].copy()
        
        if not owen_df.empty and 'Q' in owen_df.columns and not owen_df['Q'].isna().all():
            fig_q = px.scatter(owen_df, x='Q', y='I', color='Method', hover_data=['Model', 'Sampler'],
                               color_discrete_map=METHOD_COLORS, trendline="ols",
                               title="Group Quality (Q) vs Interpretability (I)")
            fig_q.update_traces(marker=dict(size=14, line=dict(width=1, color='white')))
            fig_q.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.info("Q-Score data is not available or contains NaNs for the Owen variants in this dataset.")

    # TAB 3: STATISTICAL SIGNIFICANCE (Combined Wilcoxon & Nemenyi)
    with tabs[2]:
        st.markdown("### Rigorous Pairwise Comparison")
        st.markdown("To accurately determine which methods are statistically different, we combine the **Wilcoxon Signed-Rank Test** (pairwise) and the **Nemenyi Post-hoc Test** (multi-group). We consider two methods to have a **True Difference** only if *both* tests yield p < 0.05.")
        
        if wil_df is not None and nem_df is not None:
            # Create a Combined DataFrame
            combined_stats = []
            
            for _, row in wil_df.iterrows():
                m1, m2 = row['Method1'], row['Method2']
                w_p = row['p_value']
                eff = row['Effect_size']
                
                # Fetch Nemenyi p-value safely
                n_p = np.nan
                try:
                    n_p = nem_df.loc[m1, m2]
                except:
                    try:
                        n_p = nem_df.loc[m2, m1]
                    except: pass
                
                # Determine Consensus
                consensus = "Yes (True Diff)" if (w_p < 0.05 and n_p < 0.05) else "No"
                
                combined_stats.append({
                    "Method 1": m1, "Method 2": m2,
                    "Effect Size": eff.title(),
                    "Wilcoxon (p)": w_p,
                    "Nemenyi (p)": n_p,
                    "Consensus": consensus
                })
            
            stat_df = pd.DataFrame(combined_stats)
            
            # Format and Display
            st.dataframe(
                stat_df.style
                .format({'Wilcoxon (p)': "{:.4f}", 'Nemenyi (p)': "{:.4f}"})
                .map(color_effect, subset=['Effect Size'])
                .map(color_consensus, subset=['Consensus']),
                use_container_width=True,
                height=450
            )
            
            st.caption("🔍 **How to read:** Look for 'Yes' in the Consensus column and a 'Large' Effect Size. This proves that Method 1 practically and statistically outperforms Method 2.")
        else:
            st.warning("Both Wilcoxon and Nemenyi CSV files are required to display the combined significance table.")
