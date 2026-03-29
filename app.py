import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="Credit Risk XAI Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    h1, h2, h3, h4 { color: #1e293b; font-weight: 600; }
    .leaderboard-card { background: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: transform 0.2s;}
    .leaderboard-card:hover { transform: translateY(-5px); border-color: #cbd5e1; }
    .rank-icon { font-size: 2.5rem; margin-bottom: 5px; }
    .insight-box { background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 15px; border-radius: 4px; margin: 15px 0; color: #334155; font-size: 1.05rem;}
    .stat-sig { color: #ef4444; font-weight: bold; }
    .stat-insig { color: #3b82f6; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
METHOD_COLORS = {
    'SHAP': '#94a3b8',         # Slate (Baseline)
    'Banzhaf': '#f59e0b',      # Amber
    'Myerson': '#22c55e',      # Green
    'Owen-Domain': '#ef4444',  # Red
    'Owen-Data': '#8b5cf6',    # Purple
    'Owen-Model': '#d946ef',   # Fuchsia
    'R-Myerson': '#0ea5e9'     # Cyan (Proposed)
}

# Clean registry without redundant percentages in titles
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
        "wilcoxon": "Lc66_wilcoxon_cliffs_results .csv", 
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
    variations = [path, path.replace(' .csv', '.csv'), path.replace('.csv', ' (1).csv')]
    for v in variations:
        if os.path.exists(v):
            try:
                df = pd.read_csv(v, index_col=0 if is_index else None)
                if 'Sampler' in df.columns:
                    df['Sampler'] = df['Sampler'].fillna('None').replace('nan', 'None')
                return df
            except: pass
    return None

def color_effect(val):
    v = str(val).lower()
    if v == 'large': return 'color: #059669; font-weight: bold;' # Green
    if v == 'medium': return 'color: #d97706; font-weight: bold;' # Orange
    return 'color: #94a3b8;' # Gray

def get_consensus(pval_w, pval_n):
    if pd.isna(pval_n): return "Pending"
    if pval_w < 0.05 and pval_n < 0.05: return "✓ Yes"
    return "✗ No"

def color_consensus(val):
    if '✓' in str(val): return 'background-color: #dcfce7; color: #166534; font-weight: bold;'
    return 'color: #94a3b8;'

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("### 🧭 Pipeline Navigation")
views = ["📊 Cross-Dataset Synthesis"] + list(DATASET_REGISTRY.keys())
selection = st.sidebar.radio("Select View:", views)

st.sidebar.markdown("---")
st.sidebar.markdown("<small><b>Research Focus:</b> Maintaining XAI stability in severely imbalanced ensembles.</small>", unsafe_allow_html=True)

# ==========================================
# VIEW 1: CROSS-DATASET SYNTHESIS
# ==========================================
if selection == "📊 Cross-Dataset Synthesis":
    st.title("Ensemble Learning and Coalition-aware Explainability for Imbalanced Credit Default")
    
    st.markdown("""
    <div class='insight-box'>
    <b>Research Synthesis:</b> This module compares 7 attribution methods across 5 datasets. 
    As the default rate drops from 30% to 1%, standard methods like SHAP experience severe instability. 
    The proposed <b>R-Myerson</b> maintains structural integrity by equitably redistributing graph-constrained contributions.
    </div>
    """, unsafe_allow_html=True)
    
    # Load global data
    global_results = []
    for name, cfg in DATASET_REGISTRY.items():
        df = load_data(cfg['main'])
        if df is not None:
            summary = df.groupby('Method')['S(α=0.5)'].mean().reset_index()
            summary['Imbalance'] = cfg['imbalance_rate']
            summary['Dataset'] = f"{name} ({cfg['imbalance_rate']}%)"
            global_results.append(summary)
            
    if global_results:
        combined = pd.concat(global_results).sort_values('Imbalance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.line(combined, x='Dataset', y='S(α=0.5)', color='Method', 
                          color_discrete_map=METHOD_COLORS, markers=True,
                          title="Explainer Stability Degradation across Imbalance Levels")
            fig.update_layout(xaxis_title="Datasets (Decreasing Minority Class →)", yaxis_title="Mean Performance-Interpretability (S-Score)", template="plotly_white", hovermode="x unified", height=500)
            # Emphasize R-Myerson line
            fig.update_traces(line=dict(width=4) if 'R-Myerson' in fig.data else dict(width=2))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### Overall Dominance")
            overall_mean = combined.groupby('Method')['S(α=0.5)'].mean().sort_values(ascending=True).reset_index()
            fig_bar = px.bar(overall_mean, x='S(α=0.5)', y='Method', orientation='h', color='Method', color_discrete_map=METHOD_COLORS, text_auto='.3f')
            fig_bar.update_layout(showlegend=False, template="plotly_white", height=450, xaxis_title="Global Mean S-Score")
            st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# VIEW 2: DATASET SPECIFIC
# ==========================================
else:
    cfg = DATASET_REGISTRY[selection]
    st.header(f"{selection}")
    st.caption(f"Dataset Profile: {cfg['label']}")
    
    # Load Data
    main_df = load_data(cfg['main'])
    wil_df = load_data(cfg['wilcoxon'])
    nem_df = load_data(cfg['nemenyi'], is_index=True)
    corr_df = load_data(cfg['corr'])
    
    if main_df is None:
        st.error(f"⚠️ Results file `{cfg['main']}` not found.")
        st.stop()

    # --- TOP 3 PODIUM ---
    top3 = main_df.sort_values('S(α=0.5)', ascending=False).head(3).reset_index(drop=True)
    cols = st.columns(3)
    medals = ["🥇 1st Place", "🥈 2nd Place", "🥉 3rd Place"]
    for i in range(len(top3)):
        with cols[i]:
            st.markdown(f"""
            <div class='leaderboard-card'>
                <div class='rank-icon'>{medals[i]}</div>
                <h3 style='margin:5px 0; color:#0f172a;'>{top3.loc[i, 'Method']}</h3>
                <p style='color:#64748b; margin:0;'>{top3.loc[i, 'Model']} + {top3.loc[i, 'Sampler']}</p>
                <h2 style='color:#0ea5e9; margin:10px 0;'>{top3.loc[i, 'S(α=0.5)']:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- TABS ---
    t1, t2, t3 = st.tabs(["🎯 AUC vs Interpretability", "🧩 Q vs I (Group Quality)", "🔬 Statistical Consensus"])
    
    # --- TAB 1: AUC vs I ---
    with t1:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig_p = px.scatter(main_df, x='AUC', y='I', color='Method', symbol='Model',
                             hover_data=['Sampler'], color_discrete_map=METHOD_COLORS,
                             title="Pareto Front (Accuracy vs. Interpretability)")
            fig_p.update_traces(marker=dict(size=14, opacity=0.8, line=dict(width=1, color='white')))
            fig_p.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig_p, use_container_width=True)
            
        with c2:
            st.markdown("#### Correlation Insight")
            if corr_df is not None:
                rho, p = corr_df['Spearman_rho'].iloc[0], corr_df['Spearman_p'].iloc[0]
                sig_text = "<span class='stat-sig'>Significant Trade-off</span>" if p < 0.05 else "<span class='stat-insig'>Independent Relationship</span>"
                st.markdown(f"""
                <div style='padding:15px; border:1px solid #e2e8f0; border-radius:8px;'>
                <b>Spearman ρ:</b> {rho:.3f} <br>
                <b>p-value:</b> {p:.3f} <br><br>
                <b>Status:</b> {sig_text}<br><br>
                <i>Analysis:</i> {'As accuracy increases, interpretability tends to decrease, confirming a strict trade-off in this dataset.' if p < 0.05 else 'Because the p-value > 0.05, Accuracy and Interpretability are independent. We can optimize explainability (via R-Myerson) without sacrificing the ensemble predictive power.'}
                </div>
                """, unsafe_allow_html=True)
            else: st.info("Correlation metrics pending.")

    # --- TAB 2: Q vs I ---
    with t2:
        st.markdown("#### Does better feature grouping lead to better explanations?")
        owen_df = main_df[main_df['Method'].isin(['Owen-Domain', 'Owen-Data', 'Owen-Model'])].dropna(subset=['Q', 'I'])
        
        if not owen_df.empty:
            qc1, qc2 = st.columns([1.5, 1])
            with qc1:
                fig_q = px.scatter(owen_df, x='Q', y='I', color='Method', hover_data=['Model'],
                                   color_discrete_map=METHOD_COLORS, trendline="ols",
                                   title="Group Quality (Q) vs Interpretability (I)")
                fig_q.update_traces(marker=dict(size=12))
                fig_q.update_layout(template="plotly_white", height=450)
                st.plotly_chart(fig_q, use_container_width=True)
            with qc2:
                # Calculate correlation for Q vs I using pandas (bypassing scipy dependency)
                q_rho = owen_df['Q'].corr(owen_df['I'], method='spearman')
                
                # Check for NaN in correlation (e.g. if variance is 0)
                if np.isnan(q_rho):
                    q_rho = 0.0
                
                st.markdown(f"""
                <div class='insight-box'>
                <b>Automated Analysis:</b><br>
                The Spearman correlation between Coalition Group Quality (Q) and Interpretability (I) is <b>ρ = {q_rho:.3f}</b>.
                <br><br>
                {f'Since there is a strong positive correlation, it proves that forming highly dependent feature coalitions directly improves the stability of the Owen values.' if q_rho > 0.3 else 'The relationship is weak, suggesting that while grouping matters, the mathematical allocation rule (like Myerson) plays a larger role in stability.'}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No valid Q-Score data available for this dataset.")

    # --- TAB 3: STATISTICAL CONSENSUS ---
    with t3:
        st.markdown("#### Rigorous Pairwise Comparison")
        st.markdown("We establish a **True Difference** only if a method pair is statistically significant in *both* the pairwise Wilcoxon test and the multi-group Nemenyi test (p < 0.05).")
        
        sc1, sc2 = st.columns([1, 1])
        
        with sc1:
            if wil_df is not None and nem_df is not None:
                # Build Consensus Table
                consensus_data = []
                for _, row in wil_df.iterrows():
                    m1, m2 = row['Method1'], row['Method2']
                    eff = row['Effect_size']
                    w_p = row['p_value']
                    
                    # Safe Nemenyi lookup
                    n_p = np.nan
                    if m1 in nem_df.index and m2 in nem_df.columns: n_p = nem_df.loc[m1, m2]
                    elif m2 in nem_df.index and m1 in nem_df.columns: n_p = nem_df.loc[m2, m1]
                    
                    consensus_data.append({
                        "Method 1": m1, "Method 2": m2,
                        "Effect Size": eff.title(),
                        "True Difference": get_consensus(w_p, n_p)
                    })
                
                st.dataframe(
                    pd.DataFrame(consensus_data).style
                    .map(color_effect, subset=['Effect Size'])
                    .map(color_consensus, subset=['True Difference']),
                    hide_index=True, height=450, use_container_width=True
                )
            else: st.warning("Requires both Wilcoxon and Nemenyi CSVs.")
            
        with sc2:
            st.markdown("<div style='text-align:center'><b>Nemenyi Post-hoc Heatmap</b></div>", unsafe_allow_html=True)
            if nem_df is not None:
                # Custom colorscale: Red for significant (<0.05), Blue for non-significant
                fig_nem = px.imshow(nem_df, text_auto=".3f", 
                                    color_continuous_scale=[[0, '#ef4444'], [0.05, '#fca5a5'], [0.051, '#bfdbfe'], [1, '#1e3a8a']], 
                                    zmin=0, zmax=0.2)
                fig_nem.update_layout(height=450, margin=dict(t=10, b=0), coloraxis_showscale=False)
                st.plotly_chart(fig_nem, use_container_width=True)
                st.markdown("<small><b>How to read:</b> <span style='color:#ef4444; font-weight:bold;'>Red tiles</span> indicate a p-value < 0.05, meaning the two methods produce statistically distinct explanations.</small>", unsafe_allow_html=True)
            else: st.warning("Nemenyi data not found.")
