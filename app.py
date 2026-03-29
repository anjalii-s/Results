import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# PAGE CONFIGURATION & PROFESSIONAL CSS
# ==========================================
st.set_page_config(
    page_title="Credit Risk XAI Framework",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    
    /* Headers */
    h1, h2, h3 { color: #0f172a; font-weight: 700; letter-spacing: -0.5px; }
    
    /* Podium / Leaderboard Styles */
    .podium-container { display: flex; justify-content: center; align-items: flex-end; gap: 15px; margin-top: 20px; margin-bottom: 40px; }
    .podium-card { background: white; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); border: 1px solid #e2e8f0; flex: 1; transition: transform 0.3s ease; }
    .podium-card:hover { transform: translateY(-8px); border-color: #cbd5e1; }
    .podium-rank { font-size: 2.5rem; margin-bottom: 10px; }
    .podium-method { font-size: 1.25rem; font-weight: 700; color: #1e293b; margin: 0; }
    .podium-config { font-size: 0.85rem; color: #64748b; margin-top: 4px; margin-bottom: 15px; }
    .podium-score { font-size: 1.5rem; font-weight: 700; color: #0ea5e9; margin: 0; }
    
    /* Card/Insight Box */
    .insight-box { background-color: #ffffff; border-left: 4px solid #3b82f6; padding: 15px 20px; border-radius: 6px; margin: 15px 0; color: #334155; font-size: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .sig-green { color: #16a34a; font-weight: 700; }
    .sig-red { color: #dc2626; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# CONSTANTS & CONFIGURATION (Professional Muted Colors)
# ==========================================
METHOD_COLORS = {
    'SHAP': '#64748b',         # Muted Slate
    'Banzhaf': '#d97706',      # Muted Amber
    'Myerson': '#16a34a',      # Muted Green
    'Owen-Domain': '#dc2626',  # Muted Red
    'Owen-Data': '#7c3aed',    # Muted Purple
    'Owen-Model': '#c026d3',   # Muted Fuchsia
    'R-Myerson': '#0284c7'     # Muted Cyan (Proposed)
}

DATASET_REGISTRY = {
    "German Credit": {
        "main": "german_results_7methods.csv",
        "wilcoxon": "german_wilcoxon_cliffs_results.csv",
        "nemenyi": "german_nemenyi_results.csv",
        "corr": "german_auc_I_correlation.csv",
        "label": "High Default Rate (30%)",
        "imb": 30.0
    },
    "Taiwan Credit": {
        "main": "taiwan_results_7methods_S.csv",
        "wilcoxon": "taiwan_wilcoxon_cliffs_results.csv",
        "nemenyi": "taiwan_nemenyi_results.csv",
        "corr": "taiwan_auc_I_correlation.csv",
        "label": "Moderate Default Rate (22.12%)",
        "imb": 22.12
    },
    "Lending Club": {
        "main": "LC10pcdefaultresults.csv",
        "wilcoxon": "lc10_wilcoxon_cliffs_results.csv",
        "nemenyi": "lc10_nemenyi_results.csv",
        "corr": "lc10_auc_I_correlation.csv",
        "label": "Industry Standard (10%)",
        "imb": 10.0
    },
    "Lending Club LC66": {
        "main": "LC66_results_7methods_noleak.csv",
        "wilcoxon": "Lc66_wilcoxon_cliffs_results .csv", 
        "nemenyi": "Lc66_nemenyi_results.csv",
        "corr": "Lc66_correlation.csv",
        "label": "Severe Imbalance (4.01%)",
        "imb": 4.01
    },
    "Coursera Loans": {
        "main": "coursera_loans_results_7methods.csv",
        "wilcoxon": "wilcoxon_cliffs_results_coursera.csv",
        "nemenyi": "nemenyi_results_coursera.csv",
        "corr": "auc_I_correlation_coursera.csv",
        "label": "Extreme Imbalance (1%)",
        "imb": 1.0
    }
}

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
@st.cache_data
def load_data(path, is_index=False):
    """Robust file loader that handles missing files and cleans 'nan' Samplers."""
    variations = [path, path.replace(' .csv', '.csv'), path.replace('.csv', ' (1).csv')]
    for v in variations:
        if os.path.exists(v):
            try:
                df = pd.read_csv(v, index_col=0 if is_index else None)
                if 'Sampler' in df.columns:
                    # Clean the Sampler column
                    df['Sampler'] = df['Sampler'].astype(str).replace(['nan', 'NaN'], 'None')
                return df
            except Exception: pass
    return None

def color_effect(val):
    """Colors the Cliff's Delta Effect size text."""
    v = str(val).lower()
    if v == 'large': return 'color: #16a34a; font-weight: bold;'
    if v == 'medium': return 'color: #d97706; font-weight: bold;'
    return 'color: #64748b;'

def color_bool(val):
    """Colors boolean text outputs."""
    if '✓' in str(val) or 'Yes' in str(val): return 'color: #16a34a; font-weight: bold;'
    return 'color: #dc2626;'

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("### 🧭 Navigation")
views = ["📊 Cross-Dataset Comparison"] + list(DATASET_REGISTRY.keys())
selection = st.sidebar.radio("Select View:", views)
st.sidebar.markdown("---")
st.sidebar.caption("Ensemble Learning & Coalition-aware Explainability for Imbalanced Credit Default.")

# ==========================================
# VIEW 1: CROSS-DATASET COMPARISON
# ==========================================
if selection == "📊 Cross-Dataset Comparison":
    st.title("Ensemble Learning and Coalition-aware Explainability for Imbalanced Credit Default")
    
    st.markdown("""
    <div class='insight-box'>
    <b>Executive Summary:</b> This dashboard unifies the results of seven attribution methods across five financial datasets. 
    It demonstrates how standard Explainable AI (XAI) methods degrade in highly imbalanced domains, and highlights the robustness 
    of the <b>R-Myerson</b> algorithm in maintaining stability without compromising accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    # Compile global dataset
    global_results = []
    for name, cfg in DATASET_REGISTRY.items():
        df = load_data(cfg['main'])
        if df is not None:
            summary = df.groupby('Method')['S(α=0.5)'].mean().reset_index()
            summary['Imbalance'] = cfg['imb']
            summary['Dataset'] = f"{name} ({cfg['imb']}%)"
            global_results.append(summary)
            
    if global_results:
        combined = pd.concat(global_results).sort_values('Imbalance', ascending=False)
        
        st.subheader("Global Explainer Stability across Default Rates")
        st.markdown("Observe how the S-Score (Performance-Interpretability balance) fluctuates as default rates become extreme (moving from left to right).")
        
        # Grouped Bar Chart instead of line graph for better visual impact
        fig_bar = px.bar(combined, x='Dataset', y='S(α=0.5)', color='Method', 
                         color_discrete_map=METHOD_COLORS, barmode='group',
                         title="Mean S-Score Comparison across Imbalance Levels")
        fig_bar.update_layout(
            xaxis_title="Datasets (Sorted by Decreasing Default Rate →)", 
            yaxis_title="Mean S(α=0.5) Score",
            template="plotly_white",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# VIEW 2: SPECIFIC DATASET DASHBOARD
# ==========================================
else:
    cfg = DATASET_REGISTRY[selection]
    st.title(f"{selection}")
    st.markdown(f"**Dataset Characteristic:** {cfg['label']}")
    
    # Load all files for the dataset
    main_df = load_data(cfg['main'])
    wil_df = load_data(cfg['wilcoxon'])
    nem_df = load_data(cfg['nemenyi'], is_index=True)
    corr_df = load_data(cfg['corr'])
    
    if main_df is None:
        st.error(f"⚠️ Primary results file (`{cfg['main']}`) not found in repository.")
        st.stop()

    # --- TOP 3 LEADERBOARD (PODIUM) ---
    st.markdown("### 🏆 Top 3 Configurations Leaderboard")
    top3 = main_df.sort_values('S(α=0.5)', ascending=False).head(3).reset_index(drop=True)
    
    # Create HTML Podium
    podium_html = "<div class='podium-container'>"
    medals = ["🥇 1st Place", "🥈 2nd Place", "🥉 3rd Place"]
    heights = ["100%", "90%", "80%"] # Visual hierarchy for the podium
    for i in range(len(top3)):
        podium_html += f"""
        <div class='podium-card' style='height: {heights[i]};'>
            <div class='podium-rank'>{medals[i]}</div>
            <p class='podium-method'>{top3.loc[i, 'Method']}</p>
            <p class='podium-config'>{top3.loc[i, 'Model']} + {top3.loc[i, 'Sampler']}</p>
            <p class='podium-score'>{top3.loc[i, 'S(α=0.5)']:.4f}</p>
        </div>
        """
    podium_html += "</div>"
    st.markdown(podium_html, unsafe_allow_html=True)

    # --- TABS NAVIGATION ---
    t1, t2, t3, t4 = st.tabs([
        "🎯 Accuracy vs Interpretability", 
        "🧩 Q vs I Analysis", 
        "🔬 Statistical Significance", 
        "🗄️ Raw Data"
    ])
    
    # ==================================
    # TAB 1: AUC vs I (PARETO)
    # ==================================
    with t1:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig_p = px.scatter(main_df, x='AUC', y='I', color='Method', symbol='Model',
                             hover_data=['Sampler'], color_discrete_map=METHOD_COLORS,
                             title="Pareto Front: Accuracy vs. Interpretability")
            fig_p.update_traces(marker=dict(size=14, opacity=0.85, line=dict(width=1, color='white')))
            fig_p.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig_p, use_container_width=True)
            
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            # Calculate and analyze AUC vs I using pandas to avoid scipy dependencies
            auc_i_rho = main_df['AUC'].corr(main_df['I'], method='spearman')
            st.markdown(f"""
            <div class='insight-box'>
            <b>Analysis of Trade-Off:</b><br><br>
            The Spearman rank correlation between predictive Accuracy (AUC) and Interpretability (I-Score) across all models here is <b>ρ = {auc_i_rho:.3f}</b>.<br><br>
            <i>Interpretation:</i> {'A negative correlation indicates a classical trade-off: highly accurate models are harder to interpret.' if auc_i_rho < -0.1 else 'A positive or near-zero correlation suggests that for this dataset, we can successfully extract highly stable explanations without sacrificing the predictive power of the ensemble.'}
            </div>
            """, unsafe_allow_html=True)

    # ==================================
    # TAB 2: Q vs I (GROUP QUALITY)
    # ==================================
    with t2:
        st.markdown("### Does better feature grouping lead to better explanations?")
        
        # Isolate Owen variants
        owen_df = main_df[main_df['Method'].isin(['Owen-Domain', 'Owen-Data', 'Owen-Model'])].copy()
        owen_clean = owen_df.dropna(subset=['Q', 'I']).copy()
        
        if len(owen_clean) >= 3:
            qc1, qc2 = st.columns([1.5, 1])
            with qc1:
                # Removed trendline and added symbol mapping to Model as requested
                fig_q = px.scatter(owen_clean, x='Q', y='I', color='Method',
                                   symbol='Model', hover_data=['Sampler'], 
                                   color_discrete_map=METHOD_COLORS,
                                   title=f"Group Quality (Q) vs Interpretability (I) — n={len(owen_clean)} points")
                fig_q.update_traces(marker=dict(size=14, line=dict(width=1, color='white')))
                fig_q.update_layout(template="plotly_white", height=500)
                st.plotly_chart(fig_q, use_container_width=True)
            
            with qc2:
                # Robust pandas spearman correlation calculation
                q_rho = owen_clean['Q'].corr(owen_clean['I'], method='spearman')
                if np.isnan(q_rho): q_rho = 0.0
                
                st.markdown(f"""
                <div class='insight-box'>
                <b>Group Quality Analysis:</b><br><br>
                The Spearman rank correlation is <b>ρ = {q_rho:.3f}</b>.<br><br>
                <i>Interpretation:</i> {'A strong positive relationship confirms that algorithmically defining better feature coalitions (higher Q) directly leads to more stable and consistent feature attributions (higher I).' if q_rho > 0.3 else 'The relationship is weak in this dataset, indicating that the baseline distribution rules impact stability more than the coalition boundaries themselves.'}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("⚠️ Not enough valid Q and I values to compute a meaningful correlation for the Owen variants in this dataset.")

    # ==================================
    # TAB 3: STATISTICAL SIGNIFICANCE
    # ==================================
    with t3:
        st.markdown("### Rigorous Pairwise Comparison")
        st.markdown("This section determines which methods genuinely differ from one another. A difference is considered a **True Consensus Difference** only if BOTH the pairwise Wilcoxon test AND the multi-comparison Nemenyi test confirm significance ($p < 0.05$).")
        
        # Include AUC vs I stats from the uploaded csv here as requested
        if corr_df is not None and not corr_df.empty:
            c_rho, c_p = corr_df['Spearman_rho'].iloc[0], corr_df['Spearman_p'].iloc[0]
            st.markdown(f"**AUC-I Correlation Test Results:** Spearman ρ = `{c_rho:.3f}` (p-value: `{c_p:.3f}`) | Kendall τ = `{corr_df['Kendall_tau'].iloc[0]:.3f}` (p-value: `{corr_df['Kendall_p'].iloc[0]:.3f}`)")
            st.divider()

        sc1, sc2 = st.columns([1, 1.2])
        
        with sc1:
            st.markdown("#### Wilcoxon & Consensus Table")
            if wil_df is not None and nem_df is not None:
                # Combine Wilcoxon and Nemenyi into a clean, reduced table
                consensus_data = []
                for _, row in wil_df.iterrows():
                    m1, m2 = row['Method1'], row['Method2']
                    eff = row['Effect_size']
                    w_sig = str(row['Significant']).strip()
                    
                    # Fetch Nemenyi p-value safely
                    n_p = 1.0
                    try:
                        n_p = float(nem_df.loc[m1, m2])
                    except KeyError:
                        try: n_p = float(nem_df.loc[m2, m1])
                        except KeyError: pass
                    
                    n_sig = "✓" if n_p < 0.05 else "✗"
                    consensus = "✓ Yes" if ('✓' in w_sig and n_sig == '✓') else "✗ No"
                    
                    consensus_data.append({
                        "Method 1": m1, "Method 2": m2,
                        "Wilcoxon Sig.": w_sig,
                        "Effect Size": eff.title(),
                        "True Consensus": consensus
                    })
                
                st.dataframe(
                    pd.DataFrame(consensus_data).style
                    .map(color_effect, subset=['Effect Size'])
                    .map(color_bool, subset=['Wilcoxon Sig.', 'True Consensus']),
                    hide_index=True, height=450, use_container_width=True
                )
            else: 
                st.warning("Both Wilcoxon and Nemenyi CSV files are required to display the consensus table.")
            
        with sc2:
            st.markdown("#### Nemenyi Post-hoc Heatmap")
            if nem_df is not None:
                # Fixed colorscale: Green for significant (p < 0.05), Red for not significant (p >= 0.05)
                # This aligns with the user's request for intuitive reading
                colorscale = [
                    [0.0, '#16a34a'],    # Deep Green for 0.0 (Highly significant)
                    [0.049, '#4ade80'],  # Light Green for just under 0.05
                    [0.05, '#fca5a5'],   # Light Red for 0.05
                    [1.0, '#dc2626']     # Deep Red for 1.0 (Not significant at all)
                ]
                fig_nem = px.imshow(nem_df, text_auto=".3f", color_continuous_scale=colorscale, zmin=0, zmax=1.0)
                fig_nem.update_layout(height=450, margin=dict(t=10, b=0), coloraxis_showscale=False)
                st.plotly_chart(fig_nem, use_container_width=True)
                st.markdown("<small><b>How to read:</b> <span class='sig-green'>Green cells (p < 0.05)</span> indicate that the two methods are statistically significantly different. <span class='sig-red'>Red cells (p ≥ 0.05)</span> indicate no significant difference.</small>", unsafe_allow_html=True)
            else: 
                st.warning("Nemenyi data not found.")

    # ==================================
    # TAB 4: RAW DATA
    # ==================================
    with t4:
        st.markdown("### Raw Analytical Data")
        st.dataframe(main_df, use_container_width=True)
        csv_data = main_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Raw CSV", data=csv_data, file_name=f"{selection}_data.csv", mime="text/csv")
