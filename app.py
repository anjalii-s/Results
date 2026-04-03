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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    
    /* Headers */
    h1, h2, h3 { color: #0f172a; font-weight: 700; letter-spacing: -0.5px; }
    
    /* Custom Metric Cards */
    .metric-card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #0369a1; margin: 10px 0; }
    .metric-label { font-size: 0.9rem; color: #64748b; font-weight: 500; text-transform: uppercase; }
    
    /* Leaderboard Cards */
    .lb-card { padding: 20px; border-radius: 12px; text-align: center; transition: transform 0.2s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .lb-card:hover { transform: translateY(-5px); }
    .lb-rank { font-size: 2rem; margin-bottom: 5px; }
    .lb-method { font-size: 1.3rem; font-weight: 700; color: #0f172a; margin: 0; }
    .lb-config { font-size: 0.9rem; color: #475569; margin: 5px 0 15px 0; }
    .lb-score { font-size: 1.5rem; font-weight: 700; color: #0369a1; margin: 0; }
    
    /* Insight Box */
    .insight-box { background-color: #ffffff; border-left: 4px solid #0369a1; padding: 15px 20px; border-radius: 6px; margin: 15px 0; color: #334155; font-size: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# CONSTANTS & CONFIGURATION (Professional Palette)
# ==========================================
METHOD_COLORS = {
    'SHAP': '#64748b',         # Slate (Baseline)
    'Banzhaf': '#d97706',      # Amber
    'Myerson': '#059669',      # Emerald
    'Owen-Domain': '#dc2626',  # Red
    'Owen-Data': '#7c3aed',    # Purple
    'Owen-Model': '#db2777',   # Fuchsia
    'R-Myerson': '#0284c7'     # Sky Blue (Proposed)
}

DATASET_REGISTRY = {
    "German Credit": {
        "main": "german_results_7methods.csv",
        "wilcoxon": "german_wilcoxon_cliffs_results.csv",
        "nemenyi": "german_nemenyi_results.csv",
        "corr": "german_auc_I_correlation.csv",
        "label": "High Default Rate",
        "imb": 30.0
    },
    "Taiwan Credit": {
        "main": "taiwan_results_7methods_Salpha (1).csv",
        "wilcoxon": "taiwan_wilcoxon_cliffs_results (2).csv",
        "nemenyi": "NEMENYI TAIWAN NEW.csv",
        "corr": "taiwan_auc_I_correlation (2).csv",
        "label": "Moderate Default Rate",
        "imb": 22.12
    },
    "LC10%": {
        "main": "LC10pcdefault.csv",
        "wilcoxon": "lc10_wilcoxon_cliffs_results (1).csv",
        "nemenyi": "lc10_nemenyi_results (1).csv",
        "corr": "lc10_auc_I_correlation (1).csv",
        "label": "Industry Standard",
        "imb": 10.0
    },
    "LC (B)": {
        "main": "LC66_results_7methods_noleak (2) (1).csv",
        "wilcoxon": "lc66WILCOXONCLIFFS.csv",
        "nemenyi": "Lc66_nemenyi_results (1).csv",
        "corr": "Lc66_correlation.csv",
        "label": "Severe Imbalance",
        "imb": 4.01
    },
    "Coursera Loans": {
        "main": "coursera_loans_results_7methods.csv",
        "wilcoxon": "wilcoxon_cliffs_results_coursera.csv",
        "nemenyi": "nemenyi_results_coursera.csv",
        "corr": "auc_I_correlation_coursera.csv",
        "label": "Extreme Imbalance",
        "imb": 1.0
    }
}

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
@st.cache_data
def load_data(path, is_index=False):
    """Robust file loader handling missing files and cleaning 'nan' Samplers."""
    variations = [path, path.replace(' .csv', '.csv'), path.replace('.csv', ' (1).csv'), path.replace('.csv', ' (2).csv')]
    for v in variations:
        if os.path.exists(v):
            try:
                df = pd.read_csv(v, index_col=0 if is_index else None)
                if 'Sampler' in df.columns:
                    # Clean the Sampler column strictly
                    df['Sampler'] = df['Sampler'].astype(str).replace(['nan', 'NaN', 'None', 'nan '], 'None')
                return df
            except Exception: pass
    return None

def color_effect(val):
    v = str(val).lower()
    if v == 'large': return 'color: #059669; font-weight: bold;'
    if v == 'medium': return 'color: #d97706; font-weight: bold;'
    return 'color: #64748b;'

def color_consensus(val):
    if '✓' in str(val): return 'background-color: #dcfce7; color: #166534; font-weight: bold;'
    return 'color: #94a3b8;'

def get_wilcoxon_sig(sig_val, p_val):
    """Safely determines if Wilcoxon is significant."""
    sig_str = str(sig_val).lower()
    if '✓' in sig_str or 'yes' in sig_str or 'true' in sig_str: return True
    try:
        return float(p_val) < 0.05
    except:
        return False

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("### 🧭 Navigation")
views = ["📊 Cross-Dataset Synthesis", "🏆 Leaderboards"] + list(DATASET_REGISTRY.keys())
selection = st.sidebar.radio("Select View:", views)
st.sidebar.markdown("---")
st.sidebar.caption("Ensemble Learning & Coalition-aware Explainability for Imbalanced Credit Default.")

# ==========================================
# VIEW 1: CROSS-DATASET SYNTHESIS
# ==========================================
if selection == "📊 Cross-Dataset Synthesis":
    st.title("Ensemble Learning and Coalition-aware Explainability for Imbalanced Credit Default")
    
    st.markdown("""
    <div class='insight-box'>
    <b>Executive Summary:</b> This dashboard unifies the results of seven attribution methods across five financial datasets. 
    By pressing the <b>Play</b> button below, you can visually track how standard Explainable AI (XAI) methods degrade as the dataset becomes increasingly imbalanced (from 30% down to 1% default rate), highlighting the robustness of the <b>R-Myerson</b> algorithm.
    </div>
    """, unsafe_allow_html=True)
    
    global_results = []
    for name, cfg in DATASET_REGISTRY.items():
        df = load_data(cfg['main'])
        if df is not None:
            df_copy = df.copy()
            df_copy['Imbalance'] = cfg['imb']
            df_copy['Dataset'] = f"{name} ({cfg['imb']}%)"
            df_copy['Config'] = df_copy['Method'] + "_" + df_copy['Model'] + "_" + df_copy['Sampler']
            global_results.append(df_copy)
            
    if global_results:
        # Combine and sort so the animation flows from highest to lowest imbalance
        combined = pd.concat(global_results).sort_values('Imbalance', ascending=False)
        
        st.subheader("Dynamic Pareto Front Shift Across Imbalance Levels")
        
        # Animated Bubble Chart
        fig_anim = px.scatter(
            combined, 
            x="AUC", y="I", 
            animation_frame="Dataset", 
            animation_group="Config",
            color="Method", 
            symbol="Model",
            size="S(α=0.5)",
            hover_name="Sampler",
            color_discrete_map=METHOD_COLORS,
            range_x=[combined['AUC'].min() - 0.02, combined['AUC'].max() + 0.02],
            range_y=[0.0, 1.05]
        )
        
        fig_anim.update_traces(marker=dict(line=dict(width=1, color='white')), opacity=0.85)
        fig_anim.update_layout(
            template="plotly_white", 
            height=600,
            xaxis_title="Predictive Accuracy (AUC)",
            yaxis_title="Interpretability (I-Score)"
        )
        # Speed up animation slightly
        fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1200
        st.plotly_chart(fig_anim, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Performance-Interpretability(Mean S-Score)")
        st.caption("Bar heights represent the **average** S-score for a given method across all base models and samplers for that dataset.")
        
        # Aggregate mean values for the bar chart
        summary_df = combined.groupby(['Dataset', 'Method', 'Imbalance'])['S(α=0.5)'].mean().reset_index()
        summary_df = summary_df.sort_values('Imbalance', ascending=False)
        
        # Calculate overall method sorting order (highest to lowest score)
        method_order = summary_df.groupby('Method')['S(α=0.5)'].mean().sort_values(ascending=False).index.tolist()
        
        # Professional Bar Chart highlighting S scores ordered high to low
        fig_bar = px.bar(
            summary_df, 
            x='Dataset', 
            y='S(α=0.5)', 
            color='Method', 
            barmode='group',
            color_discrete_map=METHOD_COLORS,
            category_orders={
                "Dataset": summary_df['Dataset'].unique().tolist(),
                "Method": method_order
            }
        )
        
        # Keeping styling consistent
        fig_bar.update_traces(marker_line_width=1, marker_line_color="white")
        
        fig_bar.update_layout(
            xaxis_title="Datasets (Decreasing Default Rate →)", 
            yaxis_title="Mean S(α=0.5) Score",
            template="plotly_white",
            hovermode="x unified",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# VIEW 2: LEADERBOARDS
# ==========================================
elif selection == "🏆 Leaderboards":
    st.title("🏆 Global & Per-Dataset Leaderboards")
    st.markdown("Dynamic rankings of top model-sampler configurations.")
    
    global_results = []
    for name, cfg in DATASET_REGISTRY.items():
        df = load_data(cfg['main'])
        if df is not None:
            df_copy = df.copy()
            df_copy['Dataset_Name'] = name
            df_copy['Imbalance'] = cfg['imb']
            df_copy['Config'] = df_copy['Model'] + '–' + df_copy['Sampler'].fillna('None')
            global_results.append(df_copy)
            
    if global_results:
        combined = pd.concat(global_results)
        metrics_of_interest = ['AUC', 'I', 'S(α=0.5)']
        
        st.markdown("### 🌍 Top Configurations Aggregated Across All Datasets")
        st.caption("Calculated dynamically by **averaging** performance across all 5 datasets and all 7 explainability methods. Ranked in descending order.")
        
        overall_df = pd.DataFrame({"Rank": range(1, 6)})
        for metric in metrics_of_interest:
            top_5 = combined.groupby('Config')[metric].mean().reset_index()
            top_5 = top_5.sort_values(by=metric, ascending=False).head(5).reset_index(drop=True)
            overall_df[f"Config ({metric})"] = top_5['Config']
            overall_df[f"{metric} Score"] = top_5[metric].apply(lambda x: f"{x:.3f}")
            
        st.dataframe(overall_df, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Top Configurations Per Dataset")
        st.caption("Calculated by **averaging** the performance of Model-Sampler configurations across all 7 explainability methods for each specific dataset. Ranked in descending order.")
        
        # Render a clean table for each individual dataset dynamically
        for ds_name in DATASET_REGISTRY.keys():
            ds_matches = combined[combined['Dataset_Name'] == ds_name]
            if not ds_matches.empty:
                st.markdown(f"#### {ds_name} *(Imbalance: {DATASET_REGISTRY[ds_name]['imb']}%)*")
                ds_df = pd.DataFrame({"Rank": range(1, 4)})
                
                for metric in metrics_of_interest:
                    top_3 = ds_matches.groupby('Config')[metric].mean().reset_index()
                    top_3 = top_3.sort_values(by=metric, ascending=False).head(3).reset_index(drop=True)
                    ds_df[f"Config ({metric})"] = top_3['Config']
                    ds_df[f"{metric} Score"] = top_3[metric].apply(lambda x: f"{x:.3f}")
                    
                st.dataframe(ds_df, hide_index=True, use_container_width=True)

# ==========================================
# VIEW 3: SPECIFIC DATASET DASHBOARD
# ==========================================
else:
    cfg = DATASET_REGISTRY[selection]
    st.title(f"{selection}")
    st.caption(f"{cfg['label']} ({cfg['imb']}%)")
    
    # Load all files for the dataset
    main_df = load_data(cfg['main'])
    wil_df = load_data(cfg['wilcoxon'])
    nem_df = load_data(cfg['nemenyi'], is_index=True)
    corr_df = load_data(cfg['corr'])
    
    if main_df is None:
        st.error(f"⚠️ Primary results file ({cfg['main']}) not found in repository.")
        st.stop()

    # --- TOP 3 LEADERBOARD (ABSOLUTE) ---
    st.markdown("### 🏆 Top 3 Configurations (Absolute Peak Scores)")
    st.caption("Ranked in descending order by the single highest **absolute peak** S(α=0.5) score achieved by any specific row (Model + Sampler + Method).")
    top3 = main_df.sort_values('S(α=0.5)', ascending=False).head(3).reset_index(drop=True)
    
    cols = st.columns(3)
    # Inline CSS styles for guaranteed rendering of the podium colors
    bg_styles = [
        "background: linear-gradient(180deg, #fef08a 0%, #ffffff 100%); border: 2px solid #fde047;", # Gold
        "background: linear-gradient(180deg, #e2e8f0 0%, #ffffff 100%); border: 2px solid #cbd5e1;", # Silver
        "background: linear-gradient(180deg, #fed7aa 0%, #ffffff 100%); border: 2px solid #fdba74;"  # Bronze
    ]
    medals = ["🥇 1st Place", "🥈 2nd Place", "🥉 3rd Place"]
    
    for i in range(len(top3)):
        with cols[i]:
            st.markdown(f"""
            <div class='lb-card' style='{bg_styles[i]}'>
                <div class='lb-rank'>{medals[i]}</div>
                <h3 class='lb-method'>{top3.loc[i, 'Method']}</h3>
                <p class='lb-config'>{top3.loc[i, 'Model']} + {top3.loc[i, 'Sampler']}</p>
                <h2 class='lb-score'>{top3.loc[i, 'S(α=0.5)']:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)

    # --- TABS NAVIGATION ---
    t1, t2, t3, t4, t5 = st.tabs([
        "🎯 Accuracy vs Interpretability", 
        "🧩 Q vs I Analysis", 
        "🔬 Statistical Significance", 
        "🏅 Top Model-Samplers",
        "🗄️ Raw Data"
    ])
    
    # ==================================
    # TAB 1: AUC vs I (PARETO)
    # ==================================
    with t1:
        c1, c2 = st.columns([1.8, 1])
        with c1:
            fig_p = px.scatter(main_df, x='AUC', y='I', color='Method', symbol='Model',
                             hover_data=['Sampler'], color_discrete_map=METHOD_COLORS,
                             title="Pareto Front: Accuracy vs. Interpretability")
            fig_p.update_traces(marker=dict(size=14, opacity=0.85, line=dict(width=1, color='white')))
            fig_p.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig_p, use_container_width=True)
            
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
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
        
        owen_df = main_df[main_df['Method'].isin(['Owen-Domain', 'Owen-Data', 'Owen-Model'])].copy()
        owen_clean = owen_df.dropna(subset=['Q', 'I']).copy()
        
        if len(owen_clean) >= 3:
            qc1, qc2 = st.columns([1.8, 1])
            with qc1:
                # Removed trendline, kept symbol shape mapping for Model clarity
                fig_q = px.scatter(owen_clean, x='Q', y='I', color='Method',
                                   symbol='Model', hover_data=['Sampler'], 
                                   color_discrete_map=METHOD_COLORS,
                                   title="Group Quality (Q) vs Interpretability (I)")
                fig_q.update_traces(marker=dict(size=14, line=dict(width=1, color='white')))
                fig_q.update_layout(template="plotly_white", height=450)
                st.plotly_chart(fig_q, use_container_width=True)
            
            with qc2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                q_rho = owen_clean['Q'].corr(owen_clean['I'], method='spearman')
                if np.isnan(q_rho): q_rho = 0.0
                
                st.markdown(f"""
                <div class='insight-box'>
                <b>Derivation Method:</b> This relationship is derived using the <b>Spearman rank correlation coefficient (ρ)</b>. We pair the Group Quality (Q) and Interpretability (I) scores for each configuration of the Owen variants, rank them, and measure their monotonic relationship. This non-parametric approach ensures robustness against outliers and non-linear patterns.
                <br><br>
                <b>Resulting Analysis:</b><br>
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
        # Display AUC-I Correlations prominently
        if corr_df is not None and not corr_df.empty:
            c_rho, c_p = corr_df['Spearman_rho'].iloc[0], corr_df['Spearman_p'].iloc[0]
            k_tau, k_p = corr_df['Kendall_tau'].iloc[0], corr_df['Kendall_p'].iloc[0]
            
            st.markdown("### Accuracy vs. Interpretability Statistical Correlation")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Spearman ρ</div>
                    <div class='metric-value'>{c_rho:.3f}</div>
                    <div style='color: #64748b; font-size:0.85rem;'>p-value: {c_p:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Kendall τ</div>
                    <div class='metric-value'>{k_tau:.3f}</div>
                    <div style='color: #64748b; font-size:0.85rem;'>p-value: {k_p:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Rigorous Pairwise Comparison")
        st.markdown("A *True Consensus Difference* is established only if BOTH the pairwise Wilcoxon test AND the multi-comparison Nemenyi test confirm significance ($p < 0.05$).")
        
        sc1, sc2 = st.columns([1.1, 1])
        
        with sc1:
            st.markdown("#### Wilcoxon & Consensus Table")
            if wil_df is not None and nem_df is not None:
                consensus_data = []
                for _, row in wil_df.iterrows():
                    m1, m2 = row['Method1'], row['Method2']
                    eff = row['Effect_size']
                    
                    # Wilcoxon boolean conversion
                    w_is_sig = get_wilcoxon_sig(row.get('Significant', ''), row.get('p_value', 1.0))
                    w_display = "✓ Yes" if w_is_sig else "✗ No"
                    
                    # Safe Nemenyi lookup
                    n_p = 1.0
                    try: n_p = float(nem_df.loc[m1, m2])
                    except KeyError:
                        try: n_p = float(nem_df.loc[m2, m1])
                        except KeyError: pass
                    
                    n_sig_bool = n_p < 0.05
                    
                    consensus = "✓ Yes" if (w_is_sig and n_sig_bool) else "✗ No"
                    
                    consensus_data.append({
                        "Method 1": m1, "Method 2": m2,
                        "Wilcoxon Sig.": w_display,
                        "Effect Size": str(eff).title(),
                        "Consensus Diff": consensus
                    })
                
                st.dataframe(
                    pd.DataFrame(consensus_data).style
                    .map(color_effect, subset=['Effect Size'])
                    .map(color_consensus, subset=['Wilcoxon Sig.', 'Consensus Diff']),
                    hide_index=True, height=450, use_container_width=True
                )
            else: 
                st.warning("Both Wilcoxon and Nemenyi CSV files are required to display the consensus table.")
            
        with sc2:
            st.markdown("#### Nemenyi Post-hoc Heatmap")
            if nem_df is not None:
                # Intuitive Binary Colorscale: Green for significant (p < 0.05), Gray/White for insignificant
                colorscale = [
                    [0.0, '#10b981'],    # Green (Significant)
                    [0.049, '#10b981'],  # Green cutoff
                    [0.05, '#f1f5f9'],   # Light Slate (Insignificant)
                    [1.0, '#f1f5f9']     # Light Slate
                ]
                fig_nem = px.imshow(nem_df, text_auto=".3f", color_continuous_scale=colorscale, zmin=0, zmax=1.0)
                fig_nem.update_layout(height=450, margin=dict(t=10, b=0, l=0, r=0), coloraxis_showscale=False)
                st.plotly_chart(fig_nem, use_container_width=True)
                st.markdown("<small><b>How to read:</b> <span style='color:#10b981; font-weight:bold;'>Green cells (p < 0.05)</span> indicate that the two methods are statistically significantly different. Gray cells indicate no significant difference.</small>", unsafe_allow_html=True)
            else: 
                st.warning("Nemenyi data not found.")

    # ==================================
    # TAB 4: TOP MODEL-SAMPLERS
    # ==================================
    with t4:
        st.markdown("### Top 5 Configurations (Averaged Across Methods)")
        st.caption("Ranked in descending order by their **average** score across all 7 XAI methods. This identifies the most consistently robust predictive pipelines.")
        
        main_df['Model_Sampler'] = main_df['Model'] + '_' + main_df['Sampler'].fillna('None')
        metrics_of_interest = ['AUC', 'I', 'S(α=0.5)']
        
        m_cols = st.columns(3)
        for i, metric in enumerate(metrics_of_interest):
            with m_cols[i]:
                st.markdown(f"<h4 style='text-align: center; color: #0f172a;'>Top 5 {metric}</h4>", unsafe_allow_html=True)
                top_5 = main_df.groupby('Model_Sampler')[metric].mean().reset_index()
                top_5 = top_5.sort_values(by=metric, ascending=False).head(5)
                
                st.dataframe(top_5.style.format({metric: "{:.4f}"}), hide_index=True, use_container_width=True)
                
                avg_best = top_5[metric].mean()
                st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #475569;'><b>Average of Top 5:</b> {avg_best:.4f}</div>", unsafe_allow_html=True)

    # ==================================
    # TAB 5: RAW DATA
    # ==================================
    with t5:
        st.markdown("### Raw Analytical Data")
        st.dataframe(main_df, use_container_width=True)
        csv_data = main_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Raw CSV", data=csv_data, file_name=f"{selection}_data.csv", mime="text/csv")
