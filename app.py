# ... existing code ...
        # Speed up animation slightly
        fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1200
        st.plotly_chart(fig_anim, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Global Explainer Stability (Mean S-Score)")
        
        # Aggregate mean values for the bar chart
        summary_df = combined.groupby(['Dataset', 'Method', 'Imbalance'])['S(α=0.5)'].mean().reset_index()
        summary_df = summary_df.sort_values('Imbalance', ascending=False)
        
        # Professional Bar Chart highlighting S scores
        fig_bar = px.bar(
            summary_df, 
            x='Dataset', 
            y='S(α=0.5)', 
            color='Method', 
            barmode='group',
            color_discrete_map=METHOD_COLORS,
            category_orders={"Dataset": summary_df['Dataset'].unique().tolist()}
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
# VIEW 2: SPECIFIC DATASET DASHBOARD
# ==========================================
# ... existing code ...
                fig_nem = px.imshow(nem_df, text_auto=".3f", color_continuous_scale=colorscale, zmin=0, zmax=1.0)
                fig_nem.update_layout(height=450, margin=dict(t=10, b=0, l=0, r=0), coloraxis_showscale=False)
                st.plotly_chart(fig_nem, use_container_width=True)
                st.markdown("<small><b>How to read:</b> <span style='color:#10b981; font-weight:bold;'>Green cells (p < 0.05)</span> indicate that the two methods are statistically significantly different. Gray cells indicate no significant difference.</small>", unsafe_allow_html=True)
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
