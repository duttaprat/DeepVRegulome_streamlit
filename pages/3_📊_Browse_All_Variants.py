import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from PIL import Image

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Browse Variants")
S3_BASE_URL = "https://deepvregulome-attention-maps-2025.s3.us-east-2.amazonaws.com/"
# --- Sidebar Controls ---
st.sidebar.header("Global Controls")
cancer_type = st.sidebar.selectbox("Select a Cancer Type:", ("Brain", "Breast", "Lung"), key="cancer_type_browser")
analysis_type = st.sidebar.selectbox("Genomic Regulatory Elements", ["Splice Sites", "TFBS Models"], key="analysis_type_browser")
variant_options = { "Substitutions (SNVs)": "CaVEMan", "Insertions & Deletions (Indels)": "sanger_raw_pindel" }
selected_variant_label = st.sidebar.selectbox("Select Variant Type", options=list(variant_options.keys()), key="variant_type_browser")
data_source = variant_options[selected_variant_label]

# --- Data Loading Function (UPDATED) ---
@st.cache_data
def load_data(cancer, analysis, source):
    """
    Loads data conditionally based on the analysis type.
    - For TFBS, loads the pre-merged master file.
    - For Splice Sites, loads and merges the original separate files.
    """
    base_path = f"data/{cancer}/"
    analysis_folder_map = {"Splice Sites": "Splice_Sites", "TFBS Models": "TFBS_Models"}
    analysis_folder = analysis_folder_map[analysis]
    tsv_path = f"{base_path}{analysis_folder}/"
    
    try:
        df_clinical = pd.read_csv(f"{base_path}patient_clinical_updated.tsv", sep='\t')
        df_clinical = df_clinical.dropna(subset=['manifest_patient_id', 'km_time', 'km_status'])
        master_file_path = f"{tsv_path}{source}_master_file.tsv"
        print(f"Loading {analysis_folder} master file: {master_file_path}")
        df_variants = pd.read_csv(master_file_path, sep="\t", low_memory=False)
        

        if analysis == "TFBS Models":
            df_tfbs_summary = pd.read_csv("data/TFBS_Summary.tsv", sep="\t")
        elif analysis == "Splice Sites":
            df_tfbs_summary = None
        

        return df_variants, df_clinical, df_tfbs_summary
        
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found. Please check your repository. Details: {e}")
        return None, None, None

# --- Plotting Functions (No Changes Needed) ---
def plot_km_curve(group_A, group_B, variant_id, p_value):
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    kmf.fit(group_A['km_time'], event_observed=group_A['km_status'], label=f"Wild-Type (n={len(group_A)})")
    fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0], mode='lines', name=f"Wild-Type (n={len(group_A)})", line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=list(kmf.confidence_interval_.index) + list(kmf.confidence_interval_.index[::-1]), y=list(kmf.confidence_interval_.iloc[:, 0]) + list(kmf.confidence_interval_.iloc[:, 1][::-1]), fill='toself', fillcolor='rgba(0,100,255,0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
    kmf.fit(group_B['km_time'], event_observed=group_B['km_status'], label=f"Mutated (n={len(group_B)})")
    fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0], mode='lines', name=f"Mutated (n={len(group_B)})", line=dict(color='crimson', width=2)))
    fig.add_trace(go.Scatter(x=list(kmf.confidence_interval_.index) + list(kmf.confidence_interval_.index[::-1]), y=list(kmf.confidence_interval_.iloc[:, 0]) + list(kmf.confidence_interval_.iloc[:, 1][::-1]), fill='toself', fillcolor='rgba(220,20,60,0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
    fig.update_layout(title={'text': f"<b>Kaplan-Meier Plot for Variant: {variant_id}</b>", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}, xaxis_title="Time (Days)", yaxis_title="Survival Probability", legend_title="Patient Group", legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95), template="plotly_white")
    return fig

# --- NEW: Function to plot a bar chart for a SINGLE splice site model ---
def plot_splice_site_performance_bars(splice_type):
    """Creates a bar chart for a single splice site model's performance."""
    labels = ["Accuracy", "Precision", "Recall", "F1 Score", "MCC"]
    
    # Data for both models
    performance_data = {
        "acceptor": [93.16, 93.14, 93.39, 93.26, 86.39],
        "donor": [94.71, 94.54, 94.79, 94.66, 89.49]
    }
    
    # Select the data based on the user's choice (e.g., 'acceptor' or 'donor')
    model_name = splice_type.capitalize()
    values = performance_data.get(splice_type.lower(), [0]*len(labels)) # Default to zeros if key is invalid
    colors = px.colors.qualitative.Set2[: len(labels)]
    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition= "outside"
    ))
    # 3) Compute dynamic y-axis range with 10% padding
    if values:
        min_val, max_val = min(values), max(values)
        span = max_val - min_val
        pad = span * 0.2 if span > 0 else max_val * 0.2
        lo = max(0, min_val - pad)
        hi = min(100, max_val + pad)
    else:
        lo, hi = 0, 100
    
    fig.update_layout(
        font=dict(size=20),
        yaxis_title="Score(%)",
        xaxis_title="Metric",
        yaxis_range=[lo, hi], # Zoom in on the high performance
        height=400
    )
    return fig

def plot_tfbs_performance_bars(model_metrics):
    """
    Plots a pastel Set2 bar chart of TFBS model performance on a 0–100 scale.
    
    model_metrics: dict with keys 'Accuracy', 'Precision', 'Recall',
                   'F1-score', 'MCC' (values between 0 and 100).
    """
    # 1) Prepare labels + raw values
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'MCC', "ROC-AUC"]
    values = [float(model_metrics.get(m, 0)) for m in labels]
    
    # 2) Build the bar chart
    colors = px.colors.qualitative.Set2[: len(labels)]
    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        showlegend=False
    ))
    # 3) Compute dynamic y-axis range with 10% padding
    if values:
        min_val, max_val = min(values), max(values)
        span = max_val - min_val
        pad = span * 0.1 if span > 0 else max_val * 0.1
        lo = max(0, min_val - pad)
        hi = min(100, max_val + pad)
    else:
        lo, hi = 0, 100
    
    # 3) Layout tweaks: fixed y-axis from 0 to 100
    fig.update_layout(
        font=dict(size=20)
    )
    fig.update_xaxes(title_text="Metric")
    fig.update_yaxes(title_text="Score(%)", range=[lo, hi])
    
    return fig


def colorize_motif(motif):
    """
    Takes a DNA sequence string and returns an HTML string with each
    nucleotide colored according to standard conventions.
    """
    color_map = {
        'A': '#10A546', # Green
        'C': '#0C63D9', # Blue
        'G': '#F57D0B', # Orange
        'T': '#D62323', # Red
        'N': '#BDBDBD'  # Grey for unknown
    }
    
    html_parts = []
    for char in motif.upper():
        color = color_map.get(char, '#000000') # Default to black
        html_parts.append(
            f'<span style="font-size: 20px; font-family: monospace; font-weight: bold; color: {color};">{char}</span>'
        )
    return "".join(html_parts)



# --- Main Page Logic ---
st.title("📊 Browse and Analyze Variants")
st.markdown("Use the sidebar to select the analysis type, then use the controls on this page to filter and explore the data.")

df_variants, df_clinical, df_tfbs_summary = load_data(cancer_type, analysis_type, data_source)

if df_variants is None:
    st.stop()

# --- UI for Splice Sites ---
# --- Conditional UI ---
if analysis_type == "Splice Sites":
    st.header("Splice Site Variant Analysis")
    

    st.markdown("#### Filter by Splice Site Type:")
    splice_type = st.selectbox("", df_variants['splice_sites_affected'].unique(), label_visibility="collapsed")
    df_filtered = df_variants[df_variants['splice_sites_affected'] == splice_type]

    # --- UPDATED: Splice Site Dashboard ---
    st.subheader(f"Dashboard for: {splice_type.capitalize()} Splice Model")
    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.markdown("**Model Performance**")
        bar_fig = plot_splice_site_performance_bars(splice_type)
        st.plotly_chart(bar_fig, use_container_width=True)
    with col2:
        st.markdown("**Variant Summary**")
        st.metric("Total Candidate Variants", f"{len(df_filtered)}")
        dbsnp_count = df_filtered['rsID'].nunique() if 'rsID' in df_filtered.columns else 0
        st.metric("Associated dbSNP IDs", f"{dbsnp_count}")
        survival_count = df_filtered[df_filtered['p_value'] < 0.05].shape[0] if 'p_value' in df_filtered.columns else 0
        st.metric("Survival-Associated", f"{survival_count}")
    st.divider()
    st.markdown(f"Displaying **{len(df_filtered)}** variants for the {splice_type} splice sites.")
    
# --- UI for TFBS Models ---
elif analysis_type == "TFBS Models":
    st.header("TFBS Variant Analysis")
    if df_tfbs_summary is None:
        st.error("Could not load TFBS summary data. Please ensure `data/TFBS_model_summary_final.tsv` exists.")
        st.stop()


    st.markdown("#### Step 1: Select a TFBS Model to Analyze")
    tfbs_model = st.selectbox("", sorted(df_variants['TFBS'].unique()))
    df_filtered = df_variants[df_variants['TFBS'] == tfbs_model]
    #st.dataframe(df_variants)
    #st.dataframe(df_filtered)
    model_summary = df_tfbs_summary[df_tfbs_summary['TFBS'] == tfbs_model].iloc[0]
    
    st.subheader(f"Dashboard for: {tfbs_model}")
    col1, col2, col3 = st.columns([2, 1, 2], gap="large")

    with col1:
        st.markdown("**Model Performance**")
        barplot_fig = plot_tfbs_performance_bars(model_summary)
        st.plotly_chart(barplot_fig, use_container_width=True)

    with col2:
        st.markdown("**Variant Summary**")
        st.metric("Candidate Variants", f"{len(df_filtered)}")
        dbsnp_count = df_filtered['rsID'].nunique()
        st.metric("Associated dbSNP IDs", f"{dbsnp_count}")
        survival_count = df_filtered[df_filtered['p_value'] < 0.05].shape[0] if 'p_value' in df_filtered else 0
        st.metric("Survival-Associated", f"{survival_count}")

    with col3:
        st.markdown("**Motif Validation (vs. JASPAR)**")
        st.metric("Identical Motif Matches", f"{int(model_summary.get('identical_match_count', 0))}")
        st.metric("Best Overall Match ID", model_summary.get("BestMatch_JASPAR_ID", "N/A"), f"q-value: {model_summary.get('BestMatch_q_value', 0):.2e}")
        st.metric("Best Identical Match ID", model_summary.get("IdenticalMatch_JASPAR_ID", "N/A"), f"q-value: {model_summary.get('IdenticalMatch_q_value', 0):.2e}")
    st.divider()

    st.markdown(f"Displaying **{len(df_filtered)}** variants for the {tfbs_model}.")
# ==============================================================================
# --- Common UI: AG-Grid and Survival Plot ---
# ==============================================================================
st.subheader("Interactive Variant Table")
st.markdown("#### Step 2: Explore Variants in the Table")

# Configure AG-Grid
gb = GridOptionsBuilder.from_dataframe(df_filtered)
gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
gb.configure_selection('single', use_checkbox=True, suppressRowDeselection=False)
gb.configure_side_bar()
grid_options = gb.build()

# Hide non-essential columns
columns_to_hide = ['GBM_patient_ids', 'chromosome', 'CHROM', 'TFBS', 'TFBS_category', 'splice_sites_affected', 'variant_start_position', 'variant_end_position', 'ref_nucleotide', 'alternative_nucleotide', 'composite_key', 'diff_subsequence', '3bp_flanking_diff', 'variant_diff', 'S3_path', 'JASPAR_IDs']
column_defs = grid_options['columnDefs']
grid_options['columnDefs'] = [col for col in column_defs if col['field'] not in columns_to_hide]

grid_response = AgGrid(
    df_filtered, gridOptions=grid_options, update_mode=GridUpdateMode.SELECTION_CHANGED,
    height=600, width='100%', allow_unsafe_jscode=True,
)

selected_rows_df = pd.DataFrame(grid_response['selected_rows'])



if not selected_rows_df.empty:
    st.divider()
    st.markdown(
        "<h2 style='text-align: center;'>Detailed Analysis for Selected Variant</h2>",
        unsafe_allow_html=True
    )
    
    # Get all the information for the selected variant from the first row of the DataFrame
    selected_variant_info = selected_rows_df.iloc[0]
    variant_id = selected_variant_info.get('variant_id', 'N/A')
    patient_ids_str = selected_variant_info.get('GBM_patient_ids', '')

    if analysis_type == "TFBS Models":
        # --- Create the 1x2 Grid Layout ---
        col1, col2 = st.columns(2, gap="large")
        st.markdown("<hr>", unsafe_allow_html=True) # Visual separator

        # --- Quadrant 1 (Top-Left): Clinical Impact ---
        with col1:
            st.subheader("Model Prediction & Disruption")
            
            st.markdown(
                "<h5 style='text-align: center;'>Prediction Scores</h5>",
                unsafe_allow_html=True
            )
            ref_prob = selected_variant_info.get('Ref_probab', 0)
            alt_prob = selected_variant_info.get('Alt_probab', 0)
            disruption_score = selected_variant_info.get('Loss of Function based on LogOddRatio', 0)
            score_change = selected_variant_info.get('ScoreChange', 0)
            
            score_col1, score_col2 = st.columns(2)
            score_col1.metric("Wild Type Probability", f"{ref_prob:.2f}")
            score_col2.metric(
                "Mutated Probability", 
                f"{alt_prob:.2f}",
                delta=f"{(alt_prob - ref_prob):.2f}",
                delta_color="inverse"
            )
            disruption_col1 , disruption_col2 = st.columns(2)
            disruption_col1.metric("Disruption Score (LogOddRatio)", f"{disruption_score:.2f}")
            disruption_col2.metric("Disruption Score (Score Change)", f"{score_change:.2f}")
            
            st.divider()
            st.markdown(
                "<h5 style='text-align: center;'>Variant association with attention score and known MOTIFs</h5>",
                unsafe_allow_html=True
            )
            relative_path = selected_variant_info.get('S3_path') # Use the column name 'S3_path'
            if relative_path and isinstance(relative_path, str):
                # Construct the full URL
                full_image_url = S3_BASE_URL + relative_path
                # Or, if you stored a relative path:
                # full_image_url = S3_BASE_URL + heatmap_path
                
                st.image(full_image_url, use_column_width=True)
                st.caption("Attention scores for the Wild Type (top) and Mutated (bottom) sequences, shown with a +/- 10bp buffer around the variant.")
            else:
                st.info("No attention heatmap available for this variant.")

            st.divider()
            # --- Associated Motif Information ---
            st.markdown(
                "<h5 style='text-align: center;'>Disrupted Motif(s) & JASPAR Validation</h5>",
                unsafe_allow_html=True
            )

            associated_motifs = selected_variant_info.get('Associated_motifs')
            jaspar_ids = selected_variant_info.get('JASPAR_IDs')

            if pd.notna(associated_motifs) and pd.notna(jaspar_ids):
                # Prepare the lists of motifs and IDs
                motif_list = [m.strip() for m in str(associated_motifs).split(',')]
                jaspar_id_list = [j.strip() for j in str(jaspar_ids).split(',')]
                
                # # --- Create the styled HTML strings ---

                # # For the motifs, now using the colorize_motif function
                # motif_html_parts = []
                # for motif in motif_list:
                #     motif_html_parts.append(colorize_motif(motif))
                # motifs_display_html = ", ".join(motif_html_parts)

                # # For the JASPAR links
                # link_html_parts = []
                # for j_id in jaspar_id_list:
                #     jaspar_url = f"https://jaspar.genereg.net/matrix/{j_id}/"
                #     link_html_parts.append(f'<a href="{jaspar_url}" target="_blank" style="font-size: 20px;">{j_id}</a>')
                # links_display_html = ", ".join(link_html_parts)

                # # --- Display the label and the value on the same line ---
                # st.markdown(f"**Predicted Motif(s):** {motifs_display_html}", unsafe_allow_html=True)
                # st.markdown(f"**Best JASPAR Match(es):** {links_display_html}", unsafe_allow_html=True)
                        
                # st.caption("Motif location is highlighted in the heatmap with a dotted line.")
                # # Create one column per motif
                # cols = st.columns(len(motif_list), gap="large")
                # for idx, (motif, j_id) in enumerate(zip(motif_list, jaspar_id_list)):
                #     with cols[idx]:
                #         # 1) Colorized motif centered
                #         colored_html = colorize_motif(motif)
                #         st.markdown(
                #             f"<div style='text-align:center; margin-bottom:8px;'>{colored_html}</div>",
                #             unsafe_allow_html=True,
                #         )

                #         # 2) Clickable JASPAR button
                #         jaspar_url = f"https://jaspar.genereg.net/matrix/{j_id}/"
                #         if hasattr(st, "link_button"):
                #             st.link_button(label=j_id, url=jaspar_url)
                #         else:
                #             st.markdown(
                #                 f"""
                #                 <div style='text-align:center;'>
                #                   <a
                #                     href="{jaspar_url}"
                #                     target="_blank"
                #                     style="
                #                       display:inline-block;
                #                       background-color:#0072B2;
                #                       color:white;
                #                       padding:6px 12px;
                #                       border-radius:4px;
                #                       text-decoration:none;
                #                     "
                #                   >{j_id}</a>
                #                 </div>
                #                 """,
                #                 unsafe_allow_html=True,
                #          )
                # Build your lists
                # motif_list     = [m.strip() for m in str(associated_motifs).split(",")]
                # jaspar_id_list = [j.strip() for j in str(jaspar_ids).split(",")]

                # Two columns: left for motifs, right for JASPAR links
                # Build your data
                    
                #jaspar_id_list = ["MA0113.4", "MA1930.2"]  # corresponding JASPAR IDs

                # Helper to colorize each motif
                def motif_cell(motif):
                    return f"<span style='font-family:monospace; font-size:18px;'>{colorize_motif(motif)}</span>"

                # Helper to render each JASPAR button
                def button_cell(j_id):
                    url = f"https://jaspar.genereg.net/matrix/{j_id}/"
                    return (
                        f"<a href='{url}' target='_blank' "
                        "style='display:inline-block; padding:6px 12px; "
                        "background-color:#0072B2; color:white; border-radius:4px;"
                        "text-decoration:none; font-size:16px;'>"
                        f"{j_id}</a>"
                    )

                # Build table rows
                # Header row label + one <td> per motif
                motifs_html = "".join(f"<td style='padding:4px 12px;'>{motif_cell(m)}</td>" 
                                      for m in motif_list)
                buttons_html = "".join(f"<td style='padding:4px 12px;'>{button_cell(j)}</td>" 
                                       for j in jaspar_id_list)

                table_html = f"""
                <div style="width:100%; display:flex; justify-content:center; margin:20px 0;">
                  <table style="border-collapse:collapse; text-align:center; table-layout:auto;">
                    <tr>
                      <th style="padding:8px 16px; text-align:left;">Predicted Motif(s)</th>
                      {motifs_html}
                    </tr>
                    <tr>
                      <th style="padding:8px 16px; text-align:left;">Best JASPAR Match(es)</th>
                      {buttons_html}
                    </tr>
                  </table>
                </div>
                """

                st.markdown(table_html, unsafe_allow_html=True)


                st.markdown(
                    "<h6 style='text-align: center;'>Motif positions are highlighted in the heatmap with a dotted line.</h6>",
                    unsafe_allow_html=True
                )
            else:
                st.info("No known JASPAR motif was found to be directly disrupted by this variant.")



        # --- Quadrant 2 (Top-Right): Visual Explanation ---
        with col2:
            st.subheader("Clinical Significance & Annotation")
            
            rsID = selected_variant_info.get('rsID')
            if pd.notna(rsID):
                ncbi_url = f"https://www.ncbi.nlm.nih.gov/snp/{rsID}"
                st.markdown(f"**dbSNP ID:** [{rsID}]({ncbi_url})")
                st.markdown(f"**Clinical Significance:** [View on NCBI]({ncbi_url}#clinical_significance)")
            else:
                st.markdown("**dbSNP ID:** Not Available")

            patient_ids_str = selected_variant_info.get('GBM_patient_ids', '')
            p_value = selected_variant_info.get('p_value')
            hr_value = selected_variant_info.get('HR')

            # Check if survival data is available
            if pd.notna(p_value) and patient_ids_str:
                # Display the key statistics below the plot
                stat_col1, stat_col2 = st.columns(2)
                stat_col1.metric("Hazard Ratio (HR)", f"{hr_value:.2f}")
                stat_col2.metric("Log-Rank p-value", f"{p_value:.2e}")


                mutated_patient_ids = [pid.strip().split('_')[0] for pid in patient_ids_str.split(',')]
                df_clinical['group'] = df_clinical['manifest_patient_id'].apply(lambda x: 'Mutated' if x in mutated_patient_ids else 'Wild-Type')
                group_A = df_clinical[df_clinical['group'] == 'Wild-Type']
                group_B = df_clinical[df_clinical['group'] == 'Mutated']
                
                km_fig = plot_km_curve(group_A, group_B, variant_id, p_value)
                st.plotly_chart(km_fig, use_container_width=True)

                
            else:
                st.info("Survival analysis data is not available for this variant.")


    elif analysis_type == "Splice Sites":
            col_annot, col_surv  = st.columns([1, 2], gap="large")
            with col_annot:
                st.subheader("Variant Annotation")
                rsID = selected_variant_info.get('rsID')
                if pd.notna(rsID):
                    ncbi_url = f"https://www.ncbi.nlm.nih.gov/snp/{rsID}"
                    st.metric("dbSNP ID", rsID)
                    if hasattr(st, "link_button"):
                            st.link_button("Open in dbSNP", ncbi_url)
                            st.link_button("Clinical Significance", f"{ncbi_url}#clinical_significance")
                    else:
                        st.markdown(f"[Open in dbSNP]({ncbi_url})")
                        st.markdown(f"**Clinical Significance:** [View on NCBI]({ncbi_url}#clinical_significance)")
                else:
                    st.markdown("**dbSNP ID:** Not Available")
            with col_surv:
                st.subheader("Clinical Significance & Annotation")
                patient_ids_str = selected_variant_info.get('GBM_patient_ids', '')
                p_value = selected_variant_info.get('p_value')
                hr_value = selected_variant_info.get('HR')

                # Check if survival data is available
                if pd.notna(p_value) and patient_ids_str:
                    # Display the key statistics below the plot
                    stat_col1, stat_col2 = st.columns(2)
                    stat_col1.metric("Hazard Ratio (HR)", f"{hr_value:.2f}")
                    stat_col2.metric("Log-Rank p-value", f"{p_value:.2e}")


                    mutated_patient_ids = [pid.strip().split('_')[0] for pid in patient_ids_str.split(',')]
                    df_clinical['group'] = df_clinical['manifest_patient_id'].apply(lambda x: 'Mutated' if x in mutated_patient_ids else 'Wild-Type')
                    group_A = df_clinical[df_clinical['group'] == 'Wild-Type']
                    group_B = df_clinical[df_clinical['group'] == 'Mutated']
                    
                    km_fig = plot_km_curve(group_A, group_B, variant_id, p_value)
                    st.plotly_chart(km_fig, use_container_width=True)

                else:
                    st.info("Survival analysis data is not available for this variant.")