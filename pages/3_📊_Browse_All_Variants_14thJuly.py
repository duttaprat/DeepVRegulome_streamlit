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
            df_tfbs_summary = pd.read_csv("data/300bp_TFBS_accuracy_Stat.csv", sep=",")
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

# --- Main Page Logic ---
st.title("📊 Browse and Analyze Variants")
st.markdown("Use the sidebar to select the analysis type, then use the controls on this page to filter and explore the data.")

df_variants, df_clinical, df_tfbs_summary = load_data(cancer_type, analysis_type, data_source)

if df_variants is None:
    st.stop()

# --- UI for Splice Sites ---
if analysis_type == "Splice Sites":
    st.header("Splice Site Variant Analysis")
    splice_type = st.selectbox("Filter by Splice Site Type:", df_variants['splice_sites_affected'].unique())
    df_filtered = df_variants[df_variants['splice_sites_affected'] == splice_type]
    
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

st.markdown(f"Displaying **{len(df_filtered)}** variants for the selected filter.")
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
columns_to_hide = ['GBM_patient_ids', 'chromosome', 'variant_start_position', 'variant_end_position', 'ref_nucleotide', 'alternative_nucleotide']
column_defs = grid_options['columnDefs']
grid_options['columnDefs'] = [col for col in column_defs if col['field'] not in columns_to_hide]

grid_response = AgGrid(
    df_filtered, gridOptions=grid_options, update_mode=GridUpdateMode.SELECTION_CHANGED,
    height=600, width='100%', allow_unsafe_jscode=True,
)

selected_rows_df = pd.DataFrame(grid_response['selected_rows'])



if not selected_rows_df.empty:
    st.divider()
    st.header("Detailed Analysis for Selected Variant")
    
    # Get all the information for the selected variant from the first row of the DataFrame
    selected_variant_info = selected_rows_df.iloc[0]
    variant_id = selected_variant_info.get('variant_id', 'N/A')
    patient_ids_str = selected_variant_info.get('GBM_patient_ids', '')


    # --- Create the 2x2 Grid Layout ---
    row1_col1, row1_col2 = st.columns(2, gap="large")
    st.markdown("<hr>", unsafe_allow_html=True) # Visual separator
    row2_col1, row2_col2 = st.columns(2, gap="large")

    # --- Quadrant 1 (Top-Left): Clinical Impact ---
    with row1_col1:
        st.subheader("Survival Analysis")
        
        patient_ids_str = selected_variant_info.get('GBM_patient_ids', '')
        p_value = selected_variant_info.get('p_value')
        hr_value = selected_variant_info.get('HR')

        # Check if survival data is available
        if pd.notna(p_value) and patient_ids_str:
            mutated_patient_ids = [pid.strip().split('_')[0] for pid in patient_ids_str.split(',')]
            df_clinical['group'] = df_clinical['manifest_patient_id'].apply(lambda x: 'Mutated' if x in mutated_patient_ids else 'Wild-Type')
            group_A = df_clinical[df_clinical['group'] == 'Wild-Type']
            group_B = df_clinical[df_clinical['group'] == 'Mutated']
            
            km_fig = plot_km_curve(group_A, group_B, variant_id, p_value)
            st.plotly_chart(km_fig, use_container_width=True)

            # Display the key statistics below the plot
            stat_col1, stat_col2 = st.columns(2)
            stat_col1.metric("Hazard Ratio (HR)", f"{hr_value:.2f}")
            stat_col2.metric("Log-Rank p-value", f"{p_value:.2e}")
        else:
            st.info("Survival analysis data is not available for this variant.")

    # --- Quadrant 2 (Top-Right): Visual Explanation ---
    with row1_col2:
        st.subheader("Attention Heatmap and Motif analysis")
        
        
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

    # --- Quadrant 3 (Bottom-Left): Quantitative Prediction ---
    with row2_col1:
        st.subheader("Model Prediction Scores")
        
        st.code(variant_id) # Display the variant ID clearly
        
        ref_prob = selected_variant_info.get('Ref_probab', 0)
        alt_prob = selected_variant_info.get('Alt_probab', 0)
        disruption_score = selected_variant_info.get('Loss of Function based on LogOddRatio', 0)
        
        # Use columns for a cleaner layout of metrics
        score_col1, score_col2 = st.columns(2)
        score_col1.metric("Wild Type Probability", f"{ref_prob:.4f}")
        score_col2.metric(
            "Mutated Probability", 
            f"{alt_prob:.4f}",
            delta=f"{(alt_prob - ref_prob):.4f}",
            delta_color="inverse"
        )
        st.metric("Disruption Score (LogOddRatio)", f"{disruption_score:.4f}")


    # --- Quadrant 4 (Bottom-Right): Biological Context ---
    with row2_col2:
        st.subheader("Genomic & Motif Context")
        
        # dbSNP Information
        rsID = selected_variant_info.get('rsID')
        if pd.notna(rsID):
            ncbi_url = f"https://www.ncbi.nlm.nih.gov/snp/{rsID}"
            st.markdown(f"**dbSNP ID:** [{rsID}]({ncbi_url})")
            # You can add the ClinVar link directly
            st.markdown(f"**Clinical Significance:** [View on NCBI]({ncbi_url}#clinical_significance)")
        else:
            st.markdown("**dbSNP ID:** Not Available")

        # Associated Motif Information
        st.markdown("**Associated Motif:**")
        associated_motifs = selected_variant_info.get('Associated_motifs')
        if associated_motifs and isinstance(associated_motifs, str) and associated_motifs.lower() != 'nan':
            # You can create a link to JASPAR if you have a way to map motif names to JASPAR IDs
            st.code(associated_motifs)
            st.caption("This motif is highlighted in the heatmap with a dotted orange line.")
        else:
            st.info("No known JASPAR motif was found to be directly disrupted by this variant.")


if not selected_rows_df.empty:
    st.divider()
    st.header("🔍 Detailed Analysis for Selected Variant")

    sv = selected_rows_df.iloc[0]
    variant_id = sv['variant_id']
    rsid       = sv.get('rsID', None)
    pval       = sv.get('p_value', None)
    hr         = sv.get('HR', None)
    probs_ref  = sv['Ref_probab']
    probs_alt  = sv['Alt_probab']
    motifs     = sv.get('Associated_motifs', [])
    patient_ids_str = sv.get('GBM_patient_ids', '')

    # first row: 2 columns
    r1c1, r1c2 = st.columns(2, gap="large")
    # second row: 2 columns
    r2c1, r2c2 = st.columns(2, gap="large")

    # ──────────────────────────────────────────────────────────────────────
    # 1) dbSNP + ClinVar link + clinical significance
    with r1c1:
        st.subheader("dbSNP & ClinVar")
        if rsid:
            snp_url = f"https://www.ncbi.nlm.nih.gov/snp/{rsid}"
            clin_url = snp_url + "#clinical_significance"
            st.markdown(f"- **rsID:** [{rsid}]({snp_url})")
            st.markdown(f"- [View clinical significance]({clin_url})")
            # You could optionally scrape or load ClinVar via requests here
            st.markdown("> *ClinVar*: Not reported")
        else:
            st.info("No dbSNP mapping available.")

    # 2) Kaplan–Meier
    with r1c2:
        st.subheader("Survival Analysis")
        if pval is not None and hr is not None:
            # format p-value in scientific notation: e.g. 1.2e-03
            p_sci = f"{pval:.2e}"
            st.write(f"**p-value:** {p_sci}  **HR:** {hr:.2f}")
            # regenerate the plot
            mutated = [pid.split("_")[0] for pid in patient_ids_str.split(",") if pid]
            df_clinical['group'] = df_clinical['manifest_patient_id'].apply(
                lambda x: 'Mutated' if x in mutated else 'Wild-Type'
            )
            gA = df_clinical[df_clinical['group']=="Wild-Type"]
            gB = df_clinical[df_clinical['group']=="Mutated"]
            fig = plot_km_curve(gA, gB, variant_id, pval)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Survival data unavailable for this variant.")

    # ──────────────────────────────────────────────────────────────────────
    # 3) Variant & motif summary
    with r2c1:
        st.subheader("Variant & Motifs")
        # Variant string
        st.markdown(f"**Variant:** `{variant_id}`")
        # Probabilities
        st.markdown(f"- **Wild-Type Prob:** {probs_ref:.3f}  **Mutant Prob:** {probs_alt:.3f}")
        # Motif list or fallback
        if motifs:
            motif_md = []
            for m in motifs:
                # if you know a JASPAR permalink, embed it:
                jaspar_url = f"https://jaspar.genereg.net/matrix/{m}/"
                motif_md.append(f"[{m}]({jaspar_url})")
            st.markdown("- **Associated motifs:** " + ", ".join(motif_md))
            st.markdown("> *In the heatmap below,* motifs are outlined in **orange dashed**.")
        else:
            st.markdown("- **Associated motifs:** None found")

    # 4) Attention heatmap
    with r2c2:
        st.subheader("Attention Heatmap")
        img_path = sv.get('S3_path')
        if img_path:
            full_url = S3_BASE_URL + img_path
            st.image(full_url,
                     caption="Attention heatmap ±10 bp around variant",
                     use_column_width=True)
        else:
            st.info("No attention map available.")

