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

# --- Sidebar Controls ---
st.sidebar.header("Global Controls")
cancer_type = st.sidebar.selectbox("Select a Cancer Type:", ("Brain", "Breast", "Lung"), key="cancer_type_browser")
analysis_type = st.sidebar.selectbox("Genomic Regulatory Elements", ["Splice Sites", "TFBS Models"], key="analysis_type_browser")
variant_options = { "Substitutions (SNVs)": "CaVEMan", "Insertions & Deletions (Indels)": "sanger_raw_pindel" }
selected_variant_label = st.sidebar.selectbox("Select Variant Type", options=list(variant_options.keys()), key="variant_type_browser")
data_source = variant_options[selected_variant_label]

# --- Data Loading Functions ---
@st.cache_data
def load_data(cancer, analysis, source):
    """Loads all necessary data files based on user selections."""
    base_path = f"data/{cancer}/"
    analysis_folder_map = {"Splice Sites": "Splice_Sites", "TFBS Models": "TFBS_Models"}
    analysis_folder = analysis_folder_map[analysis]
    tsv_path = f"{base_path}{analysis_folder}/"
    
    try:
        df_variants = pd.read_csv(f"{tsv_path}{source}_combined_{analysis_folder}_Variants_frequency_with_gene_information.tsv", sep="\t")
        df_clinical = pd.read_csv(f"{base_path}patient_clinical_updated.tsv", sep='\t')
        df_dbsnp = pd.read_csv(f"{tsv_path}{source}_combined_{analysis_folder}_Intersect_withDBSNP.tsv", sep='\t')
        df_clinical = df_clinical.dropna(subset=['manifest_patient_id', 'km_time', 'km_status'])
        
        # Pre-process variants data
        df_variants['variant_information'] = df_variants.apply(
            lambda row: f"{row['chromosome']}:{row['variant_start_position']}:{row['ref_nucleotide']}>{row['alternative_nucleotide']}",
            axis=1
        )
        all_cols = df_variants.columns.tolist()
        all_cols.insert(0, all_cols.pop(all_cols.index('variant_information')))
        df_variants = df_variants[all_cols]
        
        df_tfbs_summary = None
        if analysis == "TFBS Models":
            try:
                df_tfbs_summary = pd.read_csv("data/df_summary.tsv", sep="\t")
            except FileNotFoundError:
                st.warning("`TFBS_model_summary.tsv` not found. TFBS dashboard will not be available.")

        return df_variants, df_clinical, df_tfbs_summary, df_dbsnp
        
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found. Please check your repository. Details: {e}")
        return None, None, None, None

# --- Plotting Functions ---
def plot_km_curve(group_A, group_B, variant_id, p_value):
    # (Same as previous version)
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    # Group A (Wild-Type)
    kmf.fit(group_A['km_time'], event_observed=group_A['km_status'], label=f"Wild-Type (n={len(group_A)})")
    fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0], mode='lines', name=f"Wild-Type (n={len(group_A)})", line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=list(kmf.confidence_interval_.index) + list(kmf.confidence_interval_.index[::-1]), y=list(kmf.confidence_interval_.iloc[:, 0]) + list(kmf.confidence_interval_.iloc[:, 1][::-1]), fill='toself', fillcolor='rgba(0,100,255,0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
    # Group B (Mutated)
    kmf.fit(group_B['km_time'], event_observed=group_B['km_status'], label=f"Mutated (n={len(group_B)})")
    fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0], mode='lines', name=f"Mutated (n={len(group_B)})", line=dict(color='crimson', width=2)))
    fig.add_trace(go.Scatter(x=list(kmf.confidence_interval_.index) + list(kmf.confidence_interval_.index[::-1]), y=list(kmf.confidence_interval_.iloc[:, 0]) + list(kmf.confidence_interval_.iloc[:, 1][::-1]), fill='toself', fillcolor='rgba(220,20,60,0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
    fig.update_layout(title={'text': f"<b>Survival Analysis for Variant: {variant_id}</b><br>Log-Rank Test p-value: {p_value:.4f}", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}, xaxis_title="Time (Days)", yaxis_title="Survival Probability", legend_title="Patient Group", legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95), template="plotly_white")
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

df_variants, df_clinical, df_tfbs_summary, df_dbsnp = load_data(cancer_type, analysis_type, data_source)

if df_variants is None:
    st.stop()

# ==============================================================================
# --- UI for Splice Sites ---
# ==============================================================================
if analysis_type == "Splice Sites":
    st.header("Splice Site Variant Analysis")
    
    # --- Filter Controls ---
    splice_type = st.selectbox("Filter by Splice Site Type:", df_variants['splice_sites_affected'].unique())
    df_filtered = df_variants[df_variants['splice_sites_affected'] == splice_type]
    
    st.markdown(f"Displaying **{len(df_filtered)}** variants for **{splice_type}** sites.")
    
# ==============================================================================
# --- UI for TFBS Models ---
# ==============================================================================
elif analysis_type == "TFBS Models":
    st.header("TFBS Variant Analysis")

    # --- Filter Controls ---
    st.markdown("#### Step 1: Select a TFBS Model to Analyze")
    tfbs_model = st.selectbox("", sorted(df_variants['TFBS'].unique()))
    df_filtered = df_variants[df_variants['TFBS'] == tfbs_model]
    df_dbsnp_filtered = df_dbsnp[df_dbsnp['TFBS'] == tfbs_model]

    # --- TFBS Dashboard Section ---
    if df_tfbs_summary is not None:
        model_summary = df_tfbs_summary[df_tfbs_summary['TFBS'] == tfbs_model].iloc[0]
        
        st.subheader(f"Dashboard for: {tfbs_model}")
        col1, col2, col3 = st.columns([2, 1, 1])
        # first column: overall counts
        with col1:
            st.markdown("**Model Performance**")
            barplot_fig = plot_tfbs_performance_bars(model_summary)
            st.plotly_chart(barplot_fig, use_container_width=True)

        # second column: JASPAR info
        with col2:
            st.markdown("**Variant Summary**")
            total = len(df_filtered)
            st.metric("Candidate Variants (>10% Patients)", f"{total}")
            unique_rsids = df_dbsnp_filtered['rsID'].nunique()
            st.metric("Associated Unique dbSNP IDs(rsIDs)", f"{unique_rsids}")
            
        # third column: performance bar plot
        with col3:
            st.markdown("**Motif Matches in JASPAR**")
            st.metric("Identical Matches", f"{int(model_summary.get('identical_match_count', 0))}")
            # identical match ID + its q-value
            st.metric("Best Overall Match ID", model_summary.get("BestMatch_JASPAR_ID", "N/A"), f"q-value: {model_summary.get('BestMatch_q_value', 0):.2e}")
            st.metric("Best Identical Match ID", model_summary.get("IdenticalMatch_JASPAR_ID", "N/A"), f"q-value: {model_summary.get('IdenticalMatch_q_value', 0):.2e}")
        
        

    st.markdown(f"Displaying **{len(df_filtered)}** variants for the **{tfbs_model}** model.")

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

# --- Survival Plot and Motif Analysis Section ---
if not selected_rows_df.empty:
    st.divider()
    st.header("Step 3: Detailed Analysis for Selected Variant")
    
    selected_variant_info = selected_rows_df.iloc[0]
    variant_id = selected_variant_info.get('variant_information', 'Unknown Variant')
    patient_ids_str = selected_variant_info.get('GBM_patient_ids', '')

    col_km, col_motif = st.columns(2, gap="large")

    with col_km:
        st.subheader("Kaplan-Meier Survival Plot")
        if not patient_ids_str:
            st.warning("No patient IDs found for this variant.")
        else:
            mutated_patient_ids = [pid.strip().split('_')[0] for pid in patient_ids_str.split(',')]
            df_clinical['group'] = df_clinical['manifest_patient_id'].apply(lambda x: 'Mutated' if x in mutated_patient_ids else 'Wild-Type')
            
            group_A = df_clinical[df_clinical['group'] == 'Wild-Type']
            group_B = df_clinical[df_clinical['group'] == 'Mutated']
            
            if not group_A.empty and not group_B.empty:
                results = logrank_test(group_A['km_time'], group_B['km_time'], event_observed_A=group_A['km_status'], event_observed_B=group_B['km_status'])
                km_fig = plot_km_curve(group_A, group_B, variant_id, results.p_value)
                st.plotly_chart(km_fig, use_container_width=True)
            else:
                st.warning("Cannot perform survival analysis. One or both patient groups are empty.")

    with col_motif:
        if analysis_type == "TFBS Models":
            st.subheader("Motif Disruption Analysis")
            
            # Display dbSNP ID if available
            dbsnp_id = selected_variant_info.get('rsID', 'Not Available')
            st.metric("dbSNP ID", dbsnp_id)
            
            # Display pre-generated attention heatmap
            st.markdown("##### Attention Heatmap")
            heatmap_path = selected_variant_info.get('Attention_Heatmap_Path') # The new column from your data
            if heatmap_path and isinstance(heatmap_path, str):
                try:
                    image = Image.open(heatmap_path)
                    st.image(image, caption="Attention scores over reference and alternative sequences.", use_column_width=True)
                except FileNotFoundError:
                    st.error(f"Attention map image not found at path: {heatmap_path}")
            else:
                st.info("No attention heatmap available for this variant.")

