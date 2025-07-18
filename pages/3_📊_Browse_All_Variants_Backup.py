import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Browse Variants")

st.title("📊 Browse All Variants")
st.markdown("Use the sidebar to select the dataset. Then, select a single variant from the table below to generate its Kaplan-Meier survival plot.")
st.info("ℹ️ The `GBM_patient_ids` and original variant coordinate columns are used for analysis but are hidden from the table for clarity and data privacy.")
st.divider()

# --- Sidebar for selections ---
st.sidebar.header("Data Selection")
cancer_type = st.sidebar.selectbox("Select Cancer Type", ["Brain"], key="cancer_type_browser")
analysis_type = st.sidebar.selectbox("Genomic Regulatory Elements", ["Splice Sites", "TFBS Models"], key="analysis_type_browser")

analysis_options = {
    "Substitutions (SNVs)": "CaVEMan",
    "Insertions & Deletions (Indels)": "sanger_raw_pindel"
}
selected_analysis_label = st.sidebar.selectbox("Select Variant Type", options=list(analysis_options.keys()), key="variant_type_browser")
data_source = analysis_options[selected_analysis_label]

# --- Data Loading Function ---
@st.cache_data
def load_browser_data(cancer, analysis, source):
    """Loads and preprocesses the necessary data files for the browser page."""
    base_path = f"data/{cancer}/"
    analysis_folder_map = {"Splice Sites": "Splice_Sites", "TFBS Models": "TFBS_Models"}
    analysis_folder = analysis_folder_map[analysis]
    
    tsv_path = f"{base_path}{analysis_folder}/"
    
    try:
        df_transcript_info = pd.read_csv(f"{tsv_path}{source}_combined_{analysis_folder}_Variants_frequency_with_gene_information.tsv", sep="\t")
        df_clinical = pd.read_csv(f"{base_path}patient_clinical_updated.tsv", sep='\t')
        
        df_clinical = df_clinical.dropna(subset=['manifest_patient_id', 'km_time', 'km_status'])
        
        # --- NEW: Pre-process the dataframe here, once ---
        # Create the variant_information column
        df_transcript_info['variant_information'] = df_transcript_info.apply(
            lambda row: f"{row['chromosome']}:{row['variant_start_position']}:{row['ref_nucleotide']}>{row['alternative_nucleotide']}",
            axis=1
        )
        # Reorder columns to make 'variant_information' first
        all_cols = df_transcript_info.columns.tolist()
        all_cols.insert(0, all_cols.pop(all_cols.index('variant_information')))
        df_transcript_info = df_transcript_info[all_cols]
        
        return df_transcript_info, df_clinical
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found. Please ensure your repository is structured correctly. Details: {e}")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None, None

# --- Plotting Function ---
def plot_km_curve(group_A, group_B, variant_id, p_value):
    """Generates an interactive Kaplan-Meier plot using Plotly."""
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

    # Update layout
    fig.update_layout(title={'text': f"<b>Survival Analysis for Variant: {variant_id}</b><br>Log-Rank Test p-value: {p_value:.4f}", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}, xaxis_title="Time (Days)", yaxis_title="Survival Probability", legend_title="Patient Group", legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95), template="plotly_white")
    return fig

# --- Main Application Logic ---
df_info, df_clinical = load_browser_data(cancer_type, analysis_type, data_source)

if df_info is not None and df_clinical is not None:
    st.header("Variant Data Table")
    
    # Configure the AG-Grid
    gb = GridOptionsBuilder.from_dataframe(df_info)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
    gb.configure_selection('single', use_checkbox=True, suppressRowDeselection=False)
    gb.configure_side_bar()
    grid_options = gb.build()

    # --- NEW: Define all columns to hide from the display ---
    columns_to_hide = [
        'GBM_patient_ids', 
        'chromosome', 
        'variant_start_position', 
        'variant_end_position', 
        'ref_nucleotide', 
        'alternative_nucleotide'
    ]
    column_defs = grid_options['columnDefs']
    grid_options['columnDefs'] = [col for col in column_defs if col['field'] not in columns_to_hide]

    grid_response = AgGrid(
        df_info,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        enable_enterprise_modules=False,
        height=600,
        width='100%',
        reload_data=True,
        allow_unsafe_jscode=True,
    )
    
    selected_rows_df = pd.DataFrame(grid_response['selected_rows'])

    # --- Survival Plot Generation ---
    if not selected_rows_df.empty:
        st.divider()
        st.subheader("Kaplan-Meier Survival Plot")
        
        selected_variant_info = selected_rows_df.iloc[0]
        variant_id = selected_variant_info.get('variant_information', 'Unknown Variant')
        patient_ids_str = selected_variant_info.get('GBM_patient_ids', '')
        
        if not patient_ids_str:
            st.warning("The selected variant does not have associated patient IDs.")
        else:
            mutated_patient_ids = [pid.strip().split('_')[0] for pid in patient_ids_str.split(',')]
            df_clinical['group'] = df_clinical['manifest_patient_id'].apply(
                lambda x: 'Mutated' if x in mutated_patient_ids else 'Wild-Type'
            )
            
            group_A = df_clinical[df_clinical['group'] == 'Wild-Type']
            group_B = df_clinical[df_clinical['group'] == 'Mutated']
            
            if not group_A.empty and not group_B.empty:
                results = logrank_test(
                    group_A['km_time'], group_B['km_time'], 
                    event_observed_A=group_A['km_status'], event_observed_B=group_B['km_status']
                )
                p_value = results.p_value

                km_fig = plot_km_curve(group_A, group_B, variant_id, p_value)
                st.plotly_chart(km_fig, use_container_width=True)
            else:
                st.warning("Cannot perform survival analysis. One or both patient groups (Mutated/Wild-Type) are empty for this variant.")
else:
    st.info("Waiting for data to load or data loading failed. Please check the sidebar selections and file paths.")

