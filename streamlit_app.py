import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import scipy.stats as stats
from plotly.subplots import make_subplots
import math
from decimal import Decimal, ROUND_HALF_UP



st.set_page_config(
    layout="wide",
    page_title="DeepVRegulome: Predicting Variant Impact Using DNABERT-based Finetuned Models"
)
# Title of the app
st.title("DeepVRegulome: DNABERT-based deep-learning framework for predicting the functional impact of short genomic variants on the human regulome")

# Subtitle right below the main title
#st.subheader("A DNABERT-based framework for predicting the functional impact of genomic variants on the human regulome.")

# Sidebar for selecting analysis parameters
st.sidebar.header("Select Analysis Parameters")

# Cancer type selection
#cancer_type = st.sidebar.selectbox("Select Cancer Type", ["Brain", "Lung", "Breast"])
cancer_type= "Brain"

# Analysis type selection
analysis_type = st.sidebar.selectbox("Genomic Regulatory Elements", ["Splice Sites", "TFBS Models"])


# Define a mapping from the analysis type selected in the sidebar to the corresponding folder names
analysis_type_mapping = {
    "Splice Sites": "Splice_Sites",
    "TFBS Models": "TFBS_Models"
}

# Get the folder name based on the selected analysis type
analysis_type_folder = analysis_type_mapping[analysis_type]



# Data source selection
# data_source = st.sidebar.selectbox("Genomic Analysis Tools", ["CaVEMan", "sanger_raw_pindel"])

# In the sidebar
analysis_options = {
    "Substitutions (SNVs)": "CaVEMan",
    "Insertions & Deletions (Indels)": "sanger_raw_pindel"
}

# Create the selectbox with user-friendly keys
selected_analysis = st.sidebar.selectbox(
    "Select Variant Type",
    options=list(analysis_options.keys())
)

# Map the user's choice back to the data_source variable your code uses
data_source = analysis_options[selected_analysis]

# You can add a small helper text for more clarity
st.sidebar.info(f"Displaying data from the {data_source} analysis pipeline.")

# Now, the rest of your code can use the 'data_source' variable as it did before,
# without any other changes needed.



# The 'data' folder is in the same directory as the streamlit_app.py
base_path = f"data/{cancer_type}/" 
tsv_files_path = f"{base_path}{analysis_type_folder}/"
json_files_path = f"{base_path}{analysis_type_folder}/"

# Get the selected file path
clinical_file_path = f"{base_path}patient_clinical_updated.tsv"


def compute_common_patients(group):
    # Extract 'GBM_patient_ids' and convert them to sets
    gbm_ids_series = group['GBM_patient_ids']
    sets_of_ids = [set(s.strip().split(',')) for s in gbm_ids_series]
    # Compute the intersection of IDs
    common_ids = set.union(*sets_of_ids)
    # Compute count and percentage
    count = len(common_ids)
    percentage = round((count / 190) * 100, 2)
    # Prepare the result
    result = group.iloc[0].copy()
    result['GBM_patient_ids'] = ','.join(sorted(common_ids))
    result['common_GBM_patient_count'] = count
    result['common_GBM_patient_percentage'] = percentage
    return result


def calculate_p_values(df_kmf, df_transcript_info):
    kmf = KaplanMeierFitter()
    cph = CoxPHFitter(penalizer=0.1)  # Adding a penalizer

    # Add new columns to df_transcript_info to store results
    df_transcript_info['logrank_p_value'] = None
    df_transcript_info['coxph_p_value'] = None
    df_transcript_info['concordance_index'] = None
    df_transcript_info['hazard_ratio'] = None
    df_transcript_info['variant_information'] = None

    for idx, row in df_transcript_info.iterrows():
        selected_patients_ids = [pid.split('_')[0] for pid in row['patient_ids'].split(',')]
        selected_patients = df_kmf[df_kmf['manifest_patient_id'].isin(selected_patients_ids)]
        df_transcript_info['variant_information'] = df_transcript_info.apply(
            lambda row: f"{row['chromosome']}:{row['variant_start_position']}:{row['ref_nucleotide']}>{row['alternative_nucleotide']}",
            axis=1
        )

        df_kmf["group"] = df_kmf.apply(lambda r: "B" if r['manifest_patient_id'] in selected_patients_ids else "A", axis=1)
        df_kmf["group_numeric"] = df_kmf["group"].apply(lambda x: 1 if x == "B" else 0)
        group_A = df_kmf[df_kmf['group'] == 'A']
        group_B = df_kmf[df_kmf['group'] == 'B']

        if len(group_A) > 1 and len(group_B) > 1:  # Ensure there are enough data points for analysis
            # Perform log-rank test
            results_logrank = logrank_test(group_A['km_time'], group_B['km_time'], event_observed_A=group_A['km_status'], event_observed_B=group_B['km_status'])
            logrank_p_value = results_logrank.p_value

            # Fit Cox Proportional Hazards model
            df_kmf_clean = df_kmf[['manifest_patient_id', 'project_id', 'group', 'group_numeric', 'km_time', 'km_status']].dropna()

            # Diagnostic check for variance
            events = df_kmf_clean['km_status'].astype(bool)

            try:
                cph.fit(df_kmf_clean, duration_col='km_time', event_col='km_status', formula='group_numeric')
                
                coxph_p_value = cph.summary.loc['group_numeric', 'p']
                hazard_ratio = cph.summary.loc['group_numeric', 'exp(coef)']
                concordance_index = cph.concordance_index_

                # Store the results in the transcript info DataFrame
                df_transcript_info.at[idx, 'logrank_p_value'] = logrank_p_value
                df_transcript_info.at[idx, 'coxph_p_value'] = coxph_p_value
                df_transcript_info.at[idx, 'concordance_index'] = concordance_index
                df_transcript_info.at[idx, 'hazard_ratio'] = hazard_ratio
            except Exception as e:
                print(f"Error fitting CoxPH model for index {idx}: {e}")
                df_transcript_info.at[idx, 'logrank_p_value'] = None
                df_transcript_info.at[idx, 'coxph_p_value'] = None
                df_transcript_info.at[idx, 'concordance_index'] = None
                df_transcript_info.at[idx, 'hazard_ratio'] = None
        else:
            df_transcript_info.at[idx, 'logrank_p_value'] = None
            df_transcript_info.at[idx, 'coxph_p_value'] = None
            df_transcript_info.at[idx, 'concordance_index'] = None
            df_transcript_info.at[idx, 'hazard_ratio'] = None
    
    # Move 'variant_information' to the first column
    cols = ['variant_information'] + [col for col in df_transcript_info if col != 'variant_information']
    df_transcript_info = df_transcript_info[cols]
    return df_transcript_info


def plot_km_curve(group_A, group_B, idx, title, logrank_p_value):
    df_temp = group_A[['manifest_patient_id', 'project_id', 'km_time', 'km_status']]
    print(df_temp[df_temp.isnull().any(axis=1)])
    kmf = KaplanMeierFitter()
    kmf.fit(group_A['km_time'], event_observed=group_A['km_status'], label=f'Group A[{len(group_A)} patients]')
    kmf_A_sub = kmf.survival_function_
    ci_A = kmf.confidence_interval_
    kmf.fit(group_B['km_time'], event_observed=group_B['km_status'], label=f'Group B[{len(group_B)} patients]')
    kmf_B_sub = kmf.survival_function_
    ci_B = kmf.confidence_interval_
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=kmf_A_sub.index, 
        y=kmf_A_sub.iloc[:, 0], 
        mode='lines', 
        name=f'Group A[{len(group_A)} patients]'
    ))
    fig.add_trace(go.Scatter(
        x=kmf_B_sub.index, 
        y=kmf_B_sub.iloc[:, 0], 
        mode='lines', 
        name=f'Group B[{len(group_B)} patients]', 
        line=dict(color='orange')  # Set line color to orange for Group B
    ))
    fig.add_trace(go.Scatter(
        x=list(ci_A.index) + list(ci_A.index[::-1]),
        y=list(ci_A.iloc[:, 0]) + list(ci_A.iloc[:, 1][::-1]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=list(ci_B.index) + list(ci_B.index[::-1]),
        y=list(ci_B.iloc[:, 0]) + list(ci_B.iloc[:, 1][::-1]),
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.update_layout(
        title={
            'text': f"{title}<br>Log Rank p-value: {logrank_p_value:.4f}",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Time (Days)',
        yaxis_title='Survival Probability'
    )
    #st.plotly_chart(fig)
    return(fig)






# Function to load the data from all sheets
@st.cache_data
def load_data(tsv_files_path, json_files_path,data_source, analysis_type_folder):
    try:
        df_variants_frequency = pd.read_csv(f"{tsv_files_path}{data_source}_combined_{analysis_type_folder}_Variants_frequency.tsv", sep="\t")
        df_intersect_with_dbsnp =pd.read_csv(f"{tsv_files_path}{data_source}_combined_{analysis_type_folder}_Intersect_withDBSNP.tsv", sep="\t")
        df_transcript_info = pd.read_csv(f"{tsv_files_path}{data_source}_combined_{analysis_type_folder}_Variants_frequency_with_gene_information.tsv", sep="\t")
        df_clinvar = pd.read_csv(f"{tsv_files_path}{data_source}_combined_{analysis_type_folder}_ClinVar_Information.tsv", sep="\t")
        json_file_path=  f"{json_files_path}precomputed_candidate_counts_{data_source}.json"
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
        # st.write(f"{tsv_files_path}{data_source}_combined_{analysis_type_folder}_Variants_frequency.tsv")
        # st.write(f"{tsv_files_path}{data_source}_combined_{analysis_type_folder}_Intersect_withDBSNP.tsv")
        # st.write(f"{tsv_files_path}{data_source}_combined_{analysis_type_folder}_Variants_frequency_with_gene_information.tsv")
        # st.write(f"{tsv_files_path}{data_source}_combined_{analysis_type_folder}_ClinVar_Information.tsv")
        # st.write( f"{json_files_path}precomputed_candidate_counts_{data_source}.json")
        return df_variants_frequency, df_intersect_with_dbsnp, df_transcript_info, df_clinvar, json_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


    
# Load the data
df_variants_frequency, df_intersect_with_dbsnp, df_transcript_info, df_clinvar, json_data = load_data(tsv_files_path, json_files_path, data_source, analysis_type_folder)

# Example: Load clinical data (make sure to define `clinical_file_path`)
df_clinical = pd.read_csv(clinical_file_path, sep='\t')

# def format_floats(value):
#     if pd.notnull(value):
#         if abs(value) >= 1:
#             return "{:.2f}".format(value)
#         elif abs(value) > 0:
#             return "{:.3f}".format(value)
#     return value  # For 0 or NaN, return as is

def format_floats(value):
    if pd.notnull(value):
        if abs(value) >= 1:
            return round(value, 2)
        elif abs(value) > 0:
            return round(value, 3)
    return value  # For 0 or NaN, return as is

def format_floats_decimal(value):
    if pd.notnull(value):
        if abs(value) >= 1:
            return Decimal(value).quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)
        elif abs(value) > 0:
            return Decimal(value).quantize(Decimal('0.000'), rounding=ROUND_HALF_UP)
    return value  # For 0 or NaN, return as is

def advanced_pagination_aggrid(df, df_name, entries_per_page=500):
    total_rows = len(df)
    if total_rows == 0:
        st.write("No entries to display.")
        return

    # Define the columns to display for each analysis type
    columns_to_display_mapping = {
        'Splice Sites': [
            'index',
            'Variant Information',
            'splice_start_position',
            'splice_end_position',
            'splice_sites_affected',
            'Loss of Function based on LogOddRatio',
            'GBM_patient_count',
            'GBM_patient_percentage',
            'strand',
            'transcript_id',
            'exon_id',
            'transcript_type'
        ],
        'TFBS Models': [
            'index',
            'Variant Information',
            'TFBS_start_position',
            'TFBS_end_position',
            'TFBS',
            'TFBS_category',
            'Loss of Function based on LogOddRatio',
            'GBM_patient_count',
            'GBM_patient_percentage',
            'gene_ids',
            'gene_names',
            'gene_biotypes',
            'strands',
            'all_transcript_ids',
            'all_transcript_biotypes',
            'all_exon_ids'
        ]
    }

    # Add 'Variant Information' column
    df['Variant Information'] = df.apply(
        lambda row: f"{row['chromosome']}:{row['variant_start_position']}:{row['ref_nucleotide']}>{row['alternative_nucleotide']}",
        axis=1
    )
    df.index += 1
    df['index'] = df.index

    # Get the columns to display
    columns_to_display = columns_to_display_mapping.get(analysis_type, df.columns.tolist())

    # Select only the desired columns
    df_display = df[columns_to_display].copy()
    float_columns = df_display.select_dtypes(include=['float64']).columns.tolist()
    # for col in float_columns:
    #     df_display[col] = df_display[col].apply(format_floats)
    for col in float_columns:
        df_display[col] = df_display[col].apply(format_floats)
    

    
    

    # Build AgGrid options
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=entries_per_page)
    gb.configure_default_column(filter='agTextColumnFilter', sortable=True, resizable=True)

    grid_options = gb.build()

    


    
    
    # **Add header tooltips using descriptions**
    for col_def in grid_options['columnDefs']:
        col_name = col_def.get('field', col_def.get('headerName'))
        if col_name in descriptions:
            col_def['headerTooltip'] = descriptions[col_name]

    # Display the DataFrame using AgGrid
    grid_response = AgGrid(
        df_display,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        #fit_columns_on_grid_load=True,
        enable_enterprise_modules=False,
        height=500,
        width='100%',
        reload_data=True
    )
    # Get the filtered data
    filtered_data = grid_response['data']
    # Create columns for layout
    st.markdown("""
        <style>
        /* Apply styles to the download button */
        .stDownloadButton > button {
            background-color: #2596be;  /* Non-hover color */
            color: white;
            padding: 8px 16px;
            font-size: 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .stDownloadButton > button:hover {
            background-color: #1e83a9;  /* Hover color */
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    col_entries, col_download = st.columns([9, 1])

    with col_download:
        # Download button for the filtered DataFrame
        tsv_data = pd.DataFrame(filtered_data).to_csv(sep='\t', index=False)
        st.download_button(
            label="Download Data",
            data=tsv_data,
            file_name=df_name,
            mime='text/tab-separated-values',
        )





    
    

    




if df_variants_frequency is not None and df_intersect_with_dbsnp is not None and df_transcript_info is not None:
    # Base descriptions
    descriptions = {
        "Variant Information": "A concise representation of the variant, combining chromosome, position, and nucleotide change in the format 'chromosome:position:reference_nucleotide>alternative_nucleotide'.",
        # "variant_start_position": "The start position of the variant.",
        # "variant_end_position": "The end position of the variant.",
        "GBM_patient_count": "The number of GBM patients with this variant.",
        "GBM_patient_percentage": "The percentage of GBM patients with this variant.",
        "Loss of Function based on LogOddRatio": (
            "The score calculated based on the DNABERT probability for both the reference and alternative sequences. "
            "A higher positive score indicates greater functionality disruption."
        )
    }

    # Adjusting the dynamic start and end position descriptions
    key_prefix = analysis_type.split()[0]

    descriptions[key_prefix + "_start_position"] = f"The start position of the {analysis_type}."
    descriptions[key_prefix + "_end_position"] = f"The end position of the {analysis_type}."

    # Add analysis_type specific descriptions
    if analysis_type == "Splice Sites":
        descriptions.update({
            "splice_sites_affected": "The category of Splice Sites affected.",
            "strand": "The DNA strand ('+' or '-') on which the splice site is located.",
            "transcript_id": "The identifier of the transcript affected by the variant.",
            "exon_id": "The identifier of the exon affected by the variant.",
            "transcript_type": "The type of transcript affected by the variant."
        })
    elif analysis_type == "TFBS Models":
        descriptions.update({
            "TFBS": "The name of the Transcription Factor Binding Site.",
            "TFBS_category": "The category of the TFBS affected.",
            "TFBS_start_position": "The start position of the Transcription Factor Binding Site.",
            "TFBS_end_position": "The end position of the Transcription Factor Binding Site.",
            "gene_ids": "The identifiers of the genes associated with the TFBS.",
            "gene_names": "The names of the genes associated with the TFBS.",
            "gene_biotypes": "The biotypes of the genes associated with the TFBS.",
            "strands": "The DNA strands ('+' or '-') associated with the TFBS.",
            "all_transcript_ids": "All transcript identifiers associated with the TFBS.",
            "all_transcript_biotypes": "All transcript biotypes associated with the TFBS.",
            "all_exon_ids": "All exon identifiers associated with the TFBS."
        })

    
        # Smalldescribtion
    # Dynamic description based on sidebar inputs
    description = f"""
    ### Analysis Overview
    <p>This dashboard provides insights into the genomic variants affecting {analysis_type.lower()} in GDC {cancer_type} cancer patients, analyzed using the {data_source} Genomic Analysis Tool.
    The visualizations below display the distribution of these variants along with their predicted effects on regulatory elements based on DNABERT predictions.</p>
    <p>This dashboard also assesses the <strong>clinical significance</strong> of variants by correlating them with data from dbSNP and ClinVar. You can explore survival analysis plots to understand the potential impact of these variations on patient outcomes.</p> 
    """


    st.markdown(description, unsafe_allow_html=True)
    
    
    # Calculate the required height to center the content
    total_height = 600  # Example total height of the column
    content_height = 300  # Approximate height of the content
    top_padding = (total_height - content_height) // 2

    col1, col2 = st.columns([5, 5])



    with col1:
        try:
            # Add top padding
            st.write('<div style="height: {}px;"></div>'.format(top_padding), unsafe_allow_html=True)
            df_json = pd.DataFrame(json_data).T
            st.dataframe(df_json)
            st.markdown("""
            <div align="center">
                <strong>Data Statistics for Predicted Candidate Variants in the {} Regions</strong><br>
                <span>Including the count of associated DBSNP IDs and ClinVar annotations.</span>
            </div>
            """.format(analysis_type), unsafe_allow_html=True)



        except Exception as e:
            st.error(f"Failed to load sunburst chart: {str(e)}")



    with col2:
        try:
            df_clinical = df_clinical[(df_clinical["project_id"] == "CPTAC-3") | (df_clinical["project_id"] == "TCGA-GBM")].reset_index(drop=True)
            total_patients = df_clinical.shape[0]
            df_clinical['total_patients'] = f'Total GBM Patients: <br> {total_patients}'
            # Replace None values in relevant columns
            df_clinical['primary_diagnosis'] = df_clinical['primary_diagnosis'].fillna('Unknown')
            df_clinical['disease_type'] = df_clinical['disease_type'].fillna('Unknown')

            sunburst_fig = px.sunburst(
                df_clinical,
                path=['total_patients', 'project_id', 'gender', 'primary_diagnosis'],
                color='project_id',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            sunburst_fig.update_layout(
                margin=dict(t=5, l=5, r=5, b=5),
                #sunburstcolorway=px.colors.qualitative.Set2,
                #sunburstcolorway=[ "#AB63FA",  "#00CC96",  "#FFA15A", "#EF553B", "#19D3F3","#636EFA"],
                extendsunburstcolors=True,
                font=dict(size=13, color="white")  # Increase font size here
            )
            
            # print(px.colors.qualitative.Set2)
            # # Print color assignments
            # color_assignments = {proj_id: px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
            #                      for i, proj_id in enumerate(df_clinical['project_id'].unique())}
            # print(color_assignments)

            # Define hover templates
            hover_templates = {
                'total_patients': '<b>Total GBM Patients</b><br>Count: %{value}<extra></extra>',
                'project_id': '<b>Project ID</b>: %{label}<br>Count: %{value}<extra></extra>',
                'gender': '<b>Gender</b>: %{label}<br>Count: %{value}<extra></extra>',
                'primary_diagnosis': '<b>Primary Diagnosis</b>: %{label}<br>Count: %{value}<extra></extra>'
            }

            # Apply hover templates to each trace based on level
            for trace in sunburst_fig.data:
                labels = trace['labels']
                trace_hovertemplates = []
                for label in labels:
                    if label.startswith('Total GBM Patients'):
                        trace_hovertemplates.append(hover_templates['total_patients'])
                    elif label in df_clinical['project_id'].values:
                        trace_hovertemplates.append(hover_templates['project_id'])
                    elif label in df_clinical['gender'].values:
                        trace_hovertemplates.append(hover_templates['gender'])
                    elif label in df_clinical['primary_diagnosis'].values:
                        trace_hovertemplates.append(hover_templates['primary_diagnosis'])
                    else:
                        trace_hovertemplates.append('<b>%{label}</b><br>Count: %{value}<extra></extra>')
                trace.hovertemplate = trace_hovertemplates



            st.plotly_chart(sunburst_fig, use_container_width=True)
            st.markdown("<div style='text-align: center;'>Distribution Chart of the GBM cancer patients collected from GDC portal.</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing clinical data: {str(e)}")

    df_clinical = df_clinical.drop(columns=['total_patients'])
    df_clinical = df_clinical.dropna(subset=['manifest_patient_id', 'project_id', 'km_time', 'km_status', 'disease_type'])
    
    
    
    
    description = f"""
    ### Details Description of the Candidate {analysis_type} with associated Variants  
    <p> Here is the brief description of each columns of the dataframe.</p> 
    """
    st.markdown(description, unsafe_allow_html=True)
    
    # Display column descriptions
    html_content = ""
    for column, description in descriptions.items():
        html_content += f"<p style='margin-bottom:2px; text-indent:20px;'><b>- {column}:</b> {description}</p>"
    st.markdown(html_content, unsafe_allow_html=True)
    st.markdown("")
    
    
    #advanced_pagination(df_transcript_info, f"df_significant_variants_frequency_{data_source}_{analysis_type_folder}.tsv")
    advanced_pagination_aggrid(df_transcript_info, f"df_significant_variants_frequency_{data_source}_{analysis_type_folder}.tsv")

    
    if analysis_type == 'Splice Sites':
        st.write("""
        ### Splice Sites Affected Distribution

        This section provides an overview of the distribution of splice sites affected by genomic variants. The pie chart below categorizes the affected splice sites into 'acceptor' and 'donor' types. Each slice of the pie represents the proportion of affected splice sites within the selected cancer type and analysis parameters. This visualization helps in understanding which types of splice sites are more frequently affected by the variants in the dataset.
        """)
        pie_fig = px.pie(df_variants_frequency, names='splice_sites_affected')
        st.plotly_chart(pie_fig)

    elif analysis_type == 'TFBS Models':
        st.write("""
        ### TFBS Distribution
        
        This section provides an overview of the distribution of Transcription Factor Binding Sites (TFBS) models based on Candidate Variants, DBSNP Counts, ClinVar Counts, and model accuracy, with models sorted in decreasing order of predicted Candidate Variants. The top panel shows stacked bar plots of Candidate Variant and DBSNP counts, while the bottom panel represents inverted ClinVar counts. Accuracy for each model is displayed as a line plot on a secondary y-axis in the top panel. The Candidate Variants, identified using Pindel, are visualized in descending order. Vertical lines are used to separate every five TFBS models, providing a clear visualization of the variant distributions and model performance across these top transcription factor binding site models.
        """)
        st.markdown(
            "<h4 style='text-align: center; font-weight: bold;'>"
            "Analysis of TFBS Models: Candidate Variants, DBSNP Counts, ClinVar Counts, and Accuracy"
            "</h3>",
            unsafe_allow_html=True
        )
        # Group by 'TFBS' and count occurrences
        tfbs_counts = df_variants_frequency['TFBS'].value_counts().sort_values(ascending=False).reset_index()
        tfbs_counts.columns = ['TFBS', 'Variant_Count']
        #st.dataframe(tfbs_counts)
        dbsnp_counts = df_intersect_with_dbsnp['TFBS'].value_counts().sort_values(ascending=False).reset_index()
        dbsnp_counts.columns = ['TFBS', 'DBSNP_Count']
        #st.dataframe(dbsnp_counts)
        clinical_tfbs_counts =  df_clinvar['TFBS'].value_counts().reset_index()
        clinical_tfbs_counts.columns = ['TFBS', 'ClinVar_Count']
        #st.dataframe(clinical_tfbs_counts)
        
        ##getting best Accuracy
        df_best_accuracy = pd.read_csv("/home/campus.stonybrook.edu/pdutta/Github/Postdoc/DNABERT_data_processing/TFBS/300bp_TFBS_accuracy_Stat.tsv", sep="\t")
        # Multiply by 100 and format to two decimal places
        df_best_accuracy[['eval_acc', 'eval_f1', 'eval_mcc']] = df_best_accuracy[['eval_acc', 'eval_f1', 'eval_mcc']].apply(lambda x: (x * 100).round(2))
        df_best_accuracy = df_best_accuracy[df_best_accuracy['eval_acc']>=85]
        #st.dataframe(df_best_accuracy)
        
        # Step 4: Merge the counts with df_best_accuracy data to have both accuracy and counts
        df_final = pd.merge(df_best_accuracy, tfbs_counts, left_on='tags', right_on='TFBS', how='inner')
        #st.dataframe(df_final)
        # Merge with dbsnp_counts
        df_final = pd.merge(df_final, dbsnp_counts, on='TFBS', how='left')
        #st.dataframe(df_final)
        # Merge with clinical_tfbs_counts
        df_final = pd.merge(df_final, clinical_tfbs_counts, on='TFBS', how='left')
        #st.dataframe(df_final)
        
        df_final.fillna(0, inplace=True)
        df_final['DBSNP_Count'] = df_final['DBSNP_Count'].astype(int)
        # Step 5: Sort by accuracy
        df_final_sorted = df_final.sort_values(by='Variant_Count', ascending=False).reset_index(drop=True)
        df_final_sorted.fillna(0, inplace=True)
        df_final_sorted[['Variant_Count', 'DBSNP_Count', 'ClinVar_Count']] = df_final_sorted[['Variant_Count', 'DBSNP_Count', 'ClinVar_Count']].astype(int)
        
        
        
        # Invert ClinVar Count by multiplying by -1 for downward bars
        df_final_sorted['ClinVar_Count_Inverted'] = df_final_sorted['ClinVar_Count'] * -1

        # Dynamic dropdown options based on DataFrame length
        max_labels = len(df_final_sorted)
        options = ["All"] + [f"Top {i}" for i in [10, 20, 50, 100, 200, 300] if i < max_labels] + [f"Top {max_labels}"]
        num_labels = st.selectbox("Select the number of TFBS models to display:", options=options, index=0)

        # Filter data based on selection
        if num_labels == "All":
            df_final_sorted_display = df_final_sorted
        else:
            top_n = int(num_labels.split(" ")[1])
            df_final_sorted_display = df_final_sorted.head(top_n)

        
        #st.dataframe(df_final_sorted_display)
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0,
            specs=[[{"secondary_y": True}], [{}]]
        )

        # Add stacked bar plot for Candidate Variants counts
        fig.add_trace(
            go.Bar(
                x=df_final_sorted_display['tags'],
                y=df_final_sorted_display['Variant_Count'],
                name='Candidate Variants Count',
                marker_color='rgba(255, 99, 71, 0.7)',
                opacity=0.7
            ),
            row=1, col=1
        )

        # Add stacked bar plot for DBSNP counts
        fig.add_trace(
            go.Bar(
                x=df_final_sorted_display['tags'],
                y=df_final_sorted_display['DBSNP_Count'],
                name='DBSNP Count',
                marker_color='rgba(0, 206, 209, 0.7)',
                opacity=0.7
            ),
            row=1, col=1
        )

        # Add bar plot for ClinVar counts
        fig.add_trace(
            go.Bar(
                x=df_final_sorted_display['tags'],
                y=df_final_sorted_display['ClinVar_Count_Inverted'],
                name='ClinVar Count',
                marker_color='rgba(60, 179, 113, 0.9)',
                opacity=0.7
            ),
            row=2, col=1
        )

        # Set y-axis ticks for ClinVar counts to display positive values
        fig.update_yaxes(
            tickvals=[-2, -4, -6, -8, -10, -12],  # Adjusted tick values
            ticktext=[2, 4, 6, 8, 10, 12],
            row=2, col=1
        )

        # Add line plot for Accuracy
        fig.add_trace(
            go.Scatter(
                x=df_final_sorted_display['tags'],
                y=df_final_sorted_display['eval_acc'],
                name='Accuracy (%)',
                mode='lines+markers',
                line=dict(color='royalblue', width=2)
            ),
            secondary_y=True, row=1, col=1
        )

        # Add vertical lines every 5 models
        for i in range(5, len(df_final_sorted_display['tags']), 5):
            x_value = df_final_sorted_display['tags'][i]
            fig.add_shape(
                type='line',
                x0=x_value,
                y0=0,
                x1=x_value,
                y1=1,
                xref='x',
                yref='paper',
                line=dict(color='grey', width=0.3, dash='dot')
            )

        # Update layout
        fig.update_layout(
            yaxis=dict(title='Candidate Variant & DBSNP Count', titlefont=dict(size=14), tickfont=dict(size=12)),
            yaxis2=dict(title='Accuracy (%)', range=[0, 100], titlefont=dict(size=14), tickfont=dict(size=12)),
            yaxis3=dict(title='ClinVar Count', titlefont=dict(size=14), tickfont=dict(size=12)),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.4,
                xanchor='center',
                x=0.5,
                font=dict(size=14)
            ),
            barmode='stack',
            height=800
        )

        # Adjust axes titles
        fig.update_yaxes(title_text="Candidate Variant & DBSNP Count", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="ClinVar Count", row=2, col=1)
        fig.update_xaxes(title_text="TFBS Models [Sorted based on predicting candidate variants]", row=2, col=1)

        # Set x-axis ticks based on filtered data
        fig.update_xaxes(
            tickmode='array',
            tickvals=df_final_sorted_display['tags'],
            ticktext=df_final_sorted_display['tags'],
            tickangle=-45,
            tickfont=dict(size=12)
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        
        
   
        
        
        
        
        
        
        
        
        
        
        
        
       
    
    # Filter data based on selected splice site
    # Add formatted text above the select box
    if analysis_type == 'Splice Sites':
        st.write("""#### Select Splice Site to Filter Data to visualize the selected splice data and the distribution of patients affected by variants:""")

        # Create the select box
        selected_site = st.selectbox(
            "Choose a splice site to see detailed information and patient distribution",
            df_transcript_info['splice_sites_affected'].unique()
        )
        filtered_data = df_transcript_info[df_transcript_info['splice_sites_affected'] == selected_site].reset_index(drop=True)
        filtered_clinvar = df_clinvar[df_clinvar['splice_sites_affected']==f"{selected_site}"].reset_index(drop=True)
        #st.dataframe(filtered_clinvar)

        st.write(f"""
        ##### Filtered Data for '{selected_site}' Splice Site 

        The table below displays detailed information for genomic variants that affect the selected splice site type ('{selected_site}'), but ***only includes variants that impact at least 10% of patients***. This data highlights the chromosome location, nucleotide changes, and predicted impact on splice site function, helping to focus on variants with more significant patient impact for further analysis.
    """)
        advanced_pagination_aggrid(filtered_data,f"df_significant_variants_frequency_{data_source}_{analysis_type_folder}_{selected_site}.tsv")
    elif analysis_type == 'TFBS Models':
        st.write("""#### Select TFBS data to Filter Data to visualize the selected TFBS and the distribution of patients affected by variants:""")

        # Create the select box
        selected_site = st.selectbox(
            "Choose a transcription factor site to see detailed information and patient distribution",
            df_transcript_info['TFBS'].unique()
        )
        filtered_data = df_transcript_info[df_transcript_info['TFBS'] == selected_site].reset_index(drop=True)
        filtered_clinvar = df_clinvar[df_clinvar['TFBS']==f"{selected_site}"].reset_index(drop=True)
        # st.dataframe(filtered_clinvar)
        # st.dataframe(df_clinvar)

        st.write(f"""
        ##### Filtered Data for '{selected_site}'  

        The table below displays detailed information for genomic variants that affect the selected splice site type ('{selected_site}'), but ***only includes variants that impact at least 10% of patients***. This data highlights the chromosome location, nucleotide changes, and predicted impact on splice site function, helping to focus on variants with more significant patient impact for further analysis.
    """)
        advanced_pagination_aggrid(filtered_data,f"df_significant_variants_frequency_{data_source}_{analysis_type_folder}_{selected_site}.tsv")

    
    
    
    # Create bins for the histogram
    bins = [i for i in range(10, 101, 10)]
    filtered_data['percentage_bin'] = pd.cut(filtered_data['GBM_patient_percentage'], bins=bins, right=True, include_lowest=True)
    #st.dataframe(filtered_data)
    
    if analysis_type == 'Splice Sites':
        group_columns = ["chromosome", "splice_start_position", "splice_end_position"]
    elif analysis_type == 'TFBS Models':
        group_columns = ["chromosome", "TFBS_start_position", "TFBS_end_position"]
        
    if len(filtered_data) > 1:
        unique_regions = filtered_data.groupby(group_columns).apply(compute_common_patients).reset_index(drop=True)
        unique_regions['common_GBM_percentage_bin'] = pd.cut(unique_regions['common_GBM_patient_percentage'], bins=bins, right=True, include_lowest=True)
        bin_counts_unique = unique_regions['common_GBM_percentage_bin'].value_counts().sort_index().reset_index()
        bin_counts_unique.columns = ['common_GBM_percentage_bin', 'count']
        bin_counts_unique['common_GBM_percentage_bin'] = bin_counts_unique['common_GBM_percentage_bin'].astype(str)
    else:
        unique_regions =filtered_data
        # Plot for unique regions
        bin_counts_unique = unique_regions['percentage_bin'].value_counts().sort_index().reset_index()
        bin_counts_unique.columns = ['percentage_bin', 'count']
        bin_counts_unique['percentage_bin'] = bin_counts_unique['percentage_bin'].astype(str)
    #print(unique_regions)
    st.dataframe(unique_regions)
    # Create bins for the histogram of unique regions

    #Plot for unique regions
    # bin_counts_unique = unique_regions['percentage_bin'].value_counts().sort_index().reset_index()
    # bin_counts_unique.columns = ['percentage_bin', 'count']
    # bin_counts_unique['percentage_bin'] = bin_counts_unique['percentage_bin'].astype(str)

   

    # Prepare varaint data for Plotly
    bin_counts = filtered_data['percentage_bin'].value_counts().sort_index().reset_index()
    bin_counts.columns = ['percentage_bin', 'count']
    bin_counts['percentage_bin'] = bin_counts['percentage_bin'].astype(str)
    
    st.markdown("""
    <div style="text-align: center; font-size: 30px; font-weight: 600;">
        <br> Distribution of Patients Affected by Regulatory Regions and Variants
    </div>
    """, unsafe_allow_html=True)
    # Plotly bar chart
    st.write("""

    This section provides two bar charts that illustrate how genetic variations affect the patient population: 
    
    ### 1. Region-Wise Distribution
    The **region-wise distribution** chart shows the number of unique **genomic regions** affecting different proportions of the patient population. Each bar represents the count of unique regions within a given percentage range (e.g., 10-20%, 20-30%, etc.).

    - **Purpose**: This chart highlights the impact of distinct genomic regions on the patient population, providing insights into which regions are more prevalent across patients.

    ---

    ### 2. Variant-Wise Distribution (Interactive)
    The **variant-wise distribution** chart illustrates the total number of individual **variants** affecting patients, grouped by percentage ranges (e.g., 10-20%, 20-30%, etc.).

    - **Purpose**: This chart helps explore the overall burden of genetic variants, considering all individual occurrences across all genomic regions.
    - **Interactive Features**:
        - **Hover**: Hover over a bar to see the count of variants in that percentage range.
        - **Click**: Click on a bar to filter and explore detailed data about the variants within that range.
    This interactive chart allows for deeper exploration of the variant-wise data and its distribution among patients.
    
    ---

    Together, these visualizations provide complementary insights into the prevalence of genomic regions and individual variants across the patient population.
    """, unsafe_allow_html=True)
    

#     To interact with the chart:
#     <ul>
#         <li><strong>Hover over the bars</strong> to see the count of patients in each percentage range.</li>
#         <li><strong>Click on a bar</strong> to filter the data and display detailed variant information for that range.</li>
#     </ul>

#     This visualization helps identify how widely different variant regions affect the patient population, providing insights into the distribution and prevalence of specific variants within the selected splice site type.
# """, unsafe_allow_html=True)
    
    
    ## Plotly bar chart distribution for regions 
    if len(filtered_data) > 1:
        fig_unique = px.bar(
            bin_counts_unique, 
            x='common_GBM_percentage_bin', 
            y='count', 
            color='common_GBM_percentage_bin',
            title="Percentage of Patients Affected by Unique Regulatory Regions",
            labels={'common_GBM_percentage_bin': 'Percentage of Patients', 'count': 'Count'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
    else:
        fig_unique = px.bar(
            bin_counts_unique, 
            x='percentage_bin', 
            y='count', 
            color='percentage_bin',
            title="Percentage of Patients Affected by Unique Regulatory Regions",
            labels={'percentage_bin': 'Percentage of Patients', 'count': 'Count'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
    # Update the layout to standardize the title formatting
    fig_unique.update_layout(
        title={
            'text': "Percentage of Patients Affected by Unique Regulatory Regions",
            'y': 0.95,  # Vertical position of the title
            'x': 0.5,   # Horizontal position (centered)
            'xanchor': 'center',
            'yanchor': 'top'
        },
        title_font=dict(size=18)  # Adjust font size as needed
    )

    fig_unique.update_traces(texttemplate='%{y}', textposition='outside')
    st.plotly_chart(fig_unique)
    
    
    ## Plotly bar chart distribution for variants 
    fig = px.bar(bin_counts, x='percentage_bin', y='count', color='percentage_bin',
                 title="Percentage of Patients Affected by Variants.",
                 labels={'percentage_bin': 'Percentage of Patients', 'count': 'Count'},
                 color_discrete_sequence=px.colors.qualitative.Set2)
    # Add count labels on top of the bars
    fig.update_traces(texttemplate='%{y}', textposition='outside')

    # Capture click events using streamlit-plotly-events
    selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False)

    # Display filtered data based on selected bin
    if selected_points:
        clicked_bin = selected_points[0]['x']
        st.write(f"""
        ### Detailed Variant Information for Patients in the {clicked_bin} Range.

        The table below shows the variant regions affecting patients within the {clicked_bin} percentage range.
        """)

        # Parse the bin range
        # bin_start, bin_end = map(int, clicked_bin.replace('(', '').replace(']', '').split(','))
        # filtered_rows = filtered_data[(filtered_data['percentage_of_patients'] >= bin_start) &
        #                               (filtered_data['percentage_of_patients'] < bin_end)].reset_index(drop=True)
        # Parse the bin range
        bin_start, bin_end = map(lambda x: float(x.strip()), clicked_bin.replace('(', '').replace(']', '').split(','))
        #print(bin_start, bin_end)
        filtered_rows = filtered_data[(filtered_data['GBM_patient_percentage'] > bin_start) &
                                      (filtered_data['GBM_patient_percentage'] <= bin_end)].reset_index(drop=True)
        
        #advanced_pagination_aggrid(filtered_rows, f"df_significant_variants_frequency_{data_source}_{analysis_type_folder}_{selected_site}_{bin_start}_{bin_end}.tsv")
        df_transcript_info = calculate_p_values(df_clinical, filtered_rows)
        #advanced_pagination_aggrid(df_transcript_info, f"df_significant_clinical_{data_source}_{analysis_type_folder}_{selected_site}_{bin_start}_{bin_end}.tsv")
        df_clinical = df_clinical.drop(columns=['group', 'group_numeric'])
        
        
        
         # Add a small header for the dataframe
        st.markdown("""
        ### Variants Information
        To observe the survival plot for that particular variant click on specific row.
        """)



#         # Display the DataFrame without patient_ids column
#         #st.dataframe(df_transcript_info.drop(columns=['patient_ids', 'percentage_bin', 'variant_start_position', 'variant_end_position', 'ref_nucleotide', 'alternative_nucleotide']))
        

        # Define the columns for each analysis type
        splice_columns = [
            'variant_information', 'splice_start_position', 'splice_end_position',
            'Loss of Function based on LogOddRatio', 'transcript_id', 'exon_id',
            'transcript_type', 'GBM_patient_count', 'logrank_p_value', 'coxph_p_value',
            'concordance_index', 'hazard_ratio'
        ]

        tfbs_columns = [
            'variant_information', 'TFBS_start_position', 'TFBS_end_position', 
            'variant_start_position', 'variant_end_position', 'Ref_probab', 'Alt_probab',
            'ScoreChange', 'Loss of Function based on LogOddRatio', 'GBM_patient_count', 
            'gene_ids', 'gene_names', 'gene_biotypes', 'contigs', 'all_transcript_ids', 
            'all_transcript_biotypes', 'all_exon_ids', 'TFBS_category', 
            'GBM_patient_percentage', 'logrank_p_value', 'coxph_p_value', 
            'concordance_index', 'hazard_ratio'
        ]

        # Filter columns based on analysis type
        if analysis_type == 'Splice Sites':
            display_df = df_transcript_info[splice_columns]
        elif analysis_type == 'TFBS Models':
            display_df = df_transcript_info[tfbs_columns]
        # Configure the grid options
        gb = GridOptionsBuilder.from_dataframe(display_df)
        # Increase font size for both cells and headers
        custom_css = {
            ".ag-header-cell-label": {
                "font-size": "16px !important",
            },
            ".ag-cell": {
                "font-size": "14px !important",
            },
            ".ag-row-selected": {
                "background-color": "#d3d3d3 !important",  # Light grey background for the selected row
            },
        }

        # Apply pagination, sidebar, single selection, and row numbers
        gb.configure_pagination(paginationAutoPageSize=False)
        gb.configure_side_bar()  # Add a sidebar
        gb.configure_selection('single', suppressRowDeselection=True)  # Enable single row selection
        gb.configure_grid_options(domLayout='normal')
        #gb.add_row_number_column()  # Add row numbers to the grid

        # Build grid options with custom CSS
        grid_options = gb.build()

        # Display the DataFrame with interactive elements
        grid_response = AgGrid(
            df_transcript_info,
            gridOptions=grid_options,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            fit_columns_on_grid_load=False,
            enable_enterprise_modules=True,
            custom_css=custom_css  # Apply custom CSS for font size and visibility
        )
        

        # Extract selected rows safely
        selected_rows = grid_response.get('selected_rows', [])

        if len(selected_rows) > 0:
            #st.write("Selected Rows:", selected_rows)  # For debugging, showing the selected rows
            variant_info = selected_rows.iloc[0]['variant_information']
            selected_variant_data = df_transcript_info[df_transcript_info['variant_information'] == variant_info].iloc[0]
            selected_patients_ids = [pid.split('_')[0] for pid in selected_variant_data['patient_ids'].split(',')]
            selected_patients = df_clinical[df_clinical['manifest_patient_id'].isin(selected_patients_ids)]
            df_clinical["group"] = df_clinical.apply(lambda r: "B" if r['manifest_patient_id'] in selected_patients_ids else "A", axis=1)
            df_clinical["group_numeric"] = df_clinical["group"].apply(lambda x: 1 if x == "B" else 0)
            # st.write(selected_variant_data, selected_patients_ids, selected_patients, variant_info)
            #st.dataframe(filtered_clinvar)
            
            st.markdown("<h3 style='text-align: center;'>Detailed Variant Information from ClinVar</h3>", unsafe_allow_html=True)

            st.markdown("Below, you'll find a comprehensive breakdown of the variant's details, organized into categories for easy navigation. Each tab provides specific insights into different aspects of the variant, including its genomic details, population frequencies, clinical significance, molecular impact, and more.")

            # Ensure that selected_variant_start is converted to the correct type if needed
            selected_variant_start = int(variant_info.split(":")[1])

            
            if analysis_type == 'Splice Sites':
                filtered_result = filtered_clinvar[
                    (filtered_clinvar['chromosome'] == selected_rows.iloc[0]['chromosome']) &
                    (filtered_clinvar['splice_start_position'] == selected_rows.iloc[0]['splice_start_position']) &
                    (filtered_clinvar['splice_end_position'] == selected_rows.iloc[0]['splice_end_position']) &
                    (filtered_clinvar['variant_start_position'] == selected_variant_start)
                ].reset_index(drop=True)

            elif analysis_type == 'TFBS Models':
                filtered_result = filtered_clinvar[
                    (filtered_clinvar['chr'] == selected_rows.iloc[0]['chromosome']) &
                    (filtered_clinvar['start_x'] == selected_rows.iloc[0]['TFBS_start_position']) &
                    (filtered_clinvar['end_x'] == selected_rows.iloc[0]['TFBS_end_position']) &
                    (filtered_clinvar['variant_start_position'] == selected_variant_start)
                ].reset_index(drop=True)
            #st.dataframe(filtered_result)

            # Check if the filtered DataFrame is empty
            if filtered_result.empty:
                st.write(f"***There is no clinical information available from ClinVar for the variant {variant_info}.***")
            else:
                filtered_result = filtered_result.drop_duplicates(subset=['chr', 'start_x', 'end_x', 'REF_x', 'ALT_x', 'rsID']).reset_index(drop=True)
                st.dataframe(filtered_result)
                for index, row in filtered_result.iterrows():
                    #st.markdown(f"### Variant Information for ClinVar Entry {index+1}")
                    st.markdown("""
                        <style>
                        .tooltip {
                            position: relative;
                            display: inline-block;
                            border-bottom: 1px dotted black;
                        }

                        .tooltip .tooltiptext {
                            visibility: hidden;
                            width: 220px;
                            background-color: black;
                            color: #fff;
                            text-align: center;
                            border-radius: 6px;
                            padding: 5px;
                            position: absolute;
                            z-index: 1;
                            bottom: 125%; /* Position the tooltip above the text */
                            left: 50%;
                            margin-left: -110px;
                            opacity: 0;
                            transition: opacity 0.3s;
                        }

                        .tooltip:hover .tooltiptext {
                            visibility: visible;
                            opacity: 1;
                        }
                        
                        /* Tab container: Make tabs stretch across the full width */
                        div.stTabs div[data-baseweb="tab-list"] {
                            display: flex;
                            justify-content: space-between;
                        }

                        /* Individual tabs */
                        div.stTabs div[data-baseweb="tab"] {
                            flex-grow: 1; /* Make each tab grow to fill the available space */
                            background-color: #f0f0f5; /* Light grey background for tabs */
                            color: #333; /* Dark text color */
                            font-weight: bold; /* Bold font */
                            border-radius: 5px; /* Rounded corners */
                            padding: 10px; /* Padding inside tabs */
                            margin-right: 5px; /* Space between tabs */
                            text-align: center; /* Center the text in the tab */
                        }

                        /* Active tab */
                        div.stTabs div[data-baseweb="tab"][aria-selected="true"] {
                            background-color: #4CAF50; /* Green background for active tab */
                            color: white; /* White text for active tab */
                            border: 2px solid #4CAF50; /* Border for active tab */
                        }

                        /* Hover effect for tabs */
                        div.stTabs div[data-baseweb="tab"]:hover {
                            background-color: #ddd; /* Light grey background on hover */
                            color: #000; /* Black text on hover */
                            cursor: pointer; /* Pointer cursor on hover */
                        }
                                            </style>
                    """, unsafe_allow_html=True)
                    
                    
                    # Create tabs
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                        "Variant Description",
                        "Population Frequency",
                        "Clinical Significance and Disease Association",
                        "Molecular Consequence",
                        "Other Identifiers",
                        "Clinical Observations",
                        "Quality and Filtering"
                    ])

                    with tab1:
                        # Genomic Location and Variant Description
                        rsid = row['rsID']
                        rsid_link = f"https://www.ncbi.nlm.nih.gov/snp/{rsid}"
                        st.markdown(f"**rsID:** [**{rsid}**]({rsid_link})", unsafe_allow_html=True)
                        st.markdown(f"""
                            **HGVS Nomenclature** 
                            <span class="tooltip">[?]
                              <span class="tooltiptext">HGVS nomenclature that provides a standardized way to describe the variant</span>
                            </span>: {row['CLNHGVS']}
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                            **Variant Classification** 
                            <span class="tooltip">[?]
                              <span class="tooltiptext">Indicating the type of mutation</span>
                            </span>: {row['CLNVC']}
                        """, unsafe_allow_html=True)

                    with tab2:
                        # Population Frequency
                        st.write(f"**Minor Allele Frequency:** {row['minorAlleleFreq']}")
                        st.markdown(f"""
                            **AF ESP** 
                            <span class="tooltip">[?]
                              <span class="tooltiptext">Allele frequency in the Exome Sequencing Project (ESP), primarily from European and African American ancestry</span>
                            </span>: {row['AF_ESP']}
                        """, unsafe_allow_html=True)

                        # AF ExAC with Tooltip
                        st.markdown(f"""
                            **AF ExAC** 
                            <span class="tooltip">[?]
                              <span class="tooltiptext">Allele frequency in the Exome Aggregation Consortium (ExAC), which aggregates exome data from diverse populations</span>
                            </span>: {row['AF_EXAC']}
                        """, unsafe_allow_html=True)

                        # AF 1000 Genomes Project with Tooltip
                        st.markdown(f"""
                            **AF 1000 Genomes Project** 
                            <span class="tooltip">[?]
                              <span class="tooltiptext">Allele frequency in the 1000 Genomes Project, covering genetic variation across multiple populations worldwide</span>
                            </span>: {row['AF_TGP']}
                        """, unsafe_allow_html=True)

                    with tab3:
                        # Clinical Significance and Disease Association
                        st.write(f"**Clinical Significance:** {row['CLNSIG']}")
                        st.write(f"**Clinical Condition:** {row['CLNDN']}")
                        st.write(f"**Additional conditions related to the primary clinical condition:** {row['CLNDNINCL']}")
                        st.write(f"**Associated Disease Databases:** {row['CLNDISDB']}")
                        st.write(f"**Additional Associated Disease Databases:** {row['CLNDISDBINCL']}")
                        st.write(f"**Review Status:** {row['CLNREVSTAT']}")

                    with tab4:
                        # Molecular Consequence
                        st.write(f"**Molecular Consequence:** {row['MC']}")
                        gene_info = row['GENEINFO']
                        genes = gene_info.split('|')  # Split if there are multiple genes listed
                        # Initialize a list to hold formatted gene links
                        gene_links = []
                        # Iterate through each gene and create a hyperlink
                        for gene in genes:
                            gene_name, gene_id = gene.split(':')  # Split into gene name and gene ID
                            gene_link = f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}"  # Create the link to NCBI Gene
                            gene_links.append(f"[**{gene_name}**]({gene_link})")  # Format as a Markdown link
                        # Combine original gene_info and the formatted links
                        st.markdown(f"**Gene Information:** {gene_info} ({' | '.join(gene_links)})", unsafe_allow_html=True)

                    with tab5:
                        # Other Identifiers
                        st.write(f"**Allele ID:**(Unique identifier within ClinVar) {row['ALLELEID']}")
                        st.write(f"**DBVARID:**(Identifier for structural variants in the dbVar database) {row['DBVARID']}")
                        st.write(f"**RS:** {row['RS']}")

                    with tab6:
                        # Clinical Observations
                        st.write(f"**SCIDN:** {row['SCIDN']}")
                        st.write(f"**SCIDNINCL:** {row['SCIDNINCL']}")
                        st.write(f"**SCIDISDB:** {row['SCIDISDB']}")
                        st.write(f"**SCIDISDBINCL:** {row['SCIDISDBINCL']}")
                        st.write(f"**SCIREVSTAT:** {row['SCIREVSTAT']}")
                        st.write(f"**SCI:** {row['SCI']}")
                        st.write(f"**SCIINCL:** {row['SCIINCL']}")

                    with tab7:
                        # Quality and Filtering
                        st.write(f"**Quality:** {row['QUAL_x']}")
                        st.write(f"**Filter:** {row['FILTER_x']}")
                    
                    
                    

                    st.markdown("---")  # Add a horizontal line to separate entries
            
            

            



            #st.dataframe(df_clinical)
            # st.write(df_clinical.shape[0])
            # st.dataframe(df_clinical.groupby('group').size().reset_index(name='count'))

            # Aggregating data to count occurrences
            df_counts = df_clinical.groupby(['group', 'project_id']).size().reset_index(name='count')
            fig1 = px.bar(df_counts, x='group', y='count', color='project_id', title='Project ID Distribution by Group', 
                          color_discrete_sequence=px.colors.qualitative.Set3)
            # Center the title and adjust font size
            fig1.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})


            

            df_counts = df_clinical.groupby(['group', 'gender']).size().reset_index(name='count')
            fig2 = px.bar(df_counts, x='group', y='count', color='gender', title='Gender Distribution by Group', 
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})


            # Aggregating data to count occurrences for race
            df_counts = df_clinical.groupby(['group', 'race','ethnicity']).size().reset_index(name='count')
            df_counts['group'] = df_counts['group'].replace({'A': 'Group A', 'B': 'Group B'})
            
            

            # Create the Sunburst chart
            fig3 = px.sunburst(
                df_counts,
                path=['group', 'race', 'ethnicity'],  # Define the hierarchy: group -> race
                values='count',  # The size of each segment
                color='group',  # Color by race to distinguish between categories
                title='Race and Ethnicity Distribution within Groups',
                color_discrete_sequence=px.colors.qualitative.Set2  # Use Plotly color palette
            )

            # Customize the layout to make it more visually clear
            fig3.update_layout(
                margin=dict(t=40, l=0, r=0, b=0),
                sunburstcolorway=px.colors.qualitative.Set2,  # Ensure consistent color usage
                uniformtext=dict(minsize=10, mode='hide'),  # Adjust text visibility
            )
            fig3.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})

            
#             df_counts = df_clinical.groupby(['group', 'primary_diagnosis']).size().reset_index(name='count')
#             fig5 = px.bar(df_counts, x='group', y='count', color='primary_diagnosis', title='Primary Diagnosis Distribution by Group', 
#                           color_discrete_sequence=px.colors.qualitative.Set2)
#             fig5.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})

#             df_counts = df_clinical.groupby(['group', 'disease_type']).size().reset_index(name='count')
#             st.dataframe(df_counts)
#             fig6 = px.bar(df_counts, x='group', y='count', color='disease_type', title='Disease Type Distribution by Group', 
#                           color_discrete_sequence=px.colors.qualitative.Set2)
#             fig6.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})
            
            
            
           
            
            
            
            
            
#             group_A = df_clinical[(df_clinical['group'] == 'A') & (df_clinical['disease_type'] == "GBM")]
#             group_B = df_clinical[(df_clinical['group'] == 'B') & (df_clinical['disease_type'] == "LGG")]
#             fig7 =  plot_km_curve(group_A, group_B, title=f"LGG Patients(Group-B) Harboring Variant {variant_info} vs. GBM Patients(Group-A) Lacking the Variant.")
#             fig7.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})
            
#             group_A = df_clinical[(df_clinical['group'] == 'A') & (df_clinical['disease_type'] == "LGG")]
#             group_B = df_clinical[(df_clinical['group'] == 'B') & (df_clinical['disease_type'] == "GBM")]
#             fig8 =  plot_km_curve(group_A, group_B, title=f"GBM Patients(Group-B) Harboring Variant {variant_info} vs. LGG Patients(Group-A) Lacking the Variant.")
#             fig8.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})
            
            # Title centered
            st.markdown("<h3 style='text-align: center;'>Visualization of Clinical Data Stratification by Cohorts</h3>", unsafe_allow_html=True)

            # Arranging data distribution plots in a 2+2+1 format
            col1, col2 = st.columns([1, 1])  # Give more space to the column with the Sunburst chart

            with col1:
                st.plotly_chart(fig1)  # Disease Type Distribution by Group
                st.plotly_chart(fig2)  # Primary Diagnosis Distribution by Group


            with col2:
                st.markdown("<div style='height: 250px;'></div>", unsafe_allow_html=True)  # Add vertical space before the Sunburst
                st.plotly_chart(fig3)  # Sunburst chart for Race and Ethnicity Distribution
                st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)  # Add vertical space after the Sunburst
            
            # Title centered
            st.markdown("<h3 style='text-align: center;'>Interpreting Survival Analysis for GBM Brain cancer Patients.</h3>", unsafe_allow_html=True)
            
            
            
            group_A = df_clinical[(df_clinical['group'] == 'A') & (df_clinical['disease_type'] == "GBM")]
            group_B = df_clinical[(df_clinical['group'] == 'B') & (df_clinical['disease_type'] == "GBM")]
            logrank_p_value=selected_rows.iloc[0]['logrank_p_value']
            fig4 =  plot_km_curve(group_A, group_B, idx=None, title=f"KM Plot for GBM Brain Cancer Patients with Variant {variant_info}", logrank_p_value=logrank_p_value)
            fig4.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})
            st.plotly_chart(fig4)  
            #fig4.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})


                
            
#             st.markdown("<h3 style='text-align: center;'>Survival Plot of Mortality Risk in Different Brain Cancer Patients.</h3>", unsafe_allow_html=True)
#             col1, col2 = st.columns(2)

#             with col1:
#                 st.plotly_chart(fig7)  # First Kaplan-Meier plot

#             with col2:
#                 st.plotly_chart(fig8)  # Second Kaplan-Meier plot
            


            

            
            ####################################################################################################################################################################
            
    
            # group_A = df_clinical[df_clinical['group'] == 'A']
            # group_B = df_clinical[df_clinical['group'] == 'B']
            # plot_km_curve(group_A, group_B, title=variant_info, logrank_p_value=selected_rows.iloc[0]['logrank_p_value'])


        #return group_A, group_B, selected_variant_data['variant_information'], selected_variant_data['logrank_p_value']

#         # Handle the selection
#         if selected_rows:  # Ensure there are selected rows
#             st.write("Selected Variant Information:", selected_rows[0])  # Debugging selected variant
#             selected_variant = selected_rows[0]['variant_information']
#             st.session_state['variant_clicked'] = selected_variant

#         # Generate KM plot if a variant is clicked
#         if 'variant_clicked' in st.session_state and st.session_state['variant_clicked']:
#             st.write("Generating KM plot for variant:", st.session_state['variant_clicked'])  # Debugging KM plot generation
#             group_A, group_B, variant_information, logrank_p_value = get_km_plot_data(st.session_state['variant_clicked'])
#             plot_km_curve(group_A, group_B, idx=None, title=variant_information, logrank_p_value=logrank_p_value)



        # Function to generate KM plot on clicking a variant
#         @st.cache_data
#         def get_km_plot_data(variant_info):
#             selected_variant_data = df_transcript_info[df_transcript_info['variant_information'] == variant_info].iloc[0]
#             selected_patients_ids = [pid.split('_')[0] for pid in selected_variant_data['patient_ids'].split(',')]
#             selected_patients = df_clinical[df_clinical['manifest_patient_id'].isin(selected_patients_ids)]
#             df_clinical["group"] = df_clinical.apply(lambda r: "B" if r['manifest_patient_id'] in selected_patients_ids else "A", axis=1)
#             df_clinical["group_numeric"] = df_clinical["group"].apply(lambda x: 1 if x == "B" else 0)
#             group_A = df_clinical[df_clinical['group'] == 'A']
#             group_B = df_clinical[df_clinical['group'] == 'B']
#             return group_A, group_B, selected_variant_data['variant_information'], selected_variant_data['logrank_p_value']

#         # Handle button clicks
#         if 'variant_clicked' not in st.session_state:
#             st.session_state['variant_clicked'] = None

#         for idx, row in df_transcript_info.iterrows():
#             if st.button(f'Show Plot: {row["variant_information"]}', key=row["variant_information"]):
#                 st.session_state['variant_clicked'] = row['variant_information']
                

#         if st.session_state['variant_clicked']:
#             group_A, group_B, variant_information, logrank_p_value = get_km_plot_data(st.session_state['variant_clicked'])
#             plot_km_curve(group_A, group_B, idx=None, title=variant_information, logrank_p_value=logrank_p_value)
        

      

        # Display the table
        #st.plotly_chart(fig)


        
        
else:
    st.write("Failed to load data. Please check the file path and format.")
    
    
    
    
# def calculate_p_values(df_kmf, df_transcript_info):
#     kmf = KaplanMeierFitter()
#     cph = CoxPHFitter(penalizer=0.1)  # Adding a penalizer

#     # Add new columns to df_transcript_info to store results
#     df_transcript_info['logrank_p_value'] = None
#     df_transcript_info['coxph_p_value'] = None
#     df_transcript_info['concordance_index'] = None
#     df_transcript_info['hazard_ratio'] = None
#     df_transcript_info['variant_information'] = None
#     df_transcript_info['Disease_type Chi-square Statistics'] = None
#     df_transcript_info['Disease_type Chi-square p-value'] = None

#     for idx, row in df_transcript_info.iterrows():
#         selected_patients_ids = [pid.split('_')[0] for pid in row['patient_ids'].split(',')]
#         selected_patients = df_kmf[df_kmf['manifest_patient_id'].isin(selected_patients_ids)]
#         df_transcript_info['variant_information'] = df_transcript_info.apply(
#             lambda row: f"{row['chromosome']}:{row['variant_start_position']}:{row['ref_nucleotide']}>{row['alternative_nucleotide']}",
#             axis=1
#         )

#         df_kmf["group"] = df_kmf.apply(lambda r: "B" if r['manifest_patient_id'] in selected_patients_ids else "A", axis=1)
#         df_kmf["group_numeric"] = df_kmf["group"].apply(lambda x: 1 if x == "B" else 0)
#         group_A = df_kmf[df_kmf['group'] == 'A']
#         group_B = df_kmf[df_kmf['group'] == 'B']
        
#         # Create a contingency table
#         contingency_table = pd.crosstab(df_kmf['group'], df_kmf['disease_type'])
#         # Perform Chi-square test
#         chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
#         df_transcript_info.at[idx, 'Disease_type Chi-square Statistics'] = chi2
#         df_transcript_info.at[idx, 'Disease_type Chi-square p-value'] = p

#         if len(group_A) > 1 and len(group_B) > 1:  # Ensure there are enough data points for analysis
            
            
#             # Perform log-rank test
#             results_logrank = logrank_test(group_A['km_time'], group_B['km_time'], event_observed_A=group_A['km_status'], event_observed_B=group_B['km_status'])
#             logrank_p_value = results_logrank.p_value

#             # Fit Cox Proportional Hazards model
#             df_kmf_clean = df_kmf[['manifest_patient_id', 'project_id', 'group', 'group_numeric', 'km_time', 'km_status']].dropna()

#             # Diagnostic check for variance
#             events = df_kmf_clean['km_status'].astype(bool)

#             try:
#                 cph.fit(df_kmf_clean, duration_col='km_time', event_col='km_status', formula='group_numeric')
                
#                 coxph_p_value = cph.summary.loc['group_numeric', 'p']
#                 hazard_ratio = cph.summary.loc['group_numeric', 'exp(coef)']
#                 concordance_index = cph.concordance_index_

#                 # Store the results in the transcript info DataFrame
#                 df_transcript_info.at[idx, 'logrank_p_value'] = logrank_p_value
#                 df_transcript_info.at[idx, 'coxph_p_value'] = coxph_p_value
#                 df_transcript_info.at[idx, 'concordance_index'] = concordance_index
#                 df_transcript_info.at[idx, 'hazard_ratio'] = hazard_ratio
#             except Exception as e:
#                 print(f"Error fitting CoxPH model for index {idx}: {e}")
#                 df_transcript_info.at[idx, 'logrank_p_value'] = None
#                 df_transcript_info.at[idx, 'coxph_p_value'] = None
#                 df_transcript_info.at[idx, 'concordance_index'] = None
#                 df_transcript_info.at[idx, 'hazard_ratio'] = None
#         else:
#             df_transcript_info.at[idx, 'logrank_p_value'] = None
#             df_transcript_info.at[idx, 'coxph_p_value'] = None
#             df_transcript_info.at[idx, 'concordance_index'] = None
#             df_transcript_info.at[idx, 'hazard_ratio'] = None
    
#     # Move 'variant_information' to the first column
#     cols = ['variant_information'] + [col for col in df_transcript_info if col != 'variant_information']
#     df_transcript_info = df_transcript_info[cols]
#     return df_transcript_info


# def plot_km_curve(group_A, group_B, title):
#     results_logrank = logrank_test(group_A['km_time'], group_B['km_time'], event_observed_A=group_A['km_status'], event_observed_B=group_B['km_status'])
#     logrank_p_value = results_logrank.p_value
#     kmf = KaplanMeierFitter()
#     kmf.fit(group_A['km_time'], event_observed=group_A['km_status'], label=f'Group A[{len(group_A)} patients]')
#     kmf_A_sub = kmf.survival_function_
#     ci_A = kmf.confidence_interval_
#     kmf.fit(group_B['km_time'], event_observed=group_B['km_status'], label=f'Group B[{len(group_B)} patients]')
#     kmf_B_sub = kmf.survival_function_
#     ci_B = kmf.confidence_interval_
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=kmf_A_sub.index, y=kmf_A_sub.iloc[:, 0], mode='lines', name=f'Group A[{len(group_A)} patients]'))
#     fig.add_trace(go.Scatter(x=kmf_B_sub.index, y=kmf_B_sub.iloc[:, 0], mode='lines', name=f'Group B[{len(group_B)} patients]', line=dict(color='orange')))
#     fig.add_trace(go.Scatter(
#         x=list(ci_A.index) + list(ci_A.index[::-1]),
#         y=list(ci_A.iloc[:, 0]) + list(ci_A.iloc[:, 1][::-1]),
#         fill='toself',
#         fillcolor='rgba(31, 119, 180, 0.2)',
#         line=dict(color='rgba(255,255,255,0)'),
#         hoverinfo="skip",
#         showlegend=False
#     ))
#     fig.add_trace(go.Scatter(
#         x=list(ci_B.index) + list(ci_B.index[::-1]),
#         y=list(ci_B.iloc[:, 0]) + list(ci_B.iloc[:, 1][::-1]),
#         fill='toself',
#         fillcolor='rgba(255, 127, 14, 0.2)',
#         line=dict(color='rgba(255,255,255,0)'),
#         hoverinfo="skip",
#         showlegend=False
#     ))
#     fig.update_layout(
#         title={
#             'text': f"{title}<br>Log Rank p-value: {logrank_p_value:.4f}",
#             'y': 0.9,
#             'x': 0.5,
#             'xanchor': 'center',
#             'yanchor': 'top'
#         },
#         xaxis_title='Time (Days)',
#         yaxis_title='Survival Probability'
#     )
#     return(fig)