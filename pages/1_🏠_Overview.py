import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="DeepVRegulome",
    page_icon="🧬"
)


# --- Header ---
st.title("🏠 DeepVRegulome Overview")
st.subheader("A DNABERT-based framework for predicting the functional impact of genomic variants on the human regulome")
st.divider()






# --- Patient Cohort Sunburst Chart ---
st.header("GBM Patient Cohort Overview")

try:
    ######
    # Calculate the required height to center the content
    total_height = 600  # Example total height of the column
    content_height = 300  # Approximate height of the content
    top_padding = (total_height - content_height) // 2

    col1, col2 = st.columns([6, 4])



    with col1:
        try:
            # --- Summary Statistics (Recreating Table 1) ---
            st.markdown("This table summarizes the total number of variants analyzed and the subset predicted as functionally disruptive by DeepVRegulome, focusing on those present in >10% of the GBM patient cohort.")

            # Data for the summary table - directly from your paper's Table 1
            summary_data = {
                "Fine-tuned Models": ["Splice Sites", "Splice Sites", "Splice Sites", "Splice Sites", "ChIP-seq Models", "ChIP-seq Models", "ChIP-seq Models", "ChIP-seq Models"],
                "Subtype": ["Acceptor", "Acceptor", "Donor", "Donor", "Histone Markers", "Histone Markers", "TFs", "TFs"],
                "Variant Type": ["SNVs", "Indels", "SNVs", "Indels", "SNVs", "Indels", "SNVs", "Indels"],
                "Total Variants": [19968, 34718, 23656, 20171, 127297, 254566, 932133, 2130274],
                "Total Regions": [14743, 19897, 15849, 13491, 57013, 87949, 358414, 533433],
                "Predicted Functional Variants (>10% Samples)": ["299 (1)", "1,822 (437)", "673 (4)", "504 (130)", "387 (2)", "536 (150)", "19,709 (1,087)", "35,867 (8,598)"],
                "Affected Regions (>10% Samples)": ["265 (3)", "1,375 (408)", "545 (4)", "423 (122)", "311 (5)", "373 (129)", "16,566 (322)", "30,411 (7,297)"]
            }
            summary_df = pd.DataFrame(summary_data)

            # Display the dataframe with styling
            st.dataframe(summary_df, use_container_width=True)




        except Exception as e:
            st.error(f"Failed to load sunburst chart: {str(e)}")



    with col2:
        try:
            clinical_file_path = "data/Brain/patient_clinical_updated.tsv"
            df_clinical = pd.read_csv(clinical_file_path, sep='\t')
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

except FileNotFoundError:
    st.error("Could not find the clinical data file. Please ensure `data/Brain/patient_clinical_updated.tsv` is in your repository.")
except Exception as e:
    st.error(f"An error occurred while loading the patient cohort chart: {e}")

