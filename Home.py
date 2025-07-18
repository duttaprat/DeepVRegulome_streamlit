import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="DeepVRegulome",
    page_icon="🧬",
    initial_sidebar_state="expanded" # Ensure the sidebar is open by default
)

st.markdown("""
<style>
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)


# Title of the app
st.title("🧬DeepVRegulome: DNABERT-based deep-learning framework for predicting the functional impact of short genomic variants on the human regulome")
st.subheader("Welcome to the interactive data portal for **DeepVRegulome**, an interactive platform for exploring the functional impact of genomic variants.")


st.divider()
# --- Introduction and User Guidance ---
#col1, col2 = st.columns(2, gap="large")
col1, col2 = st.columns([4, 5], gap="large")

with col1:
    st.markdown("""
    ### Your Gateway to Genomic Discovery
    """)
    st.markdown("""
    This portal is the official interactive companion to our under review *Nature Methods* publication on **DeepVRegulome**. 

     It is designed to help researchers explore our models, data, and key findings in an intuitive way.
    """)
    # This is the most important part: A clear call to action.
    st.success("To begin, please select the **'🏠 Overview'** page from the sidebar on the left.", icon="👈")
    
    st.markdown("""
    **Navigate through the application using the sidebar on the left to:**
    - **🏠 Overview:** View high-level statistics and patient cohort data.
    - **Model Performance:** Evaluate the accuracy and predictive power of our underlying models.
    - **Browse All Variants:** Interactively filter and explore the complete dataset of predicted functional variants and their associated survival plots.
    - **Motif Validation:** See how our models learned biologically meaningful transcription factor binding motifs.
    """)

    # st.header("Abstract")
    # st.info("""
    # Whole-genome sequencing (WGS) has revealed numerous non-coding short variants whose functional impacts remain 
    # poorly understood. Despite recent advances in deep-learning genomic approaches, accurately predicting and 
    # prioritizing clinically relevant mutations in gene regulatory regions remains a major challenge. Here we 
    # introduce DeepVRegulome, a deep-learning method for prediction and interpretation of functionally disruptive 
    # variants in the human regulome, which combines 700 DNABERT fine-tuned models, trained on vast amounts of ENCODE 
    # gene regulatory regions, with variant scoring, motif analysis, attention-based visualization, and survival 
    # analysis. We showcase its application on TCGA glioblastoma WGS dataset in prioritizing survival-associated 
    # mutations and regulatory regions. The analysis identified 572 splice-disrupting and 9,837 transcription-factor 
    # binding site altering mutations occurring in greater than 10% of glioblastoma samples. Survival analysis 
    # linked 1352 mutations and 563 disrupted regulatory regions to patient outcomes, enabling stratification via 
    # non-coding mutation signatures. All the code, fine-tuned models, and an interactive data portal are publicly 
    # available.
    # """)


with col2:
    st.header("Framework Architecture")
    try:
        # Make sure you have an image of Figure 1 from your paper in an 'assets' folder
        image = Image.open("assets/Figure1_architecture.PNG")
        st.image(image, caption="Architecture of the DeepVRegulome computational framework.", use_column_width=True)
    except FileNotFoundError:
        st.error("Architecture image not found. Please add 'Figure1_architecture.png' to an 'assets' folder in your repository.")

st.divider()


# --- Citation Information ---
st.header("How to Cite")
st.markdown("""
If you use the data or models from this portal in your research, please cite our publication:

**Dutta, P. et al. DeepVRegulome: DNABERT-based deep-learning framework for predicting the functional impact of short genomic variants on the human regulome. *Nature Methods* (Under Revision).**
""")