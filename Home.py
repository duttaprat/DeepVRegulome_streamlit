import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest
from google.oauth2 import service_account
import streamlit.components.v1 as components

# --- Page Configuration (Should be the first command) ---
st.set_page_config(
    layout="wide",
    page_title="DeepVRegulome",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- Google Analytics Tracking Code ---
# This injects the script into the app's HTML head.
components.html(
    """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-X7CEN7XS7F"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());

      gtag('config', 'G-X7CEN7XS7F');
</script>
    """,
    height=0,
    width=0,
)

# --- CSS for Vertical Alignment ---
st.markdown("""
<style>
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Main Page Content ---
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
    - **🔬 Model Performance:** Evaluate the accuracy and predictive power of our underlying models.
    - **📊 Browse All Variants:** Interactively explore the full dataset, view motif validation results, and generate survival plots on-demand.
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

st.divider()

# --- Google Analytics Display Section ---
@st.cache_data(ttl=3600) # Cache the data for 1 hour
def get_analytics_data():
    """Fetches and parses visitor data from the Google Analytics Data API."""
    try:
        creds_dict = st.secrets["google_credentials"]
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        client = BetaAnalyticsDataClient(credentials=credentials)
        property_id = st.secrets["google_property_id"]

        request = RunReportRequest(
            property=f"properties/{property_id}",
            dimensions=[{"name": "country"}],
            metrics=[{"name": "totalUsers"}],
            date_ranges=[{"start_date": "2024-01-01", "end_date": "today"}],
        )
        response = client.run_report(request)

        rows = [{'Country': row.dimension_values[0].value, 'Visitors': int(row.metric_values[0].value)} for row in response.rows]
        if not rows: return 0, 0, pd.DataFrame()

        df = pd.DataFrame(rows)
        return df['Visitors'].sum(), df['Country'].nunique(), df.nlargest(5, 'Visitors')
    except Exception as e:
        st.error(f"Failed to fetch analytics data. Please ensure secrets are configured correctly. Error: {e}")
        return 0, 0, pd.DataFrame()



st.header("🌎 Community Engagement")

total_users, total_countries, df_top_countries = get_analytics_data()

if total_users > 0:
    col_a, col_b = st.columns([1, 2], gap="large")
    with col_a:
        st.metric("Total Unique Viewers", f"{total_users:,}")
        st.metric("Countries Reached", total_countries)
        st.caption("Live data reflects viewership since launch.")
    with col_b:
        fig = px.bar(
            df_top_countries.sort_values('Visitors', ascending=True),
            x='Visitors', y='Country', orientation='h', title='Top 5 Viewer Countries',
            text='Visitors', marker_color='#0072B2'
        )
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Analytics data is still collecting. Please check the Google Analytics 'Realtime' report to verify setup.")

