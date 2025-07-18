import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest
from google.oauth2 import service_account
import streamlit.components.v1 as components
import uuid
import requests

# --- Page Configuration (Should be the first command) ---
st.set_page_config(
    layout="wide",
    page_title="DeepVRegulome",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)



# Pull from secrets
MID    = st.secrets["ga"]["measurement_id"]
SECRET = st.secrets["ga"]["api_secret"]

def send_page_view():
    # 1) Create or reuse a per-session client ID
    if "ga_cid" not in st.session_state:
        st.session_state["ga_cid"] = str(uuid.uuid4())
    cid = st.session_state["ga_cid"]

    # 2) Build the payload for a standard page_view
    payload = {
        "client_id": cid,
        "events": [
            {
                "name": "page_view",
                "params": {
                    "page_title": "DeepVRegulome",
                    "page_location": "https://deepvregulome.streamlit.app/",
                    "engagement_time_msec": 1
                }
            }
        ]
    }

    # 3) Send to GA4
    url = (
        "https://www.google-analytics.com/mp/collect"
        f"?measurement_id={MID}&api_secret={SECRET}"
    )
    try:
        # short timeout so app startup isn’t delayed
        requests.post(url, json=payload, timeout=2)
    except:
        pass

# Fire it once at startup
send_page_view()


# # --- Google Analytics Tracking Code ---
# # This injects the script into the app's HTML head.
# st.markdown(
#     """
#     <!-- Google tag (gtag.js) -->
#     <script async src="https://www.googletagmanager.com/gtag/js?id=G-X7CEN7XS7F"></script>
#     <script>
#       window.dataLayer = window.dataLayer || [];
#       function gtag(){dataLayer.push(arguments);}
#       gtag('js', new Date());

#       gtag('config', 'G-X7CEN7XS7F');
#     </script>
#     """,
#     unsafe_allow_html=True
# )

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

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest
from google.oauth2 import service_account

# 1) Caching the GA4 query for 1h
@st.cache_data(ttl=3600)
def get_global_usage():
    creds_dict  = st.secrets["google_credentials"]
    property_id = st.secrets["google_property_id"]
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    client      = BetaAnalyticsDataClient(credentials=credentials)

    request = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[{"name":"country"}],
        metrics=[{"name":"totalUsers"}],
        date_ranges=[{"start_date":"2024-01-01","end_date":"today"}],
    )
    resp = client.run_report(request)

    rows = [
        {"country": r.dimension_values[0].value,
         "visits":  int(r.metric_values[0].value)}
        for r in resp.rows
    ]
    if not rows:
        return 0, 0, pd.DataFrame(columns=["country","visits"])

    df = pd.DataFrame(rows)
    total_users     = int(df["visits"].sum())
    total_countries = df["country"].nunique()
    top5            = df.nlargest(5, "visits").reset_index(drop=True)
    return total_users, total_countries, top5

# 2) Pull your numbers
total_users, total_countries, df_top5 = get_global_usage()

# 3) Render centered header + subheader
st.markdown(
    "<h2 style='text-align:center;'>🌎 Curious how far this tool has reached?</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:gray;'>A live snapshot of total visitors & global reach</p>",
    unsafe_allow_html=True
)

# 4) Three-column layout
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.metric("👥 Total Unique Visitors", f"{total_users:,}")

with c2:
    st.metric("🌍 Countries Represented", f"{total_countries}")

with c3:
    # mini horizontal bar chart
    fig = px.bar(
        df_top5.sort_values("visits", ascending=True),
        x="visits",
        y="country",
        orientation="h",
        text="visits",
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0,r=0,t=30,b=0),
        height=240,
        yaxis_title=None,
        font=dict(size=12),
    )
    st.plotly_chart(fig, use_container_width=True)


