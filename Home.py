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

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest, DateRange, Dimension, Metric
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


from ga_clientside import inject_ga
inject_ga()


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



# --- Google Analytics Tracking Code (Server-Side) ---
# This uses the Measurement Protocol, which is robust and doesn't rely on browser scripts.
try:
    if "ga" in st.secrets and "measurement_id" in st.secrets["ga"] and "api_secret" in st.secrets["ga"]:
        MID = st.secrets["ga"]["measurement_id"]
        SECRET = st.secrets["ga"]["api_secret"]

        if "ga_cid" not in st.session_state:
            st.session_state["ga_cid"] = str(uuid.uuid4())
        cid = st.session_state["ga_cid"]

        payload = {
            "client_id": cid,
            "events": [{"name": "page_view", "params": {"page_title": "DeepVRegulome Home"}}]
        }
        url = f"https://www.google-analytics.com/mp/collect?measurement_id={MID}&api_secret={SECRET}"
        requests.post(url, json=payload, timeout=2)
except Exception:
    # Silently pass if tracking fails.
    pass


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
st.title("🧬DeepVRegulome: DNABERT-based framework for predicting the functional impact of short genomic variants on the human regulome")
st.subheader("Welcome to the interactive data portal for **DeepVRegulome**, an interactive platform for exploring the functional impact of genomic variants.")

# --- Quick Links Row ---
link_col1, link_col2, link_col3, link_col4 = st.columns(4)
with link_col1:
    st.link_button("📦 PyPI Package", "https://pypi.org/project/deepvregulome/", use_container_width=True)
with link_col2:
    st.link_button("🤗 Models", "https://huggingface.co/duttaprat/DeepVRegulome", use_container_width=True)
with link_col3:
    st.link_button("🚀 Live Demo (Space)", "https://huggingface.co/spaces/duttaprat/DeepVRegulome", use_container_width=True)
with link_col4:
    st.link_button("💻 GitHub", "https://github.com/DavuluriLab/DeepVRegulome", use_container_width=True)

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

```
@article{dutta2025deepvregulome,
  title={DeepVRegulome: DNABERT-based deep-learning framework for predicting the functional impact of short genomic variants on the human regulome},
  author={Dutta, Pratik and Obusan, Matthew and Sathian, Rekha and Chao, Max and Surana, Pallavi and Papineni, Nimisha and Ji, Yanrong and Zhou, Zhihan and Liu, Han and Yurovsky, Alisa and others},
  journal={arXiv preprint arXiv:2511.09026},
  year={2025}
}
```
""")

st.divider()

from community_engagement import render_community_engagement
render_community_engagement()

# # --- Google Analytics Display Section ---
# @st.cache_data(ttl=3600)
# def get_analytics_data():
#     """Fetches and parses visitor data from the Google Analytics Data API."""
#     try:
#         # --- CORRECTED SECRET ACCESS ---
#         # Access the keys using the structure from your secrets file: [ga] and [ga.credentials]
#         creds_dict = st.secrets["ga"]["credentials"]
#         property_id = st.secrets["ga"]["property_id"]
        
#         credentials = service_account.Credentials.from_service_account_info(creds_dict)
#         client = BetaAnalyticsDataClient(credentials=credentials)

#         request = RunReportRequest(
#             property=f"properties/{property_id}",
#             dimensions=[{"name": "country"}],
#             metrics=[{"name": "totalUsers"}],
#             date_ranges=[{"start_date": "2024-01-01", "end_date": "today"}],
#         )
#         response = client.run_report(request)

#         rows = [{'Country': row.dimension_values[0].value, 'Visitors': int(row.metric_values[0].value)} for row in response.rows]
#         if not rows: return 0, 0, pd.DataFrame()

#         df = pd.DataFrame(rows)
#         return df['Visitors'].sum(), df['Country'].nunique(), df.nlargest(5, 'Visitors')
#     except Exception as e:
#         st.error(f"Failed to fetch analytics data. Please ensure secrets are configured correctly. Error: {e}")
#         return 0, 0, pd.DataFrame()

# st.divider()
# st.header("🌎 Community Engagement")

# total_users, total_countries, df_top_countries = get_analytics_data()

# if total_users > 0:
#     col_a, col_b = st.columns([1, 2], gap="large")
#     with col_a:
#         st.metric("Total Unique Viewers", f"{total_users:,}")
#         st.metric("Countries Reached", total_countries)
#     with col_b:
#         fig = px.bar(df_top_countries.sort_values('Visitors', ascending=True), x='Visitors', y='Country', orientation='h', title='Top 5 Viewer Countries', text='Visitors', marker_color='#0072B2')
#         fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), yaxis_title=None)
#         st.plotly_chart(fig, use_container_width=True)
# else:
#     st.info("Analytics data is still collecting. Please check back in 24-48 hours.")





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# # --- Google Analytics Display Section ---
# @st.cache_data(ttl=3600)
# def get_analytics_data():
#     """Fetches visitor data from Google Analytics Data API."""
#     try:
#         # DEBUG: Check what's in secrets
#         st.write("🔍 DEBUG: Checking secrets structure...")
#         st.write("Keys in st.secrets:", list(st.secrets.keys()))

#         if "ga" in st.secrets:
#             ga_keys = list(st.secrets["ga"].keys())
#             st.write(f"✅ Found {len(ga_keys)} keys in st.secrets['ga']:", ga_keys)
            
#             # Check specifically for credential keys
#             credential_keys = [k for k in ga_keys if k.startswith('credentials_')]
#             if credential_keys:
#                 st.write(f"✅ Found {len(credential_keys)} credential keys:", credential_keys)
#             else:
#                 st.write("❌ No credential keys found (should start with 'credentials_')")
#                 st.stop()
#         else:
#             st.error("❌ 'ga' section not found in secrets!")
#             return 0, 0, pd.DataFrame()
        
#         # Build credentials dict from flat structure
#         creds_dict = {
#             "type": st.secrets["ga"].get("credentials_type", "service_account"),
#             "project_id": st.secrets["ga"].get("credentials_project_id"),
#             "private_key_id": st.secrets["ga"].get("credentials_private_key_id"),
#             "private_key": st.secrets["ga"].get("credentials_private_key"),
#             "client_email": st.secrets["ga"].get("credentials_client_email"),
#             "client_id": st.secrets["ga"].get("credentials_client_id"),
#             "auth_uri": st.secrets["ga"].get("credentials_auth_uri", "https://accounts.google.com/o/oauth2/auth"),
#             "token_uri": st.secrets["ga"].get("credentials_token_uri", "https://oauth2.googleapis.com/token"),
#             "auth_provider_x509_cert_url": st.secrets["ga"].get("credentials_auth_provider_x509_cert_url", "https://www.googleapis.com/oauth2/v1/certs"),
#             "client_x509_cert_url": st.secrets["ga"].get("credentials_client_x509_cert_url"),
#             "universe_domain": st.secrets["ga"].get("credentials_universe_domain", "googleapis.com")
#         }
        
#         # Debug: Check if credentials were loaded
#         st.write("🔍 Checking loaded credentials:")
#         st.write(f"- project_id: {'✅ Found' if creds_dict['project_id'] else '❌ Missing'}")
#         st.write(f"- private_key: {'✅ Found' if creds_dict['private_key'] else '❌ Missing'}")
#         st.write(f"- client_email: {'✅ Found' if creds_dict['client_email'] else '❌ Missing'}")
        
#         # Check if we have the required credentials
#         if not creds_dict["project_id"] or not creds_dict["private_key"]:
#             st.error("❌ Missing required credentials (project_id or private_key)")
#             st.info("Please add all credential fields to your secrets!")
#             return 0, 0, pd.DataFrame()
        
#         property_id = st.secrets["ga"].get("property_id")
#         if not property_id:
#             st.error("❌ property_id not found in secrets")
#             return 0, 0, pd.DataFrame()
            
#         st.write(f"✅ property_id: {property_id}")
        
#         # Create credentials
#         st.write("🔄 Creating service account credentials...")
#         credentials = service_account.Credentials.from_service_account_info(creds_dict)
        
#         st.write("🔄 Initializing Analytics client...")
#         client = BetaAnalyticsDataClient(credentials=credentials)

#         # Request for country-level data
#         st.write("🔄 Requesting analytics data...")
#         request = RunReportRequest(
#             property=f"properties/{property_id}",
#             dimensions=[Dimension(name="country")],
#             metrics=[Metric(name="totalUsers")],
#             date_ranges=[DateRange(start_date="2024-11-01", end_date="today")],
#         )
#         response = client.run_report(request)

#         # Parse response
#         st.write(f"✅ Received {len(response.rows)} rows from Google Analytics")
        
#         rows = []
#         for row in response.rows:
#             country = row.dimension_values[0].value
#             users = int(row.metric_values[0].value)
#             rows.append({'Country': country, 'Visitors': users})
        
#         if not rows:
#             st.warning("No analytics data found for the date range")
#             return 0, 0, pd.DataFrame()

#         df = pd.DataFrame(rows)
#         total_visitors = df['Visitors'].sum()
#         total_countries = df['Country'].nunique()
#         top_countries = df.nlargest(5, 'Visitors')
        
#         st.success(f"✅ Successfully loaded analytics: {total_visitors} visitors from {total_countries} countries!")
        
#         return total_visitors, total_countries, top_countries
        
#     except Exception as e:
#         st.error(f"❌ Analytics error: {str(e)}")
#         import traceback
#         st.code(traceback.format_exc())
#         return 0, 0, pd.DataFrame()

# st.divider()

# # --- Google Analytics Display Section ---
# st.header("🌎 Community Engagement")

# total_users, total_countries, df_top_countries = get_analytics_data()

# if total_users > 0:
#     col_a, col_b = st.columns([1, 2], gap="large")
#     with col_a:
#         st.metric("Total Unique Visitors", f"{total_users:,}")
#         st.metric("Countries Reached", total_countries)
#     with col_b:
#         if not df_top_countries.empty:
#             fig = px.bar(
#                 df_top_countries.sort_values('Visitors', ascending=True),
#                 x='Visitors',
#                 y='Country',
#                 orientation='h',
#                 title='Top 5 Visitor Countries',
#                 text='Visitors',
#                 color_discrete_sequence=['#0072B2']
#             )
#             fig.update_layout(
#                 showlegend=False,
#                 margin=dict(l=10, r=10, t=40, b=10),
#                 yaxis_title=None,
#                 xaxis_title="Number of Visitors"
#             )
#             fig.update_traces(textposition='outside')
#             st.plotly_chart(fig, use_container_width=True)
# else:
#     st.info("Analytics data is still collecting. Check back soon!")
