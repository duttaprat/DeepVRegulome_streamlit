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

import streamlit as st
import uuid, os, csv, time, requests
import pandas as pd
import plotly.express as px

# ─── 1) Initialization ────────────────────────────────────────
LOG = "visit_log.csv"
if not os.path.exists(LOG):
    with open(LOG, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ts","visitor_id","country"])

# ─── 2) On each load, log a visit ─────────────────────────────
# 2a) Get or create a visitor ID in session_state
if "visitor_id" not in st.session_state:
    st.session_state.visitor_id = str(uuid.uuid4())

vid = st.session_state.visitor_id

# 2b) Lookup country from IP (free service; please read their TOS!)
# Note: On some hosts you won’t get a real IP; you may need st.experimental_get_query_params or headers
ip = st.experimental_get_query_params().get("client_ip", [None])[0]
if ip is None:
    ip = requests.get("https://api.ipify.org").text  # fallback: your server’s IP

try:
    country = requests.get(f"https://ipapi.co/{ip}/country_name/").text
    if not country or country.startswith("<"):
        country = "Unknown"
except Exception:
    country = "Unknown"

# 2c) Append to CSV
with open(LOG, "a") as f:
    writer = csv.writer(f)
    writer.writerow([int(time.time()), vid, country])

# ─── 3) Compute usage metrics ─────────────────────────────────
df = pd.read_csv(LOG)
unique_users     = df["visitor_id"].nunique()
unique_countries = df["country"].nunique()
country_counts   = df["country"].value_counts().reset_index()
country_counts.columns = ["country","visits"]

# ─── 4) Display on the home page ──────────────────────────────
st.markdown("<h2 style='text-align:center;'>🌎 Curious how far this tool has reached? Here's a snapshot of our live global usage:</h1>", unsafe_allow_html=True)
st.markdown("### Welcome—glad you’re here!  \nBelow are some live usage stats:")

# ─── Usage Metrics ─────────────────────────────
c1, c2, c3 = st.columns(3, gap="large")

c1.metric("👥 Unique Visitors", unique_users)
c2.metric("🌍 Countries Represented", unique_countries)
c3.metric("📈 Top Country", country_counts.iloc[0]["country"] if not country_counts.empty else "N/A")

# ─── Top 5 Countries Chart ─────────────────────
fig = px.bar(
    country_counts.head(5), 
    x="country", 
    y="visits",
    title="Top 5 Countries by Visits",
    text="visits"
)
st.plotly_chart(fig, use_container_width=True)
