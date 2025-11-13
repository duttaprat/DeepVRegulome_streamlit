import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest, 
    DateRange, 
    Dimension, 
    Metric,
    OrderBy
)
from google.oauth2 import service_account
import streamlit.components.v1 as components
import uuid
import requests
from datetime import datetime, timedelta
import json

# --- Page Configuration (Should be the first command) ---
st.set_page_config(
    layout="wide",
    page_title="DeepVRegulome",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- Google Analytics Tracking (Measurement Protocol) ---
def send_page_view(page_title="DeepVRegulome Home"):
    """Send page view event to Google Analytics"""
    try:
        if "ga" in st.secrets and "measurement_id" in st.secrets["ga"] and "api_secret" in st.secrets["ga"]:
            MID = st.secrets["ga"]["measurement_id"]
            SECRET = st.secrets["ga"]["api_secret"]
            
            # Create or reuse a per-session client ID
            if "ga_cid" not in st.session_state:
                st.session_state["ga_cid"] = str(uuid.uuid4())
            
            cid = st.session_state["ga_cid"]
            
            # Get user's approximate location (this is optional)
            user_agent = "Streamlit App"
            
            payload = {
                "client_id": cid,
                "events": [{
                    "name": "page_view", 
                    "params": {
                        "page_title": page_title,
                        "page_location": "https://deepvregulome.streamlit.app",
                        "engagement_time_msec": "100"
                    }
                }]
            }
            
            url = f"https://www.google-analytics.com/mp/collect?measurement_id={MID}&api_secret={SECRET}"
            response = requests.post(url, json=payload, timeout=2)
            return response.status_code == 204
    except Exception as e:
        st.error(f"Analytics tracking error: {e}")
        return False

# --- Google Analytics Data API Functions ---
def get_analytics_credentials():
    """Create service account credentials from secrets"""
    try:
        # Build the credentials dictionary from flattened structure
        creds_dict = {
            "type": st.secrets["ga"]["credentials_type"],
            "project_id": st.secrets["ga"]["credentials_project_id"],
            "private_key_id": st.secrets["ga"]["credentials_private_key_id"],
            "private_key": st.secrets["ga"]["credentials_private_key"],
            "client_email": st.secrets["ga"]["credentials_client_email"],
            "client_id": st.secrets["ga"]["credentials_client_id"],
            "auth_uri": st.secrets["ga"]["credentials_auth_uri"],
            "token_uri": st.secrets["ga"]["credentials_token_uri"],
            "auth_provider_x509_cert_url": st.secrets["ga"]["credentials_auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["ga"]["credentials_client_x509_cert_url"],
            "universe_domain": st.secrets["ga"]["credentials_universe_domain"]
        }
        
        # Create credentials object
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        return credentials
    except Exception as e:
        st.error(f"Error creating credentials: {e}")
        return None

def get_analytics_data():
    """Fetch analytics data from Google Analytics Data API"""
    try:
        credentials = get_analytics_credentials()
        if not credentials:
            return None
        
        # Initialize the client
        client = BetaAnalyticsDataClient(credentials=credentials)
        
        # Get property ID from secrets
        property_id = f"properties/{st.secrets['ga']['property_id']}"
        
        # Create the request for the last 30 days
        request = RunReportRequest(
            property=property_id,
            dimensions=[
                Dimension(name="country"),
                Dimension(name="city"),
                Dimension(name="date")
            ],
            metrics=[
                Metric(name="activeUsers"),
                Metric(name="sessions"),
                Metric(name="screenPageViews")
            ],
            date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
            order_bys=[
                OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="activeUsers"))
            ]
        )
        
        # Run the report
        response = client.run_report(request)
        
        # Process the response
        data = []
        for row in response.rows:
            data.append({
                'Country': row.dimension_values[0].value,
                'City': row.dimension_values[1].value,
                'Date': row.dimension_values[2].value,
                'Active Users': int(row.metric_values[0].value),
                'Sessions': int(row.metric_values[1].value),
                'Page Views': int(row.metric_values[2].value)
            })
        
        df = pd.DataFrame(data)
        
        # Calculate totals
        total_unique_visitors = df['Active Users'].sum()
        total_page_views = df['Page Views'].sum()
        
        # Get unique countries and cities
        unique_countries = df['Country'].nunique()
        unique_cities = df['City'].nunique()
        
        return {
            'dataframe': df,
            'total_unique_visitors': total_unique_visitors,
            'total_page_views': total_page_views,
            'unique_countries': unique_countries,
            'unique_cities': unique_cities,
            'top_countries': df.groupby('Country')['Active Users'].sum().sort_values(ascending=False).head(10)
        }
        
    except Exception as e:
        st.error(f"Error fetching analytics data: {e}")
        return None

# --- CSS for Styling ---
st.markdown("""
<style>
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1.2em;
        color: #666;
        margin-top: 5px;
    }
    .citation-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-left: 4px solid #1f77b4;
        margin: 20px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Send page view
send_page_view()

# --- Main Content ---
st.title("🧬 DeepVRegulome")
st.markdown("### Deep Learning Framework for Regulatory Variant Interpretation")

# --- Add citation section ---
st.markdown("---")
st.markdown("### 📚 How to Cite")

with st.container():
    st.markdown("""
    <div class="citation-box">
    <strong>If you use DeepVRegulome in your research, please cite our paper:</strong><br><br>
    
    <em>DeepVRegulome: A deep learning framework for interpreting regulatory variants in cancer</em><br>
    Authors: [Your author list here]<br>
    Journal: [Journal name, year]<br>
    DOI: [Add DOI when available]<br><br>
    
    <strong>BibTeX:</strong>
    <pre>
@article{deepvregulome2024,
  title={DeepVRegulome: A deep learning framework for interpreting regulatory variants in cancer},
  author={[Author names]},
  journal={[Journal]},
  year={2024},
  doi={[DOI]}
}
    </pre>
    </div>
    """, unsafe_allow_html=True)
    
    # Add copy button for BibTeX
    bibtex_text = """@article{deepvregulome2024,
  title={DeepVRegulome: A deep learning framework for interpreting regulatory variants in cancer},
  author={[Author names]},
  journal={[Journal]},
  year={2024},
  doi={[DOI]}
}"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 Download BibTeX",
            data=bibtex_text,
            file_name="deepvregulome.bib",
            mime="text/plain"
        )

# --- Analytics Dashboard ---
st.markdown("---")
st.markdown("### 🌎 Community Engagement")

# Fetch analytics data
analytics_data = get_analytics_data()

if analytics_data:
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{analytics_data['total_unique_visitors']:,}</div>
            <div class="metric-label">Unique Visitors (30 days)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{analytics_data['total_page_views']:,}</div>
            <div class="metric-label">Total Page Views</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{analytics_data['unique_countries']:,}</div>
            <div class="metric-label">Countries Reached</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{analytics_data['unique_cities']:,}</div>
            <div class="metric-label">Cities Reached</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show geographic distribution
    st.markdown("#### 🗺️ Geographic Distribution of Users")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Top Countries", "Time Series", "Detailed Data"])
    
    with tab1:
        # Bar chart of top countries
        fig = px.bar(
            x=analytics_data['top_countries'].values,
            y=analytics_data['top_countries'].index,
            orientation='h',
            labels={'x': 'Number of Unique Visitors', 'y': 'Country'},
            title='Top 10 Countries by Unique Visitors'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Time series of daily visitors
        df_time = analytics_data['dataframe'].groupby('Date').agg({
            'Active Users': 'sum',
            'Page Views': 'sum'
        }).reset_index()
        df_time['Date'] = pd.to_datetime(df_time['Date'], format='%Y%m%d')
        df_time = df_time.sort_values('Date')
        
        fig = px.line(
            df_time, 
            x='Date', 
            y=['Active Users', 'Page Views'],
            title='Daily Visitor Trends',
            labels={'value': 'Count', 'variable': 'Metric'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Display raw data with filtering
        st.markdown("##### 📊 Detailed Analytics Data")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            selected_countries = st.multiselect(
                "Filter by Country",
                options=analytics_data['dataframe']['Country'].unique(),
                default=[]
            )
        
        # Apply filters
        filtered_df = analytics_data['dataframe']
        if selected_countries:
            filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
        
        # Show summary stats
        st.markdown(f"**Showing {len(filtered_df)} records**")
        
        # Display dataframe
        st.dataframe(
            filtered_df.sort_values('Active Users', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Download button for data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Analytics Data (CSV)",
            data=csv,
            file_name=f"deepvregulome_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
else:
    st.info("📊 Analytics data is currently unavailable. Please check your Google Analytics configuration.")
    
    # Show configuration help
    with st.expander("🔧 Configuration Help"):
        st.markdown("""
        **To enable analytics tracking, ensure:**
        1. Google Analytics 4 property is properly set up
        2. Service account has the necessary permissions
        3. Private key in secrets is properly formatted
        4. Property ID matches your GA4 property
        
        **Required permissions for service account:**
        - Viewer role on the Google Analytics property
        - Analytics Data API enabled in Google Cloud Console
        """)

# --- About Section ---
st.markdown("---")
st.markdown("### About DeepVRegulome")
st.markdown("""
DeepVRegulome is a comprehensive deep learning framework designed to interpret regulatory variants 
in cancer genomics. Our tool integrates multiple data modalities to provide accurate predictions 
of variant impact on gene regulation and cancer susceptibility.

**Key Features:**
- 🧬 Deep learning-based variant interpretation
- 📊 Multi-modal data integration
- 🎯 Cancer-specific regulatory predictions
- 🔬 Validated on multiple cancer cohorts
""")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    DeepVRegulome v1.0 | © 2024 | Stony Brook University
</div>
""", unsafe_allow_html=True)
