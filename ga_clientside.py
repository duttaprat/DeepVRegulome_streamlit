"""
ga_clientside.py
----------------
Client-side Google Analytics (gtag.js) injection for Streamlit.

WHY THIS EXISTS
The portal currently tracks page views with the server-side Measurement
Protocol (send_page_view() in Home.py). That works for counting users
(903 recorded) but country is always "(not set)" because the request
comes from Streamlit's server IP, not the visitor's browser.

This module injects the real browser gtag.js so GA4 sees the visitor's
own IP and resolves Country / City / Device correctly. It does NOT
remove the server-side code; both can run, GA4 de-duplicates by the
gtag-generated client_id once gtag is present.

KEY STREAMLIT GOTCHA
st.markdown(..., unsafe_allow_html=True) does NOT execute <script>.
components.html() DOES execute scripts (it renders a real iframe whose
requests still carry the visitor's IP). Height must be > 0 or the
component can be skipped; we use height=1.

Usage in Home.py (call ONCE, right after st.set_page_config):
    from ga_clientside import inject_ga
    inject_ga()
"""

import streamlit as st
import streamlit.components.v1 as components


def inject_ga():
    """Inject browser-side gtag.js. Reads measurement_id from secrets."""
    try:
        mid = st.secrets["ga"]["measurement_id"]
    except Exception:
        # No measurement id configured -> do nothing, never crash the app.
        return

    # Full standard gtag.js snippet. Runs in the visitor's browser, so
    # GA4 receives the real client IP and geo-resolves Country/City.
    snippet = f"""
        <script async src="https://www.googletagmanager.com/gtag/js?id={mid}"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){{dataLayer.push(arguments);}}
          gtag('js', new Date());
          gtag('config', '{mid}', {{
            'page_title': 'DeepVRegulome',
            'page_path': '/'
          }});
        </script>
    """
    # height=1 (not 0): a 0-height component is sometimes not mounted,
    # which would silently skip the script. 1px is invisible in practice.
    components.html(snippet, height=1)
