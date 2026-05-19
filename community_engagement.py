"""
community_engagement.py
-----------------------
Community Engagement panel for the DeepVRegulome portal.

Replaces the old get_analytics_data() that crashed on a missing
st.secrets["ga"]["credentials"] key.

LAYOUT (final):
  Row 1 (3 tiles): PyPI installs (total) | Portal users | Portal page views
  Row 2 (2 tiles): Hugging Face downloads (30d) | Citations
  Then: DNABERT-2 foundation-model lineage callout.

NOTES / HONEST CONSTRAINTS:
  - PyPI total is a true cumulative count (pypistats serves the full
    daily series; deepvregulome is younger than the retention window).
  - Hugging Face exposes ONLY a rolling 30-day download count for this
    model (the API's all-time field is null for it). It is labelled
    honestly as 30-day; there is no reliable way to get an HF total.
  - Portal users / page views come live from the GA4 Data API and need a
    service-account block in secrets ([ga.credentials] + property_id).
    Until that is added, those two tiles show "—". Everything else still
    works with zero credentials.
  - CITATIONS is a manual constant. Google Scholar blocks automated
    fetching, so bump this by hand when the count changes.
  - No country chart, no GitHub tile, no copy-summary expander
    (intentionally removed).

Usage in Home.py:
    from community_engagement import render_community_engagement
    render_community_engagement()
"""

import datetime
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PYPI_PACKAGE = "deepvregulome"          # https://pypi.org/project/deepvregulome/
HF_MODEL = "duttaprat/DeepVRegulome"
HF_LINEAGE = "zhihan1996/DNABERT-2-117M"  # co-authored foundation model

# Manual count: Google Scholar blocks automated fetching. Update by hand.
CITATIONS = 1

REQUEST_TIMEOUT = 6


# ---------------------------------------------------------------------------
# DATA FETCHERS  (cached; each fails soft)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pypi_total(package: str) -> dict:
    """True cumulative install count via pypistats `overall` series."""
    out = {"ok": False, "total": 0, "version": None,
           "first_date": None, "last_date": None}
    try:
        ro = requests.get(
            f"https://pypistats.org/api/packages/{package}/overall",
            params={"mirrors": "false"},
            timeout=REQUEST_TIMEOUT,
        )
        if ro.status_code == 200:
            series = ro.json().get("data", [])
            if series:
                out["total"] = sum(x["downloads"] for x in series)
                dates = sorted(x["date"] for x in series)
                out["first_date"], out["last_date"] = dates[0], dates[-1]
                out["ok"] = True
        rv = requests.get(
            f"https://pypi.org/pypi/{package}/json", timeout=REQUEST_TIMEOUT
        )
        if rv.status_code == 200:
            out["version"] = rv.json()["info"]["version"]
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hf_downloads(model_id: str) -> dict:
    """Rolling 30-day download count for one HF model (no auth)."""
    out = {"ok": False, "downloads": 0}
    try:
        r = requests.get(
            f"https://huggingface.co/api/models/{model_id}",
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            out.update(ok=True,
                       downloads=r.json().get("downloads", 0) or 0)
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ga4_totals():
    """
    Live portal users + page views from the GA4 Data API.

    Returns None if the service-account block is absent in secrets, so
    the tiles degrade to "—" instead of erroring. Never raises.
    """
    try:
        if "ga" not in st.secrets:
            return None
        ga = st.secrets["ga"]
        if "credentials" not in ga or "property_id" not in ga:
            return None

        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        from google.analytics.data_v1beta.types import (
            RunReportRequest, DateRange, Metric,
        )
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_info(
            dict(ga["credentials"])
        )
        client = BetaAnalyticsDataClient(credentials=creds)
        req = RunReportRequest(
            property=f"properties/{ga['property_id']}",
            metrics=[Metric(name="totalUsers"),
                     Metric(name="screenPageViews")],
            date_ranges=[DateRange(start_date="2024-01-01",
                                   end_date="today")],
        )
        resp = client.run_report(req)
        if not resp.rows:
            return None
        return {
            "ok": True,
            "total_users": int(resp.rows[0].metric_values[0].value),
            "page_views": int(resp.rows[0].metric_values[1].value),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RENDERER
# ---------------------------------------------------------------------------
def render_community_engagement():
    st.divider()
    st.header("🌎 Community Engagement & Adoption")
    st.caption(
        "Live DeepVRegulome adoption across PyPI, Hugging Face, and the "
        f"interactive portal. Refreshed hourly · "
        f"{datetime.date.today().isoformat()}."
    )

    pypi = fetch_pypi_total(PYPI_PACKAGE)
    hf = fetch_hf_downloads(HF_MODEL)
    ga = fetch_ga4_totals()

    # ---- Row 1: PyPI total | Portal users | Portal page views ----
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        ht = f"Package: {PYPI_PACKAGE}"
        if pypi.get("version"):
            ht += f" · v{pypi['version']}"
        if pypi.get("first_date"):
            ht += f" · since {pypi['first_date']}"
        st.metric(
            "PyPI installs (total)",
            f"{pypi['total']:,}" if pypi["ok"] and pypi["total"] else "—",
            help=ht,
        )
    with r1c2:
        st.metric(
            "Portal users (total)",
            f"{ga['total_users']:,}" if ga and ga.get("ok") else "—",
            help="Unique users (Google Analytics, all time)",
        )
    with r1c3:
        st.metric(
            "Portal page views (total)",
            f"{ga['page_views']:,}" if ga and ga.get("ok") else "—",
            help="Total page views (Google Analytics, all time)",
        )

    # ---- Row 2: HF downloads (30d) | Citations ----
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.metric(
            "Hugging Face downloads (30 days)",
            f"{hf['downloads']:,}" if hf["ok"] else "—",
            help=f"Model: {HF_MODEL} · Hugging Face reports a rolling "
                 "30-day count for models (no all-time figure available).",
        )
    with r2c2:
        st.metric(
            "Citations",
            f"{CITATIONS:,}",
            help="Manually maintained; Google Scholar blocks automated "
                 "fetching.",
        )

    if pypi["ok"] and pypi.get("first_date"):
        st.caption(
            f"PyPI cumulative total spans {pypi['first_date']} → "
            f"{pypi['last_date']} (the package's full lifetime; published "
            "March 2026)."
        )

    # ---- Foundation-model lineage ----
    lin = fetch_hf_downloads(HF_LINEAGE)
    if lin["ok"]:
        st.info(
            f"**Foundation-model lineage:** DeepVRegulome is built on "
            f"DNABERT; its successor DNABERT-2 ({HF_LINEAGE}) has "
            f"{lin['downloads']:,} Hugging Face downloads in the last 30 "
            "days.",
            icon="🧬",
        )
