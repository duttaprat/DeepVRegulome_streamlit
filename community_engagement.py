"""
community_engagement.py
-----------------------
Community Engagement panel for the DeepVRegulome portal.

Replaces the old get_analytics_data() that crashed on a missing
st.secrets["ga"]["credentials"] key.

DESIGN DECISION (final):
GA4 country data is unreliable on Streamlit Community Cloud (the hosting
proxy + iframe sandbox mean the visitor's real IP rarely reaches GA4, so
the Country dimension stays "(not set)"). Rather than show a broken or
empty map, this panel presents the metrics that ARE reliable and strong:

  - PyPI total installs (full lifetime) + 30-day window   [no auth]
  - Hugging Face model downloads (DeepVRegulome)           [no auth]
  - DNABERT-2 foundation-model lineage downloads           [no auth]
  - GitHub stars / forks                                   [no auth]
  - GA4 total portal users + page views                    [service acct]

The country chart still renders automatically IF GA4 ever returns real
geo-resolved rows (e.g. if the portal is later moved off Streamlit
Community Cloud). If not, that section is simply omitted: no warning,
no empty map, no apology text.

Usage in Home.py:
    from community_engagement import render_community_engagement
    render_community_engagement()
"""

import datetime
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG  (DeepVRegulome only)
# ---------------------------------------------------------------------------
PYPI_PACKAGE = "deepvregulome"          # https://pypi.org/project/deepvregulome/
GITHUB_REPO = "DavuluriLab/DeepVRegulome"
HF_MODEL = "duttaprat/DeepVRegulome"
HF_LINEAGE = "zhihan1996/DNABERT-2-117M"  # co-authored foundation model

REQUEST_TIMEOUT = 6


# ---------------------------------------------------------------------------
# DATA FETCHERS  (cached; each fails soft)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pypi_stats(package: str) -> dict:
    out = {
        "ok": False, "version": None,
        "last_day": 0, "last_week": 0, "last_month": 0,
        "total": 0, "first_date": None, "last_date": None,
    }
    try:
        r = requests.get(
            f"https://pypistats.org/api/packages/{package}/recent",
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            d = r.json().get("data", {})
            out.update(
                ok=True,
                last_day=d.get("last_day", 0),
                last_week=d.get("last_week", 0),
                last_month=d.get("last_month", 0),
            )
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
def fetch_hf_model(model_id: str) -> dict:
    out = {"ok": False, "id": model_id, "downloads": 0, "likes": 0}
    try:
        r = requests.get(
            f"https://huggingface.co/api/models/{model_id}",
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            d = r.json()
            out.update(
                ok=True,
                downloads=d.get("downloads", 0) or 0,
                likes=d.get("likes", 0) or 0,
            )
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_github_stats(repo: str) -> dict:
    out = {"ok": False, "stars": 0, "forks": 0}
    try:
        r = requests.get(
            f"https://api.github.com/repos/{repo}",
            headers={"Accept": "application/vnd.github+json"},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            d = r.json()
            out.update(ok=True,
                       stars=d.get("stargazers_count", 0),
                       forks=d.get("forks_count", 0))
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ga4():
    """
    Returns GA4 portal metrics, or None if not configured.

    total_users / page_views are reliable. Country rows are included only
    if GA4 actually returns geo-resolved data; "(not set)" rows are
    filtered out so the renderer can decide to simply omit the section.
    """
    try:
        if "ga" not in st.secrets:
            return None
        ga = st.secrets["ga"]
        if "credentials" not in ga or "property_id" not in ga:
            return None

        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        from google.analytics.data_v1beta.types import (
            RunReportRequest, DateRange, Dimension, Metric,
        )
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_info(
            dict(ga["credentials"])
        )
        client = BetaAnalyticsDataClient(credentials=creds)
        pid = ga["property_id"]

        totals_req = RunReportRequest(
            property=f"properties/{pid}",
            metrics=[Metric(name="totalUsers"),
                     Metric(name="screenPageViews")],
            date_ranges=[DateRange(start_date="2024-01-01",
                                   end_date="today")],
        )
        tr = client.run_report(totals_req)
        total_users = page_views = 0
        if tr.rows:
            total_users = int(tr.rows[0].metric_values[0].value)
            page_views = int(tr.rows[0].metric_values[1].value)

        ctry_req = RunReportRequest(
            property=f"properties/{pid}",
            dimensions=[Dimension(name="country")],
            metrics=[Metric(name="totalUsers")],
            date_ranges=[DateRange(start_date="2024-01-01",
                                   end_date="today")],
        )
        cr = client.run_report(ctry_req)
        rows = [
            {"Country": r.dimension_values[0].value or "(not set)",
             "Visitors": int(r.metric_values[0].value)}
            for r in cr.rows
        ]
        cdf = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["Country", "Visitors"]
        )
        resolved = (
            cdf[(cdf["Country"] != "(not set)") & (cdf["Country"] != "")]
            if not cdf.empty else cdf
        )

        return {
            "ok": True,
            "total_users": total_users,
            "page_views": page_views,
            "countries_resolved": resolved,
            "n_countries_resolved": int(resolved["Country"].nunique())
            if not resolved.empty else 0,
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
        "Live DeepVRegulome adoption across PyPI, Hugging Face, GitHub, and "
        f"the interactive portal. Refreshed hourly · "
        f"{datetime.date.today().isoformat()}."
    )

    pypi = fetch_pypi_stats(PYPI_PACKAGE)
    gh = fetch_github_stats(GITHUB_REPO)
    hf = fetch_hf_model(HF_MODEL)
    ga = fetch_ga4()

    # ---- Headline metrics ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
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
    with c2:
        st.metric(
            "PyPI installs (30 days)",
            f"{pypi['last_month']:,}" if pypi["ok"] else "—",
        )
    with c3:
        # Prefer GA4 portal users here (strong, reliable). Fall back to
        # HF downloads if GA4 is not configured.
        if ga and ga.get("ok"):
            st.metric("Portal users (all time)",
                      f"{ga['total_users']:,}")
        else:
            st.metric(
                "Hugging Face downloads (30d)",
                f"{hf['downloads']:,}" if hf["ok"] else "—",
                help=f"Model: {HF_MODEL}",
            )
    with c4:
        if ga and ga.get("ok"):
            st.metric("Portal page views (all time)",
                      f"{ga['page_views']:,}")
        else:
            st.metric(
                "GitHub stars",
                f"{gh['stars']:,}" if gh["ok"] else "—",
                help=f"{GITHUB_REPO} · {gh['forks']} forks"
                if gh["ok"] else None,
            )

    if pypi["ok"] and pypi.get("first_date"):
        st.caption(
            f"PyPI cumulative total spans {pypi['first_date']} → "
            f"{pypi['last_date']} (the package's full lifetime; published "
            "March 2026)."
        )

    # ---- Secondary metrics row (HF + GitHub) when GA4 took the top row ----
    if ga and ga.get("ok"):
        s1, s2 = st.columns(2)
        with s1:
            st.metric(
                "Hugging Face downloads (30d)",
                f"{hf['downloads']:,}" if hf["ok"] else "—",
                help=f"Model: {HF_MODEL}",
            )
        with s2:
            st.metric(
                "GitHub stars",
                f"{gh['stars']:,}" if gh["ok"] else "—",
                help=f"{GITHUB_REPO} · {gh['forks']} forks"
                if gh["ok"] else None,
            )

    # ---- Foundation-model lineage ----
    lin = fetch_hf_model(HF_LINEAGE)
    if lin["ok"]:
        st.info(
            f"**Foundation-model lineage:** DeepVRegulome is built on "
            f"DNABERT; its successor DNABERT-2 ({HF_LINEAGE}) has "
            f"{lin['downloads']:,} Hugging Face downloads in the last 30 "
            "days.",
            icon="🧬",
        )

    # ---- Country chart: shown ONLY if GA4 actually returned real geo ----
    # On Streamlit Community Cloud this is normally empty, so nothing is
    # rendered here (no warning, no empty map). It will appear on its own
    # if the portal is ever served from a host that passes client IPs.
    if ga and ga.get("ok") and ga["n_countries_resolved"] > 0:
        resolved = ga["countries_resolved"]
        st.markdown(
            f"**Geographic reach: {ga['n_countries_resolved']} countries**"
        )
        top = resolved.nlargest(8, "Visitors")
        try:
            import plotly.express as px

            fig = px.bar(
                top.sort_values("Visitors"),
                x="Visitors", y="Country", orientation="h", text="Visitors",
            )
            fig.update_traces(marker_color="#009E73",
                              textposition="outside")
            fig.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title=None,
                height=160 + 34 * len(top),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.dataframe(top, use_container_width=True, hide_index=True)

    # ---- Grant / CV-ready summary ----
    bits = []
    if pypi["ok"] and pypi["total"]:
        bits.append(f"{pypi['total']:,} total PyPI installs")
    if pypi["ok"]:
        bits.append(f"{pypi['last_month']:,} PyPI installs in the trailing "
                     "30 days")
    if ga and ga.get("ok"):
        bits.append(f"{ga['total_users']:,} portal users and "
                     f"{ga['page_views']:,} page views")
        if ga["n_countries_resolved"] > 0:
            bits.append(f"across {ga['n_countries_resolved']} countries")
    if hf["ok"] and hf["downloads"]:
        bits.append(f"{hf['downloads']:,} Hugging Face downloads (30d)")
    if gh["ok"] and gh["stars"]:
        bits.append(f"{gh['stars']:,} GitHub stars")
    if bits:
        with st.expander("📋 Copy adoption summary (for grants / CV)"):
            st.code(
                f"DeepVRegulome adoption (as of "
                f"{datetime.date.today().isoformat()}): "
                + "; ".join(bits) + ".",
                language="text",
            )
