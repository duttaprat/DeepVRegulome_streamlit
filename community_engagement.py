"""
community_engagement.py
-----------------------
Drop-in Community Engagement panel for the DeepVRegulome portal.

WHY THIS EXISTS
The old get_analytics_data() in Home.py tried to read
st.secrets["ga"]["credentials"] and st.secrets["ga"]["property_id"].
Neither key exists in secrets.toml, so the panel always errored out
("st.secrets has no key 'credentials'").

This module replaces that with adoption metrics that need NO credentials
and are far stronger evidence of scientific uptake than portal page views:
  - PyPI download counts for the `deepvregulome` package
  - Hugging Face download counts for the hosted models
  - GitHub stars / forks for the public repo

GA4 country data is kept OPTIONAL and OFF by default. It only activates
if a valid service-account block is present in secrets. The panel never
crashes when it is absent; it just shows the sources that do work.

Usage in Home.py:
    from community_engagement import render_community_engagement
    render_community_engagement()
"""

import datetime
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG: edit these if package / repo names ever change
# ---------------------------------------------------------------------------
PYPI_PACKAGE = "deepvregulome"
GITHUB_REPO = "DavuluriLab/DeepVRegulome"
HF_MODELS = [
    "duttaprat/DeepVRegulome",
    "duttaprat/HViLM-base",
]
# DNABERT-2 is co-authored; shown separately as "foundation model lineage".
HF_LINEAGE = ["zhihan1996/DNABERT-2-117M"]

REQUEST_TIMEOUT = 6  # seconds; keep short so the page never hangs


# ---------------------------------------------------------------------------
# DATA FETCHERS  (each is independently cached and fails soft)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pypi_stats(package: str) -> dict:
    """Recent download counts from pypistats.org (no auth required)."""
    out = {"ok": False, "last_day": 0, "last_week": 0, "last_month": 0, "version": None}
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
        # version is a nice-to-have, separate endpoint
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
    """All-time + 30-day download counts for one HF model (no auth required)."""
    out = {"ok": False, "id": model_id, "downloads": 0, "downloads_30d": 0, "likes": 0}
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
                downloads_30d=d.get("downloadsAllTime", 0) or d.get("downloads", 0) or 0,
                likes=d.get("likes", 0) or 0,
            )
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_github_stats(repo: str) -> dict:
    """Stars / forks for the public repo (unauth: 60 req/hr, fine with caching)."""
    out = {"ok": False, "stars": 0, "forks": 0, "open_issues": 0}
    try:
        r = requests.get(
            f"https://api.github.com/repos/{repo}",
            headers={"Accept": "application/vnd.github+json"},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            d = r.json()
            out.update(
                ok=True,
                stars=d.get("stargazers_count", 0),
                forks=d.get("forks_count", 0),
                open_issues=d.get("open_issues_count", 0),
            )
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ga4_countries():
    """
    OPTIONAL. Returns (total_users, n_countries, top5_df) ONLY if a full
    service-account block exists in secrets. Returns None otherwise so the
    caller can simply skip the map. Never raises.
    """
    try:
        if "ga" not in st.secrets:
            return None
        ga = st.secrets["ga"]
        if "credentials" not in ga or "property_id" not in ga:
            return None  # not configured -> silently skip, no error shown

        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        from google.analytics.data_v1beta.types import (
            RunReportRequest, DateRange, Dimension, Metric,
        )
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_info(
            dict(ga["credentials"])
        )
        client = BetaAnalyticsDataClient(credentials=creds)
        req = RunReportRequest(
            property=f"properties/{ga['property_id']}",
            dimensions=[Dimension(name="country")],
            metrics=[Metric(name="totalUsers")],
            date_ranges=[DateRange(start_date="2024-01-01", end_date="today")],
        )
        resp = client.run_report(req)
        rows = [
            {"Country": row.dimension_values[0].value,
             "Visitors": int(row.metric_values[0].value)}
            for row in resp.rows
        ]
        if not rows:
            return None
        df = pd.DataFrame(rows)
        return df["Visitors"].sum(), df["Country"].nunique(), df.nlargest(5, "Visitors")
    except Exception:
        # Any failure -> behave exactly as "not configured". No scary red box.
        return None


# ---------------------------------------------------------------------------
# RENDERER
# ---------------------------------------------------------------------------
def render_community_engagement():
    st.divider()
    st.header("🌎 Community Engagement & Adoption")
    st.caption(
        "Live adoption metrics across PyPI, Hugging Face, and GitHub. "
        f"Auto-refreshed hourly · last loaded {datetime.date.today().isoformat()}."
    )

    pypi = fetch_pypi_stats(PYPI_PACKAGE)
    gh = fetch_github_stats(GITHUB_REPO)
    hf_models = [fetch_hf_model(m) for m in HF_MODELS]
    hf_total = sum(m["downloads"] for m in hf_models if m["ok"])

    # ---- Headline metrics ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "PyPI installs (30 days)",
            f"{pypi['last_month']:,}" if pypi["ok"] else "—",
            help=f"Package: {PYPI_PACKAGE}"
            + (f" · v{pypi['version']}" if pypi.get("version") else ""),
        )
    with c2:
        st.metric(
            "PyPI installs (7 days)",
            f"{pypi['last_week']:,}" if pypi["ok"] else "—",
        )
    with c3:
        st.metric(
            "Hugging Face downloads",
            f"{hf_total:,}" if any(m["ok"] for m in hf_models) else "—",
            help="Combined across hosted DeepVRegulome / HViLM models",
        )
    with c4:
        st.metric(
            "GitHub stars",
            f"{gh['stars']:,}" if gh["ok"] else "—",
            help=f"{GITHUB_REPO} · {gh['forks']} forks" if gh["ok"] else None,
        )

    # ---- Per-model HF breakdown ----
    rows = []
    for m in hf_models:
        if m["ok"]:
            rows.append(
                {"Model": m["id"], "Downloads": m["downloads"], "Likes": m["likes"]}
            )
    if rows:
        st.markdown("**Hosted model downloads (Hugging Face)**")
        df = pd.DataFrame(rows).sort_values("Downloads", ascending=False)
        try:
            import plotly.express as px

            fig = px.bar(
                df.sort_values("Downloads"),
                x="Downloads",
                y="Model",
                orientation="h",
                text="Downloads",
            )
            fig.update_traces(marker_color="#0072B2", textposition="outside")
            fig.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title=None,
                height=180 + 40 * len(df),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- Foundation-model lineage (DNABERT-2, co-authored) ----
    lineage = [fetch_hf_model(m) for m in HF_LINEAGE]
    lin_ok = [m for m in lineage if m["ok"]]
    if lin_ok:
        names = ", ".join(m["id"] for m in lin_ok)
        total = sum(m["downloads"] for m in lin_ok)
        st.info(
            f"**Foundation-model lineage:** DeepVRegulome is built on "
            f"DNABERT, whose successor DNABERT-2 ({names}) has "
            f"{total:,} downloads on Hugging Face.",
            icon="🧬",
        )

    # ---- OPTIONAL GA4 country map (only if fully configured) ----
    ga = fetch_ga4_countries()
    if ga is not None:
        total_users, n_countries, top5 = ga
        st.markdown("**Portal global reach (Google Analytics)**")
        gcol1, gcol2 = st.columns([1, 2], gap="large")
        with gcol1:
            st.metric("Portal visitors", f"{total_users:,}")
            st.metric("Countries reached", n_countries)
        with gcol2:
            try:
                import plotly.express as px

                fig = px.bar(
                    top5.sort_values("Visitors"),
                    x="Visitors",
                    y="Country",
                    orientation="h",
                    text="Visitors",
                )
                fig.update_traces(marker_color="#009E73", textposition="outside")
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis_title=None,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.dataframe(top5, use_container_width=True, hide_index=True)

    # ---- Grant/CV-ready summary line ----
    bits = []
    if pypi["ok"]:
        bits.append(f"{pypi['last_month']:,} PyPI installs in the last 30 days")
    if any(m["ok"] for m in hf_models):
        bits.append(f"{hf_total:,} Hugging Face model downloads")
    if gh["ok"]:
        bits.append(f"{gh['stars']:,} GitHub stars")
    if bits:
        with st.expander("📋 Copy adoption summary (for grants / CV)"):
            st.code(
                "DeepVRegulome adoption (as of "
                f"{datetime.date.today().isoformat()}): "
                + "; ".join(bits)
                + ".",
                language="text",
            )
