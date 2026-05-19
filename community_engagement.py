"""
community_engagement.py
-----------------------
Community Engagement panel for the DeepVRegulome portal.

LAYOUT:
  Row 1 (3 tiles): PyPI downloads | Portal users | Portal page views
  Row 2 (2 tiles): Hugging Face downloads (30d) | Citations
  Then: DNABERT-2 foundation-model lineage callout.

WHY TILES MIGHT BE BLANK ("—"):
  - PyPI: pypistats.org throttles some cloud IPs (esp. requests with no
    User-Agent). This version sends a real UA and retries once. If it
    still fails, the diagnostics panel (below) shows the exact reason.
  - Portal users / page views: these need the GA4 service-account block
    in secrets ([ga.credentials] + property_id). Blank here means that
    block is absent or invalid. This is expected until you add it.
  - HF / Citations / lineage: no credentials needed; should always show.

DIAGNOSTICS:
  Set SHOW_DIAGNOSTICS = True (or add to secrets: [debug] community = true)
  to render a small expander explaining the status of every data source.
  Leave it False for the public portal.

Usage in Home.py:
    from community_engagement import render_community_engagement
    render_community_engagement()
"""

import datetime
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

# Flip to True to see why any tile is blank. Keep False for visitors.
SHOW_DIAGNOSTICS = False

REQUEST_TIMEOUT = 8

# A real User-Agent matters: pypistats and some APIs throttle or reject
# requests that arrive with the default python-requests UA, which is a
# common reason a call works locally but returns "—" on Streamlit Cloud.
HEADERS = {
    "User-Agent": "DeepVRegulome-Portal/1.0 (research; contact via GitHub)",
    "Accept": "application/json",
}


def _get(url, params=None):
    """GET with UA + one retry. Returns (response_or_None, error_string)."""
    last_err = ""
    for attempt in range(2):
        try:
            r = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code == 200:
                return r, ""
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return None, last_err


# ---------------------------------------------------------------------------
# DATA FETCHERS  (cached; each records a status for diagnostics)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pypi_total(package: str) -> dict:
    out = {
        "ok": False,
        "total": 0,
        "version": None,
        "first_date": None,
        "last_date": None,
        "status": "",
    }

    r, err = _get(
        f"https://pypistats.org/api/packages/{package}/overall",
        params={"mirrors": "false"},
    )

    if r is None:
        out["status"] = f"pypistats overall failed: {err}"
        return out

    try:
        series = r.json().get("data", [])
        if series:
            out["total"] = sum(x["downloads"] for x in series)
            dates = sorted(x["date"] for x in series)
            out["first_date"], out["last_date"] = dates[0], dates[-1]
            out["ok"] = True
            out["status"] = "ok"
        else:
            out["status"] = "pypistats returned no data rows"
    except Exception as e:
        out["status"] = f"pypistats parse error: {e}"

    rv, _ = _get(f"https://pypi.org/pypi/{package}/json")
    if rv is not None:
        try:
            out["version"] = rv.json()["info"]["version"]
        except Exception:
            pass

    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hf_downloads(model_id: str) -> dict:
    out = {"ok": False, "downloads": 0, "status": ""}

    r, err = _get(f"https://huggingface.co/api/models/{model_id}")

    if r is None:
        out["status"] = f"HF failed: {err}"
        return out

    try:
        out["downloads"] = r.json().get("downloads", 0) or 0
        out["ok"] = True
        out["status"] = "ok"
    except Exception as e:
        out["status"] = f"HF parse error: {e}"

    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ga4_totals():
    """Live portal users + page views. Returns dict with 'status'."""
    out = {
        "ok": False,
        "total_users": 0,
        "page_views": 0,
        "status": "",
    }

    try:
        if "ga" not in st.secrets:
            out["status"] = "no [ga] section in secrets"
            return out

        ga = st.secrets["ga"]

        if "credentials" not in ga or "property_id" not in ga:
            out["status"] = (
                "missing [ga.credentials] and/or property_id in secrets "
                "(GA4 service account not set up yet)"
            )
            return out

        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        from google.analytics.data_v1beta.types import (
            RunReportRequest,
            DateRange,
            Metric,
        )
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_info(
            dict(ga["credentials"])
        )

        client = BetaAnalyticsDataClient(credentials=creds)

        req = RunReportRequest(
            property=f"properties/{ga['property_id']}",
            metrics=[
                Metric(name="totalUsers"),
                Metric(name="screenPageViews"),
            ],
            date_ranges=[
                DateRange(
                    start_date="2024-01-01",
                    end_date="today",
                )
            ],
        )

        resp = client.run_report(req)

        if not resp.rows:
            out["status"] = "GA4 returned no rows"
            return out

        out.update(
            ok=True,
            total_users=int(resp.rows[0].metric_values[0].value),
            page_views=int(resp.rows[0].metric_values[1].value),
            status="ok",
        )

    except Exception as e:
        out["status"] = f"GA4 error: {type(e).__name__}: {e}"

    return out


# ---------------------------------------------------------------------------
# VISUAL HELPERS
# ---------------------------------------------------------------------------
def _diagnostics_enabled() -> bool:
    if SHOW_DIAGNOSTICS:
        return True
    try:
        return bool(st.secrets["debug"]["community"])
    except Exception:
        return False


def _inject_metric_card_css():
    st.markdown(
        """
        <style>
        .dvr-metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #e5e7eb;
            border-left: 6px solid var(--accent);
            border-radius: 20px;
            padding: 20px 22px;
            min-height: 158px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
            transition: all 0.2s ease-in-out;
        }

        .dvr-metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.13);
        }

        .dvr-metric-top {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 14px;
        }

        .dvr-metric-icon {
            width: 40px;
            height: 40px;
            min-width: 40px;
            border-radius: 13px;
            background: rgba(15, 23, 42, 0.055);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
        }

        .dvr-metric-title {
            font-size: 15px;
            font-weight: 750;
            color: #334155;
            line-height: 1.2;
        }

        .dvr-metric-value {
            font-size: 38px;
            font-weight: 850;
            color: #0f172a;
            letter-spacing: -1px;
            margin-bottom: 5px;
        }

        .dvr-metric-subtitle {
            font-size: 12.5px;
            color: #64748b;
            font-weight: 500;
            line-height: 1.3;
        }

        .dvr-section-note {
            font-size: 13px;
            color: #64748b;
            margin-top: 0.2rem;
            margin-bottom: 1.1rem;
        }

        @media (max-width: 768px) {
            .dvr-metric-card {
                margin-bottom: 14px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def adoption_card(icon: str, title: str, value: str, subtitle: str, accent: str):
    st.markdown(
        f"""
        <div class="dvr-metric-card" style="--accent:{accent};">
            <div class="dvr-metric-top">
                <div class="dvr-metric-icon">{icon}</div>
                <div class="dvr-metric-title">{title}</div>
            </div>
            <div class="dvr-metric-value">{value}</div>
            <div class="dvr-metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    _inject_metric_card_css()

    pypi_value = (
        f"{pypi['total']:,}"
        if pypi["ok"] and pypi["total"]
        else "—"
    )

    portal_users_value = (
        f"{ga['total_users']:,}"
        if ga["ok"]
        else "—"
    )

    page_views_value = (
        f"{ga['page_views']:,}"
        if ga["ok"]
        else "—"
    )

    hf_value = (
        f"{hf['downloads']:,}"
        if hf["ok"]
        else "—"
    )

    # ---- Row 1 ----
    r1c1, r1c2, r1c3 = st.columns(3)

    with r1c1:
        pypi_subtitle = "Total package downloads"
        if pypi.get("version"):
            pypi_subtitle += f" · v{pypi['version']}"
        adoption_card(
            icon="📦",
            title="PyPI Downloads",
            value=pypi_value,
            subtitle=pypi_subtitle,
            accent="#6366f1",
        )

    with r1c2:
        adoption_card(
            icon="👥",
            title="Portal Users",
            value=portal_users_value,
            subtitle="Unique visitors tracked by GA4",
            accent="#10b981",
        )

    with r1c3:
        adoption_card(
            icon="👁️",
            title="Portal Page Views",
            value=page_views_value,
            subtitle="Total page engagement",
            accent="#f59e0b",
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ---- Row 2 ----
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        adoption_card(
            icon="🤗",
            title="Hugging Face Downloads",
            value=hf_value,
            subtitle="Rolling 30-day model downloads",
            accent="#ec4899",
        )

    with r2c2:
        adoption_card(
            icon="📄",
            title="Research Citations",
            value=f"{CITATIONS:,}",
            subtitle="Manually updated citation count",
            accent="#8b5cf6",
        )

    if pypi["ok"] and pypi.get("first_date"):
        st.caption(
            f"PyPI cumulative total spans {pypi['first_date']} → "
            f"{pypi['last_date']} "
            "(the package's full lifetime; published March 2026)."
        )

    # ---- Foundation-model lineage ----
    lin = fetch_hf_downloads(HF_LINEAGE)

    if lin["ok"]:
        st.info(
            f"**Foundation-model lineage:** DeepVRegulome builds on the "
            f"DNABERT family of genome language models; DNABERT-2 "
            f"({HF_LINEAGE}) has {lin['downloads']:,} Hugging Face downloads "
            "in the last 30 days.",
            icon="🧬",
        )

    # ---- Optional diagnostics (hidden from visitors) ----
    if _diagnostics_enabled():
        with st.expander("🔧 Data source diagnostics", expanded=True):
            st.write("**PyPI:**", pypi.get("status") or "unknown")
            st.write("**Hugging Face model:**", hf.get("status"))
            st.write("**Hugging Face lineage:**", lin.get("status"))
            st.write("**GA4 portal:**", ga.get("status"))
            st.caption(
                "Each fetch is cached for 1 hour. After fixing a cause "
                "(e.g. adding GA4 secrets), reboot the app or clear cache "
                "so the fix takes effect immediately."
            )
