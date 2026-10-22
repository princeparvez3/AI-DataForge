import pandas as pd
import streamlit as st

from app.evaluation_page import render_evaluation_page
from app.ai_annotation_page import render_ai_annotation_page
from app.analytics_page import render_analytics_page
from app.annotation_qa_page import render_annotation_qa_page
from app.adjudication_page import render_adjudication_page
from core.annotation import (
    get_annotation_summary,
    parse_labels,
    prepare_annotation_dataframe,
    save_annotations,
)
from core.cleaning import clean_dataframe
from core.data_split import get_leakage_report, split_dataset
from core.quality_checks import get_quality_report
from core.storage import load_dataframe, load_state, save_dataframe, save_state


st.set_page_config(
    page_title="AI DataForge",
    page_icon="🧠",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --df-ink: #1b2823;
        --df-muted: #68766e;
        --df-paper: #f3f4ee;
        --df-card: #ffffff;
        --df-line: #e1e6dd;
        --df-forest: #17231f;
        --df-forest-2: #20352d;
        --df-lime: #d4f77b;
        --df-violet: #b9adff;
    }

    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--df-paper);
        color: var(--df-ink);
    }

    [data-testid="stHeader"] {
        background: rgba(243, 244, 238, 0.88);
        border-bottom: 1px solid rgba(27, 40, 35, 0.06);
    }

    .block-container {
        max-width: 1580px;
        padding-top: 1.55rem;
        padding-bottom: 3rem;
        padding-left: clamp(1.2rem, 3vw, 3.2rem);
        padding-right: clamp(1.2rem, 3vw, 3.2rem);
    }

    /* Primary navigation and dataset controls now live at the top of the workspace. */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    .df-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: 0 0 0.85rem 0;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 20px;
        background: linear-gradient(115deg, #17231f 0%, #20372e 63%, #30364a 100%);
        box-shadow: 0 10px 28px rgba(23,35,31,0.10);
    }

    .df-topbar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
    }

    .df-topbar-mark {
        display: grid;
        place-items: center;
        flex: 0 0 42px;
        width: 42px;
        height: 42px;
        border-radius: 13px;
        background: var(--df-lime);
        color: var(--df-forest);
        font-size: 14px;
        font-weight: 900;
        letter-spacing: -1px;
    }

    .df-topbar-copy strong {
        display: block;
        color: #fff;
        font-size: 17px;
        letter-spacing: -0.45px;
        line-height: 1.25;
    }

    .df-topbar-copy span {
        display: block;
        margin-top: 4px;
        color: #b2c4b8;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 1.35px;
        text-transform: uppercase;
    }

    .df-topbar-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        flex: 0 0 auto;
        padding: 7px 11px;
        border: 1px solid rgba(255,255,255,0.17);
        border-radius: 999px;
        background: rgba(255,255,255,0.055);
        color: #dce8df;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 1px;
        white-space: nowrap;
    }

    .df-topnav-eyebrow {
        margin: 0.85rem 0 0.45rem 0.15rem;
        color: #758279;
        font-size: 10px;
        font-weight: 850;
        letter-spacing: 1.8px;
        text-transform: uppercase;
    }

    .st-key-active_page {
        margin: 0 0 1rem 0;
        padding: 0.65rem 0.75rem;
        border: 1px solid var(--df-line);
        border-radius: 17px;
        background: rgba(255,255,255,0.88);
        box-shadow: 0 5px 18px rgba(27,40,35,0.035);
    }

    .st-key-active_page [data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        gap: 0.45rem !important;
    }

    .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label {
        box-sizing: border-box;
        min-height: 40px;
        margin: 0 !important;
        padding: 0.48rem 0.72rem !important;
        border: 1px solid #e1e6dd;
        border-radius: 11px;
        background: #fff;
        cursor: pointer;
        transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
    }

    .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        border-color: #b9c9b8;
        background: #f7f9f3;
    }

    .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        border-color: #20372e;
        background: #20372e;
    }

    .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
    .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) span {
        color: var(--df-lime) !important;
        font-weight: 800;
    }

    .st-key-active_page [data-testid="stRadio"] input[type="radio"] {
        accent-color: var(--df-lime);
    }

    .st-key-dataset_source_mode [data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 0.45rem !important;
    }

    .st-key-dataset_source_mode [data-testid="stRadio"] label {
        padding: 0.35rem 0.6rem;
        border-radius: 10px;
    }

    .df-dataset-control {
        margin: 0 0 1.2rem 0;
        padding: 0.9rem 1rem;
        border: 1px solid var(--df-line);
        border-radius: 16px;
        background: rgba(255,255,255,0.78);
    }

    /* The left rail is the primary navigation, not a row of generic tabs. */
    section[data-testid="stSidebar"] {
        background: var(--df-forest);
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }

    section[data-testid="stSidebar"] > div {
        background: var(--df-forest);
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] span {
        color: #e9efe9;
    }

    .df-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0.3rem 0 1.55rem 0;
        padding: 0.25rem 0.15rem 1.2rem 0.15rem;
        border-bottom: 1px solid rgba(255,255,255,0.13);
    }

    .df-brand-mark {
        display: grid;
        place-items: center;
        flex: 0 0 44px;
        height: 44px;
        border-radius: 14px;
        color: #17231f;
        background: var(--df-lime);
        font-size: 15px;
        font-weight: 900;
        letter-spacing: -1px;
        box-shadow: 0 6px 18px rgba(212,247,123,0.13);
    }

    .df-brand-copy strong {
        display: block;
        font-size: 17px;
        color: #ffffff;
        letter-spacing: -0.5px;
        line-height: 1.25;
    }

    .df-brand-copy span {
        display: block;
        margin-top: 4px;
        color: #9eb1a5;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
    }

    .df-side-eyebrow {
        margin: 0.55rem 0 0.4rem 0.1rem;
        color: #90a59a;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.8px;
        text-transform: uppercase;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 5px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        min-height: 38px;
        padding: 0.38rem 0.55rem;
        border: 1px solid transparent;
        border-radius: 10px;
        transition: background 150ms ease, border 150ms ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.07);
        border-color: rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: rgba(212,247,123,0.12);
        border-color: rgba(212,247,123,0.25);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span {
        color: var(--df-lime) !important;
        font-weight: 750;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] input {
        accent-color: var(--df-lime);
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        border: 1px dashed rgba(212,247,123,0.42);
        border-radius: 12px;
        padding: 0.55rem;
        background: rgba(255,255,255,0.035);
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #98aca0;
    }

    .df-sidebar-note {
        margin-top: 1.15rem;
        padding: 0.9rem 0.85rem;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        background: rgba(255,255,255,0.045);
        color: #a8bbb0;
        font-size: 11px;
        line-height: 1.55;
    }

    .df-sidebar-note strong {
        display: block;
        margin-bottom: 4px;
        color: #edf5ec;
        font-size: 12px;
    }

    /* Product-like, dark editorial hero with lime and violet accents. */
    .df-hero {
        position: relative;
        overflow: hidden;
        margin: 0.3rem 0 1.25rem 0;
        padding: clamp(1.35rem, 3vw, 2.25rem);
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.08);
        color: #f6f8f3;
        background:
            radial-gradient(circle at 82% 16%, rgba(185,173,255,0.26), transparent 26%),
            radial-gradient(circle at 58% 100%, rgba(212,247,123,0.13), transparent 32%),
            linear-gradient(122deg, #17231f 0%, #20372e 58%, #30364a 100%);
        box-shadow: 0 18px 42px rgba(23,35,31,0.12);
    }

    .df-hero-top,
    .df-hero-bottom,
    .df-hero-main {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.25rem;
    }

    .df-hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        color: var(--df-lime);
        font-size: 10px;
        font-weight: 850;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .df-hero-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--df-lime);
        box-shadow: 0 0 0 5px rgba(212,247,123,0.1);
    }

    .df-live-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        border: 1px solid rgba(255,255,255,0.17);
        border-radius: 999px;
        padding: 7px 11px;
        background: rgba(255,255,255,0.055);
        color: #dce8df;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 1px;
        white-space: nowrap;
    }

    .df-live-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #9de7a3;
    }

    .df-hero-main {
        align-items: stretch;
        margin-top: 1.8rem;
        margin-bottom: 1.6rem;
    }

    .df-hero-copy {
        max-width: 760px;
        align-self: center;
    }

    .df-hero-copy h1 {
        margin: 0;
        color: #ffffff;
        font-size: clamp(2rem, 4.2vw, 3.55rem);
        font-weight: 780;
        letter-spacing: -2.4px;
        line-height: 1.02;
    }

    .df-hero-copy p {
        margin: 1rem 0 0 0;
        max-width: 650px;
        color: #b9c9be;
        font-size: clamp(0.9rem, 1.4vw, 1.05rem);
        line-height: 1.65;
    }

    .df-hero-art {
        display: flex;
        flex: 0 0 220px;
        flex-direction: column;
        justify-content: space-between;
        min-height: 148px;
        padding: 1.1rem 1.15rem;
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.035));
        backdrop-filter: blur(10px);
    }

    .df-hero-art .df-art-label {
        color: #b7c9bd;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
    }

    .df-hero-art .df-art-symbol {
        margin: 0.25rem 0;
        color: var(--df-lime);
        font-size: 27px;
        line-height: 1;
    }

    .df-hero-art strong {
        color: white;
        font-size: 16px;
        letter-spacing: -0.45px;
        line-height: 1.25;
    }

    .df-hero-art small {
        margin-top: 5px;
        color: #b7c9bd;
        font-size: 10px;
    }

    .df-hero-bottom {
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.12);
        color: #a5b8ab;
        font-size: 9px;
        font-weight: 750;
        letter-spacing: 1.25px;
        text-transform: uppercase;
    }

    .df-workspace-label {
        margin: 1.35rem 0 0.7rem 0.05rem;
        color: #758279;
        font-size: 10px;
        font-weight: 850;
        letter-spacing: 1.8px;
        text-transform: uppercase;
    }

    /* Metrics and workspace surfaces. */
    div[data-testid="stMetric"] {
        min-height: 108px;
        padding: 1rem 1.1rem;
        border: 1px solid var(--df-line);
        border-radius: 16px;
        background: var(--df-card);
        box-shadow: 0 4px 14px rgba(27,40,35,0.025);
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--df-muted);
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 0.65px;
        text-transform: uppercase;
    }

    div[data-testid="stMetricValue"] {
        color: var(--df-ink);
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.8px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--df-line);
        border-radius: 18px;
        background: rgba(255,255,255,0.72);
    }

    .stButton > button,
    .stDownloadButton > button {
        min-height: 42px;
        border-radius: 11px;
        border: 1px solid #d8dfd5;
        font-weight: 750;
        transition: transform 140ms ease, box-shadow 140ms ease;
    }

    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        color: #17231f;
        border-color: #c5ec6a;
        background: var(--df-lime);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(27,40,35,0.08);
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        border-color: #d7ded4;
        border-radius: 10px;
        background: #ffffff;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--df-line);
        border-radius: 14px;
        overflow: hidden;
        background: #ffffff;
    }

    div[data-testid="stAlert"] {
        border-radius: 13px;
    }

    h2, h3 {
        color: var(--df-ink);
        letter-spacing: -0.65px;
    }

    p, li {
        line-height: 1.6;
    }

    /* Keep the few nested tabs inside feature pages visually understated. */
    div[data-testid="stTabs"] button[role="tab"] {
        border-radius: 10px 10px 0 0;
        font-weight: 700;
    }

    @media (max-width: 900px) {
        .df-hero-main {
            align-items: flex-start;
        }
        .df-hero-art {
            flex-basis: 180px;
        }
        .df-hero-copy h1 {
            letter-spacing: -1.5px;
        }
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .df-topbar {
            padding: 0.9rem 1rem;
        }
        .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label {
            flex: 0 1 auto;
        }
    }

    @media (max-width: 650px) {
        .block-container {
            padding-top: 0.75rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
        }
        .df-topbar {
            align-items: flex-start;
            flex-direction: column;
            gap: 0.8rem;
            margin-bottom: 0.65rem;
            padding: 0.9rem;
            border-radius: 16px;
        }
        .df-topbar-status {
            font-size: 9px;
            padding: 6px 9px;
        }
        .st-key-active_page {
            padding: 0.5rem;
            border-radius: 14px;
        }
        .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] {
            display: grid !important;
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            gap: 0.4rem !important;
        }
        .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label {
            display: flex;
            align-items: center;
            width: 100% !important;
            min-width: 0 !important;
            min-height: 42px;
            padding: 0.45rem 0.45rem !important;
            white-space: normal;
            overflow-wrap: anywhere;
            font-size: 0.75rem;
        }
        .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label p {
            font-size: 0.75rem !important;
            line-height: 1.25 !important;
        }
        .st-key-dataset_source_mode [data-testid="stRadio"] div[role="radiogroup"] {
            display: grid !important;
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        }
        .st-key-dataset_source_mode [data-testid="stRadio"] div[role="radiogroup"] label {
            min-width: 0;
            white-space: normal;
        }
        .df-hero {
            padding: 1.1rem;
            margin-top: 0.2rem;
            border-radius: 17px;
        }
        .df-hero-top,
        .df-hero-bottom {
            align-items: flex-start;
            flex-direction: column;
            gap: 0.7rem;
        }
        .df-hero-main {
            flex-direction: column;
            margin-top: 1.1rem;
            margin-bottom: 1rem;
        }
        .df-hero-art {
            display: none;
        }
        .df-hero-copy h1 {
            font-size: clamp(1.8rem, 8vw, 2.2rem);
            letter-spacing: -1px;
            line-height: 1.08;
            overflow-wrap: anywhere;
        }
        .df-hero-copy p {
            font-size: 0.9rem;
            line-height: 1.55;
        }
        div[data-testid="stMetric"] {
            min-height: 88px;
            padding: 0.7rem;
            border-radius: 12px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.45rem;
        }
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.65rem !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
        div[data-testid="stDataFrame"] {
            max-width: 100%;
            overflow-x: auto;
        }
        div[data-testid="stDataEditor"] {
            max-width: 100%;
            overflow-x: auto;
        }
        .stButton > button,
        .stDownloadButton > button {
            width: 100%;
            white-space: normal;
        }
        h1, h2, h3 {
            overflow-wrap: anywhere;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# A second, intentionally focused visual layer gives DataForge a more distinctive
# product identity without changing any dataset or annotation behaviour.
st.markdown(
    r"""
    <style>
    :root {
        --df-ink: #171a35;
        --df-muted: #737995;
        --df-paper: #f5f6fc;
        --df-card: #ffffff;
        --df-line: #e5e7f3;
        --df-forest: #16172f;
        --df-forest-2: #242550;
        --df-lime: #8cf0d5;
        --df-violet: #8a7cff;
        --df-cyan: #72dff2;
        --df-coral: #ffb79f;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 4% 0%, rgba(137, 125, 255, 0.13), transparent 27%),
            radial-gradient(ellipse at 98% 9%, rgba(114, 223, 242, 0.12), transparent 25%),
            linear-gradient(180deg, #f7f8fd 0%, #f2f4fb 100%);
        color: var(--df-ink);
    }

    [data-testid="stHeader"] {
        background: rgba(247, 248, 253, 0.78);
        border-bottom: 1px solid rgba(36, 37, 80, 0.06);
        backdrop-filter: blur(16px);
    }

    .block-container {
        max-width: 1640px;
        padding-top: 1.05rem;
        padding-bottom: 3.25rem;
    }

    /* Premium product masthead */
    .df-topbar {
        position: relative;
        overflow: hidden;
        isolation: isolate;
        min-height: 78px;
        margin: 0 0 1rem 0;
        padding: 1rem 1.25rem;
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 22px;
        background:
            radial-gradient(circle at 78% 0%, rgba(137, 124, 255, 0.34), transparent 30%),
            radial-gradient(circle at 100% 100%, rgba(114, 223, 242, 0.17), transparent 30%),
            linear-gradient(115deg, #13152b 0%, #202147 52%, #302b63 100%);
        box-shadow: 0 18px 46px rgba(35, 34, 87, 0.16), inset 0 1px rgba(255,255,255,0.07);
    }

    .df-topbar::after {
        content: "";
        position: absolute;
        z-index: -1;
        width: 190px;
        height: 190px;
        right: 13%;
        top: -136px;
        border: 1px solid rgba(140,240,213,0.18);
        border-radius: 50%;
        box-shadow: 0 0 0 24px rgba(140,240,213,0.025), 0 0 0 49px rgba(140,240,213,0.02);
    }

    .df-topbar-brand {
        display: flex;
        align-items: center;
        gap: 14px;
        min-width: 0;
    }

    .df-topbar-mark {
        position: relative;
        display: grid;
        place-items: center;
        flex: 0 0 48px;
        width: 48px;
        height: 48px;
        border-radius: 16px;
        background: linear-gradient(140deg, #a9fff0 0%, #8cf0d5 36%, #9c8cff 100%);
        color: #171936;
        font-size: 15px;
        font-weight: 950;
        letter-spacing: -1.1px;
        box-shadow: 0 8px 24px rgba(140,240,213,0.20), inset 0 1px rgba(255,255,255,0.75);
    }

    .df-topbar-copy strong {
        display: block;
        color: #fff;
        font-size: clamp(1.05rem, 1.8vw, 1.3rem);
        font-weight: 820;
        letter-spacing: -0.75px;
        line-height: 1.15;
    }

    .df-topbar-copy span {
        display: block;
        margin-top: 5px;
        color: #b9bbdf;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 1.9px;
        text-transform: uppercase;
    }

    .df-topbar-status {
        border: 1px solid rgba(255,255,255,0.17);
        border-radius: 999px;
        padding: 8px 12px;
        background: rgba(255,255,255,0.07);
        color: #e7eaff;
        font-size: 9px;
        font-weight: 850;
        letter-spacing: 1.35px;
        box-shadow: inset 0 1px rgba(255,255,255,0.06);
    }

    .df-live-dot {
        background: var(--df-lime);
        box-shadow: 0 0 0 4px rgba(140,240,213,0.12), 0 0 14px rgba(140,240,213,0.7);
    }

    .df-topnav-eyebrow,
    .df-workspace-label {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 1.1rem 0 0.65rem 0.2rem;
        color: #7b80a0;
        font-size: 9px;
        font-weight: 900;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .df-topnav-eyebrow::before,
    .df-workspace-label::before {
        content: "";
        display: inline-block;
        width: 20px;
        height: 2px;
        border-radius: 2px;
        background: linear-gradient(90deg, #8174ff, #70dff2);
    }

    /* Nav stays horizontal on desktop and wraps into touch-friendly cells on mobile. */
    .st-key-active_page {
        margin: 0 0 1rem 0;
        padding: 0.65rem;
        border: 1px solid rgba(225,228,243,0.92);
        border-radius: 18px;
        background: rgba(255,255,255,0.82);
        box-shadow: 0 9px 26px rgba(31,34,77,0.045), inset 0 1px #fff;
        backdrop-filter: blur(18px);
    }

    .st-key-active_page [data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        gap: 0.45rem !important;
    }

    .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label {
        display: inline-flex;
        align-items: center;
        box-sizing: border-box;
        min-height: 40px;
        margin: 0 !important;
        padding: 0.45rem 0.75rem !important;
        border: 1px solid transparent;
        border-radius: 12px;
        background: transparent;
        color: #666d8c;
        cursor: pointer;
        transition: background 160ms ease, border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
    }

    .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        border-color: #dfe2f8;
        background: #f5f5ff;
        transform: translateY(-1px);
    }

    .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        border-color: rgba(120,109,255,0.48);
        background: linear-gradient(125deg, #6258e9, #726bfa);
        box-shadow: 0 5px 14px rgba(99,88,233,0.23), inset 0 1px rgba(255,255,255,0.22);
    }

    .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
    .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) span {
        color: #fff !important;
        font-weight: 800;
    }

    .st-key-active_page [data-testid="stRadio"] input[type="radio"] {
        accent-color: #a8fff0;
    }

    /* Dataset control panel */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e4e6f4;
        border-radius: 18px;
        background: rgba(255,255,255,0.78);
        box-shadow: 0 8px 24px rgba(31,34,77,0.035), inset 0 1px #fff;
    }

    .st-key-dataset_source_mode [data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 0.45rem !important;
    }

    .st-key-dataset_source_mode [data-testid="stRadio"] label {
        padding: 0.4rem 0.7rem;
        border-radius: 11px;
        background: #f8f8fd;
        border: 1px solid #e7e8f4;
    }

    .st-key-dataset_source_mode [data-testid="stRadio"] label:has(input:checked) {
        background: #edebff;
        border-color: #9e96ff;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed #b8b2ff;
        border-radius: 14px;
        padding: 0.7rem;
        background: linear-gradient(135deg, rgba(137,124,255,0.06), rgba(114,223,242,0.06));
    }

    /* Dramatic hero with fine grid texture and a halo */
    .df-hero {
        isolation: isolate;
        margin: 0.2rem 0 1.15rem 0;
        padding: clamp(1.4rem, 3.2vw, 2.6rem);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 27px;
        background:
            radial-gradient(circle at 82% 15%, rgba(158,139,255,0.38), transparent 26%),
            radial-gradient(circle at 70% 100%, rgba(114,223,242,0.20), transparent 31%),
            radial-gradient(circle at 10% 105%, rgba(255,183,159,0.12), transparent 32%),
            linear-gradient(118deg, #12142b 0%, #20224d 51%, #352f70 100%);
        box-shadow: 0 24px 58px rgba(40,37,104,0.18), inset 0 1px rgba(255,255,255,0.10);
    }

    .df-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: -1;
        pointer-events: none;
        border-radius: inherit;
        opacity: 0.2;
        background-image: linear-gradient(rgba(255,255,255,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px);
        background-size: 34px 34px;
        mask-image: linear-gradient(90deg, transparent 0%, #000 65%, #000 100%);
    }

    .df-hero-kicker {
        color: #a8fff0;
        letter-spacing: 2.1px;
    }

    .df-hero-dot {
        background: #a8fff0;
        box-shadow: 0 0 0 5px rgba(168,255,240,0.11), 0 0 18px rgba(168,255,240,0.48);
    }

    .df-live-pill {
        border-color: rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.07);
        backdrop-filter: blur(8px);
    }

    .df-hero-copy h1 {
        max-width: 880px;
        color: #fff;
        font-size: clamp(2.25rem, 4.5vw, 4.2rem);
        font-weight: 820;
        letter-spacing: -2.7px;
        line-height: 0.99;
        text-wrap: balance;
    }

    .df-hero-copy p {
        max-width: 710px;
        color: #c4c8e5;
        font-size: clamp(0.92rem, 1.25vw, 1.06rem);
        line-height: 1.75;
    }

    .df-hero-art {
        position: relative;
        overflow: hidden;
        flex: 0 0 256px;
        min-height: 230px;
        padding: 1.15rem;
        border: 1px solid rgba(255,255,255,0.19);
        border-radius: 22px;
        background: linear-gradient(145deg, rgba(255,255,255,0.14), rgba(255,255,255,0.035));
        box-shadow: inset 0 1px rgba(255,255,255,0.11), 0 15px 30px rgba(8,10,38,0.13);
        backdrop-filter: blur(16px);
    }

    .df-hero-art::before {
        content: "";
        position: absolute;
        width: 150px;
        height: 150px;
        right: -38px;
        top: 18px;
        border-radius: 50%;
        background: rgba(137,124,255,0.24);
        filter: blur(22px);
    }

    .df-hero-art .df-art-label,
    .df-hero-art .df-art-symbol,
    .df-hero-art strong,
    .df-hero-art small {
        position: relative;
        z-index: 1;
    }

    .df-hero-art .df-art-label {
        color: #c7c9e8;
        letter-spacing: 1.65px;
    }

    .df-hero-art .df-art-symbol {
        display: grid;
        place-items: center;
        width: 76px;
        height: 76px;
        margin: 1.1rem 0 0.7rem 0;
        border: 1px solid rgba(255,255,255,0.19);
        border-radius: 25px;
        background: linear-gradient(145deg, #a8fff0, #9b8cff);
        color: #171936;
        font-size: 36px;
        box-shadow: 0 12px 26px rgba(137,124,255,0.25), inset 0 1px rgba(255,255,255,0.8);
        animation: df-mark-float 4.4s ease-in-out infinite;
    }

    .df-hero-art strong {
        color: #fff;
        font-size: 17px;
    }

    .df-hero-art small {
        color: #c4c8e5;
    }

    .df-art-orbit {
        position: absolute;
        z-index: 0;
        right: -37px;
        top: 37px;
        width: 175px;
        height: 175px;
        border: 1px solid rgba(168,255,240,0.24);
        border-radius: 50%;
        pointer-events: none;
    }

    .df-art-orbit-one {
        transform: scaleY(0.48) rotate(-34deg);
        border-color: rgba(168,255,240,0.34);
    }

    .df-art-orbit-two {
        transform: scaleY(0.72) rotate(44deg);
        border-color: rgba(164,151,255,0.46);
    }

    .df-art-node {
        position: absolute;
        z-index: 2;
        width: 8px;
        height: 8px;
        right: 25px;
        top: 70px;
        border: 2px solid #201f50;
        border-radius: 50%;
        background: #a8fff0;
        box-shadow: 0 0 0 5px rgba(168,255,240,0.10), 0 0 13px rgba(168,255,240,0.55);
        pointer-events: none;
    }

    .df-art-node-two {
        top: 152px;
        right: 84px;
        background: #ffb79f;
        box-shadow: 0 0 0 5px rgba(255,183,159,0.10), 0 0 13px rgba(255,183,159,0.5);
    }

    @keyframes df-mark-float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-4px); }
    }

    .df-hero-bottom {
        color: #b4b8dc;
        border-top-color: rgba(255,255,255,0.14);
        letter-spacing: 1.35px;
    }

    /* Four custom metric cards create a more editorial dashboard than default widgets. */
    .df-metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 0.15rem 0 1.15rem 0;
    }

    .df-metric-card {
        position: relative;
        overflow: hidden;
        min-width: 0;
        padding: 1.1rem 1.15rem 1rem 1.15rem;
        border: 1px solid #e4e7f4;
        border-radius: 19px;
        background: linear-gradient(145deg, rgba(255,255,255,0.98), rgba(250,250,255,0.94));
        box-shadow: 0 9px 24px rgba(31,34,77,0.045), inset 0 1px #fff;
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }

    .df-metric-card:hover {
        transform: translateY(-3px);
        border-color: #d2d5f3;
        box-shadow: 0 16px 30px rgba(31,34,77,0.08), inset 0 1px #fff;
    }

    .df-metric-card::after {
        content: "";
        position: absolute;
        right: -28px;
        bottom: -43px;
        width: 112px;
        height: 112px;
        border-radius: 50%;
        background: var(--metric-glow, rgba(137,124,255,0.11));
        filter: blur(2px);
    }

    .df-metric-card.violet { --metric-glow: rgba(137,124,255,0.16); --metric-color: #7868fb; }
    .df-metric-card.cyan { --metric-glow: rgba(75,206,226,0.18); --metric-color: #1a9eb6; }
    .df-metric-card.coral { --metric-glow: rgba(255,183,159,0.24); --metric-color: #d67658; }
    .df-metric-card.mint { --metric-glow: rgba(140,240,213,0.26); --metric-color: #268f78; }

    .df-metric-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        position: relative;
        z-index: 1;
    }

    .df-metric-icon {
        display: grid;
        place-items: center;
        width: 35px;
        height: 35px;
        border-radius: 12px;
        background: color-mix(in srgb, var(--metric-color) 10%, white);
        color: var(--metric-color);
        font-size: 17px;
        font-weight: 900;
    }

    .df-metric-tag {
        color: #969ab3;
        font-size: 8px;
        font-weight: 900;
        letter-spacing: 1.25px;
        text-align: right;
    }

    .df-metric-value {
        position: relative;
        z-index: 1;
        margin-top: 1rem;
        color: #1d2040;
        font-size: clamp(1.65rem, 2.8vw, 2.25rem);
        font-weight: 850;
        letter-spacing: -1.2px;
        line-height: 1;
        font-variant-numeric: tabular-nums;
    }

    .df-metric-title {
        position: relative;
        z-index: 1;
        margin-top: 0.45rem;
        color: #606785;
        font-size: 0.82rem;
        font-weight: 760;
    }

    .df-metric-foot {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        gap: 7px;
        margin-top: 0.75rem;
        color: #9297af;
        font-size: 10px;
    }

    .df-metric-foot span {
        display: inline-block;
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: var(--metric-color);
    }

    /* Improve controls throughout every workflow page. */
    h1, h2, h3 {
        color: #202344;
        letter-spacing: -0.7px;
    }

    h2, h3 {
        font-weight: 800;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: #606783;
    }

    .stButton > button,
    .stDownloadButton > button {
        min-height: 43px;
        border-radius: 12px;
        border: 1px solid #dfe2f1;
        background: #fff;
        color: #34375c;
        font-weight: 760;
        box-shadow: 0 3px 9px rgba(33,35,78,0.035);
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }

    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        border: 1px solid rgba(101,88,237,0.3);
        background: linear-gradient(120deg, #665bea, #8277ff);
        color: #fff;
        box-shadow: 0 7px 17px rgba(101,88,237,0.20), inset 0 1px rgba(255,255,255,0.2);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        border-color: #aaa4fb;
        box-shadow: 0 10px 22px rgba(54,52,117,0.10);
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        border-color: #dfe2f1;
        border-radius: 12px;
        background: #fff;
        box-shadow: 0 2px 7px rgba(32,34,80,0.025);
    }

    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #8277ff;
        box-shadow: 0 0 0 3px rgba(130,119,255,0.12);
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stDataEditor"] {
        border: 1px solid #e3e6f3;
        border-radius: 15px;
        overflow: hidden;
        background: white;
        box-shadow: 0 6px 18px rgba(31,34,77,0.035);
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    hr {
        border-color: #e2e5f1;
    }

    /* Responsive treatment for tablets and Android-sized displays. */
    @media (max-width: 950px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .df-hero-copy h1 {
            letter-spacing: -1.7px;
        }
        .df-hero-art {
            flex-basis: 215px;
        }
        .df-metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 650px) {
        .block-container {
            padding-top: 0.65rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
        }
        .df-topbar {
            padding: 0.9rem;
            border-radius: 18px;
        }
        .df-topbar-brand {
            gap: 10px;
        }
        .df-topbar-mark {
            flex-basis: 42px;
            width: 42px;
            height: 42px;
            border-radius: 13px;
        }
        .df-topbar-status {
            font-size: 8px;
            padding: 6px 8px;
            letter-spacing: 0.8px;
        }
        .st-key-active_page {
            padding: 0.45rem;
            border-radius: 15px;
        }
        .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] {
            display: grid !important;
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            gap: 0.35rem !important;
        }
        .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label {
            width: 100% !important;
            min-width: 0 !important;
            min-height: 41px;
            padding: 0.42rem 0.45rem !important;
            border-radius: 10px;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        .st-key-active_page [data-testid="stRadio"] div[role="radiogroup"] label p {
            font-size: 0.72rem !important;
            line-height: 1.25 !important;
        }
        .df-hero {
            padding: 1.05rem;
            border-radius: 19px;
        }
        .df-hero-top,
        .df-hero-bottom {
            align-items: flex-start;
            flex-direction: column;
            gap: 0.65rem;
        }
        .df-hero-main {
            flex-direction: column;
            margin-top: 1.2rem;
            margin-bottom: 1.1rem;
        }
        .df-hero-art {
            display: none;
        }
        .df-hero-copy h1 {
            font-size: clamp(1.8rem, 8.5vw, 2.35rem);
            letter-spacing: -1.05px;
            line-height: 1.06;
        }
        .df-hero-copy p {
            font-size: 0.88rem;
            line-height: 1.58;
        }
        .df-metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
        }
        .df-metric-card {
            padding: 0.8rem;
            border-radius: 15px;
        }
        .df-metric-icon {
            width: 30px;
            height: 30px;
            border-radius: 10px;
            font-size: 14px;
        }
        .df-metric-tag {
            max-width: 72px;
            font-size: 7px;
            letter-spacing: 0.7px;
        }
        .df-metric-value {
            margin-top: 0.8rem;
            font-size: 1.55rem;
        }
        .df-metric-title {
            font-size: 0.75rem;
        }
        .df-metric-foot {
            margin-top: 0.55rem;
            font-size: 9px;
        }
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.6rem !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stDataEditor"] {
            max-width: 100%;
            overflow-x: auto;
        }
        .stButton > button,
        .stDownloadButton > button {
            width: 100%;
            white-space: normal;
        }
        h1, h2, h3 {
            overflow-wrap: anywhere;
        }
    }

    @media (max-width: 360px) {
        .df-topbar-status {
            white-space: normal;
            text-align: center;
        }
        .df-metric-grid {
            grid-template-columns: minmax(0, 1fr);
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def create_sample_dataset() -> pd.DataFrame:
    """Create a sample customer-support dataset."""
    return pd.DataFrame(
        [
            {"id": 1, "text": "My order has not arrived yet.", "intent": "Order Status", "sentiment": "Negative"},
            {"id": 2, "text": "The product quality is excellent.", "intent": "Feedback", "sentiment": "Positive"},
            {"id": 3, "text": "I want to return my damaged product.", "intent": "Return Request", "sentiment": "Negative"},
            {"id": 4, "text": "How can I reset my password?", "intent": "Account Help", "sentiment": "Neutral"},
            {"id": 1, "text": "My order has not arrived yet.", "intent": "Order Status", "sentiment": "Negative"},
            {"id": 6, "text": "Thank you for the quick support.", "intent": "Feedback", "sentiment": "Positive"},
            {"id": 7, "text": "Please cancel my order.", "intent": "Cancellation", "sentiment": "Neutral"},
            {"id": 8, "text": "I cannot log in to my account.", "intent": "Account Help", "sentiment": "Negative"},
        ]
    )


def clear_saved_split() -> None:
    """Remove split results when the source dataset changes."""
    for key in (
        "split_train_df",
        "split_test_df",
        "split_leakage_report",
        "split_settings",
    ):
        st.session_state.pop(key, None)


def persist_dataset(dataframe: pd.DataFrame, source: str) -> None:
    """Save the active dataset and its source to SQLite."""
    save_dataframe(dataframe, key="main_dataset")
    save_state("dataset_source", source)
    st.session_state.dataset = dataframe
    st.session_state.current_source = source


# ------------------------------------------------------------------
# Load persistent state at application startup
# ------------------------------------------------------------------
if "dataset" not in st.session_state:
    saved_dataset = load_dataframe(key="main_dataset")
    saved_source = load_state("dataset_source", default="Saved dataset")

    if saved_dataset is not None:
        st.session_state.dataset = saved_dataset
        st.session_state.current_source = saved_source
    else:
        sample_dataset = create_sample_dataset()
        st.session_state.dataset = sample_dataset
        st.session_state.current_source = "Sample dataset"
        save_dataframe(sample_dataset, key="main_dataset")
        save_state("dataset_source", "Sample dataset")

if "current_source" not in st.session_state:
    st.session_state.current_source = load_state(
        "dataset_source",
        default="Saved dataset",
    )

if "loaded_file_signature" not in st.session_state:
    st.session_state.loaded_file_signature = None


# ------------------------------------------------------------------
# Sidebar: product navigation and dataset controls
# ------------------------------------------------------------------
PAGE_OPTIONS = [
    "01  Overview",
    "02  Data Quality",
    "03  Clean Dataset",
    "04  Manual Annotation",
    "05  AI Annotation",
    "06  Analytics",
    "07  Annotation QA",
    "08  Adjudication",
    "09  Train / Test Split",
    "10  LLM Evaluation",
    "11  Export Data",
]

PAGE_DISPLAY_NAMES = {
    "01  Overview": "⌂  Overview",
    "02  Data Quality": "◈  Data Quality",
    "03  Clean Dataset": "✦  Clean Dataset",
    "04  Manual Annotation": "✎  Manual Annotation",
    "05  AI Annotation": "✧  AI Annotation",
    "06  Analytics": "◌  Analytics",
    "07  Annotation QA": "◎  Annotation QA",
    "08  Adjudication": "◆  Adjudication",
    "09  Train / Test Split": "⇄  Train / Test Split",
    "10  LLM Evaluation": "✳  LLM Evaluation",
    "11  Export Data": "↗  Export Data",
}

PAGE_DETAILS = {
    "01  Overview": (
        "WORKSPACE / OVERVIEW",
        "From raw data to trusted labels.",
        "One focused workspace to inspect datasets, improve annotation quality, evaluate AI outputs, and prepare dependable training data.",
    ),
    "02  Data Quality": (
        "DATA LAB / QUALITY CONTROL",
        "Find issues before they become model errors.",
        "Inspect missing values, duplicates, and repeated text before those issues flow into training or evaluation.",
    ),
    "03  Clean Dataset": (
        "DATA LAB / PREPARATION",
        "Clean the data. Keep control of the changes.",
        "Trim noisy whitespace and remove exact duplicate rows while preserving records that need human review.",
    ),
    "04  Manual Annotation": (
        "ANNOTATION STUDIO / HUMAN REVIEW",
        "Put consistent labels on every example.",
        "Assign labels manually, monitor coverage, and save your reviewed annotations to the workspace database.",
    ),
    "05  AI Annotation": (
        "ANNOTATION STUDIO / AI ASSIST",
        "Let AI suggest. Let reviewers decide.",
        "Generate constrained label suggestions, review the rationale, correct mistakes, and approve only the labels you trust.",
    ),
    "06  Analytics": (
        "OBSERVABILITY / ANNOTATION METRICS",
        "See progress, not just rows in a table.",
        "Track annotation coverage, label balance, and the quality signals available from your current dataset and review history.",
    ),
    "07  Annotation QA": (
        "QUALITY SYSTEM / INTER-ANNOTATOR AGREEMENT",
        "Measure how consistently labels are applied.",
        "Compare two annotators, inspect the confusion matrix, and find the disagreements that deserve another look.",
    ),
    "08  Adjudication": (
        "QUALITY SYSTEM / FINAL REVIEW",
        "Turn disagreements into clear decisions.",
        "Review conflicting annotations, select an approved final label, and save decisions back to the dataset.",
    ),
    "09  Train / Test Split": (
        "MODEL READINESS / DATA SPLITTING",
        "Split responsibly. Check for overlap.",
        "Create training and test sets, keep related records together when needed, and review potential leakage before evaluation.",
    ),
    "10  LLM Evaluation": (
        "EVALUATION LAB / RESPONSE COMPARISON",
        "Compare answers with a consistent rubric.",
        "Score response quality, record preference decisions, and export structured evaluation data for downstream workflows.",
    ),
    "11  Export Data": (
        "DATA DELIVERY / EXPORTS",
        "Move clean, structured data into your next workflow.",
        "Download the current dataset in practical machine-learning formats without losing your reviewed labels.",
    ),
}

st.markdown(
    """
    <div class="df-topbar">
        <div class="df-topbar-brand">
            <div class="df-topbar-mark">DF</div>
            <div class="df-topbar-copy">
                <strong>AI DataForge</strong>
                <span>AI data operations · Quality studio</span>
            </div>
        </div>
        <div class="df-topbar-status"><span class="df-live-dot"></span> WORKSPACE READY</div>
    </div>
    <div class="df-topnav-eyebrow">CHOOSE YOUR WORKSPACE</div>
    """,
    unsafe_allow_html=True,
)

active_page = st.radio(
    "Workspace navigation",
    options=PAGE_OPTIONS,
    format_func=lambda page: PAGE_DISPLAY_NAMES.get(page, page),
    horizontal=True,
    label_visibility="collapsed",
    key="active_page",
)

with st.container(border=True):
    st.markdown('<div class="df-topnav-eyebrow">DATASET CONTROL</div>', unsafe_allow_html=True)
    st.caption(f"Active source: {st.session_state.current_source} · Local SQLite storage · data/dataforge.db")
    source_mode = st.radio(
        "Choose dataset source",
        ["Current dataset", "Sample dataset", "Upload a file"],
        horizontal=True,
        key="dataset_source_mode",
    )

    if source_mode == "Sample dataset":
        if st.session_state.current_source != "Sample dataset":
            clear_saved_split()
            sample_dataset = create_sample_dataset()
            persist_dataset(sample_dataset, "Sample dataset")
            st.session_state.loaded_file_signature = None

    elif source_mode == "Upload a file":
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=["csv", "json", "jsonl"],
            key="dataset_upload",
        )

        if uploaded_file is not None:
            signature = f"{uploaded_file.name}:{uploaded_file.size}"

            if signature != st.session_state.loaded_file_signature:
                try:
                    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
                    uploaded_file.seek(0)

                    if extension == "csv":
                        new_dataset = pd.read_csv(uploaded_file)
                    elif extension == "jsonl":
                        new_dataset = pd.read_json(uploaded_file, lines=True)
                    else:
                        new_dataset = pd.read_json(uploaded_file)

                    if not isinstance(new_dataset, pd.DataFrame):
                        raise ValueError("The file must contain tabular records.")

                    new_dataset.columns = [
                        str(column).strip() for column in new_dataset.columns
                    ]

                    if not new_dataset.columns.is_unique:
                        raise ValueError("Column names must be unique.")

                    clear_saved_split()
                    uploaded_dataset = new_dataset.reset_index(drop=True)
                    persist_dataset(uploaded_dataset, "Uploaded file")
                    st.session_state.loaded_file_signature = signature
                    st.success("Dataset loaded and saved successfully!")

                except Exception as error:
                    st.error(f"Could not load the file: {error}")


# ------------------------------------------------------------------
# Main workspace header
# ------------------------------------------------------------------

saved_adjudication_count = st.session_state.pop(
    "adjudication_saved_count",
    None,
)
if saved_adjudication_count is not None:
    st.success(
        "Final adjudication decisions saved successfully "
        f"({saved_adjudication_count} record(s))."
    )

df = st.session_state.dataset

if df.empty:
    st.warning(
        "The dataset is empty. Select Sample dataset or upload a file "
        "containing records."
    )
    st.stop()

hero_kicker, hero_title, hero_subtitle = PAGE_DETAILS[active_page]

st.markdown(
    f"""
    <section class="df-hero">
        <div class="df-hero-top">
            <div class="df-hero-kicker"><span class="df-hero-dot"></span>{hero_kicker}</div>
            <div class="df-live-pill"><span class="df-live-dot"></span>LOCAL WORKSPACE</div>
        </div>
        <div class="df-hero-main">
            <div class="df-hero-copy">
                <h1>{hero_title}</h1>
                <p>{hero_subtitle}</p>
            </div>
            <div class="df-hero-art">
                <span class="df-art-label">AI DATAFORGE / WORKFLOW</span>
                <span class="df-art-symbol">✳</span>
                <strong>Data → Decisions</strong>
                <small>Curated · Reviewed · Ready</small>
                <div class="df-art-orbit df-art-orbit-one"></div>
                <div class="df-art-orbit df-art-orbit-two"></div>
                <div class="df-art-node df-art-node-one"></div>
                <div class="df-art-node df-art-node-two"></div>
            </div>
        </div>
        <div class="df-hero-bottom">
            <span>Pipeline · Prepare / Label / Validate / Evaluate</span>
            <span>Workspace 0.8</span>
        </div>
    </section>
    <div class="df-workspace-label">CURRENT DATASET / LIVE SNAPSHOT</div>
    """,
    unsafe_allow_html=True,
)

record_count = len(df)
field_count = len(df.columns)
missing_count = int(df.isna().sum().sum())
duplicate_count = int(df.duplicated().sum())

st.markdown(
    f"""
    <div class="df-metric-grid">
        <article class="df-metric-card violet">
            <div class="df-metric-top"><span class="df-metric-icon">▦</span><span class="df-metric-tag">DATA VOLUME</span></div>
            <div class="df-metric-value">{record_count:,}</div>
            <div class="df-metric-title">Total records</div>
            <div class="df-metric-foot"><span></span>Rows in the active dataset</div>
        </article>
        <article class="df-metric-card cyan">
            <div class="df-metric-top"><span class="df-metric-icon">⌗</span><span class="df-metric-tag">SCHEMA</span></div>
            <div class="df-metric-value">{field_count:,}</div>
            <div class="df-metric-title">Dataset fields</div>
            <div class="df-metric-foot"><span></span>Columns available to workflows</div>
        </article>
        <article class="df-metric-card coral">
            <div class="df-metric-top"><span class="df-metric-icon">◌</span><span class="df-metric-tag">COMPLETENESS</span></div>
            <div class="df-metric-value">{missing_count:,}</div>
            <div class="df-metric-title">Missing cells</div>
            <div class="df-metric-foot"><span></span>Values to review</div>
        </article>
        <article class="df-metric-card mint">
            <div class="df-metric-top"><span class="df-metric-icon">⧉</span><span class="df-metric-tag">INTEGRITY</span></div>
            <div class="df-metric-value">{duplicate_count:,}</div>
            <div class="df-metric-title">Exact duplicates</div>
            <div class="df-metric-foot"><span></span>Identical rows detected</div>
        </article>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="df-workspace-label">ACTIVE WORKSPACE / SELECTED MODULE</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------
# Workspace page 1: Dataset overview
# ------------------------------------------------------------------
if active_page == "01  Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Column Information")
    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(df[column].dtype) for column in df.columns],
            "Missing Values": [
                int(df[column].isna().sum()) for column in df.columns
            ],
        }
    )
    st.dataframe(column_info, use_container_width=True)


# ------------------------------------------------------------------
# Workspace page 2: Quality checks
# ------------------------------------------------------------------
if active_page == "02  Data Quality":
    st.subheader("Data Quality Report")
    report = get_quality_report(df)

    missing_report = pd.DataFrame(
        {
            "Column": list(report["missing_by_column"].keys()),
            "Missing Values": list(report["missing_by_column"].values()),
        }
    )
    st.write("Missing values by column")
    st.dataframe(missing_report, use_container_width=True)
    st.metric("Exact Duplicate Rows", report["duplicate_rows"])

    if report["text_column"] is not None:
        st.write(f"Text column checked: `{report['text_column']}`")
        st.metric("Repeated Text Values", report["repeated_texts"])
        if report["repeated_texts"] > 0:
            st.info(
                "Repeated text can be legitimate. Review these records "
                "before removing anything."
            )
        else:
            st.success("No repeated non-empty text values found.")
    else:
        st.info(
            "No text column was found. Repeated-text checking was skipped."
        )


# ------------------------------------------------------------------
# Workspace page 3: Basic cleaning
# ------------------------------------------------------------------
if active_page == "03  Clean Dataset":
    st.subheader("Basic Cleaning")
    st.write(
        "Trim whitespace from text values and remove exact duplicate rows. "
        "Missing values are not automatically deleted or filled."
    )

    if st.button("Apply Basic Cleaning", type="primary"):
        cleaned_df, rows_removed = clean_dataframe(
            st.session_state.dataset
        )
        clear_saved_split()
        persist_dataset(cleaned_df, st.session_state.current_source)
        st.success(
            f"Cleaning complete! Removed {rows_removed} exact duplicate row(s)."
        )
        st.rerun()


# ------------------------------------------------------------------
# Workspace page 4: Annotation studio
# ------------------------------------------------------------------
if active_page == "04  Manual Annotation":
    st.subheader("Manual Data Annotation")

    if st.session_state.pop("annotation_saved", False):
        st.success("Annotations saved successfully to the dataset and database!")

    annotation_dataset = st.session_state.dataset
    text_columns = annotation_dataset.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if not text_columns:
        st.warning("No text columns were found in this dataset.")
    else:
        preferred_names = ["text", "message", "content", "prompt", "query"]
        preferred_column = next(
            (
                column
                for name in preferred_names
                for column in text_columns
                if column.lower() == name
            ),
            text_columns[0],
        )

        text_column = st.selectbox(
            "Select the text column",
            options=text_columns,
            index=text_columns.index(preferred_column),
            key="annotation_text_column",
        )
        annotation_column = st.text_input(
            "New or existing annotation column",
            value="annotation",
            key="annotation_column_name",
        ).strip()
        labels_text = st.text_input(
            "Allowed labels (comma-separated)",
            value="Positive, Negative, Neutral, Other",
            key="annotation_labels_text",
        )
        labels = parse_labels(labels_text)

        if not annotation_column:
            st.warning("Enter a name for the annotation column.")
        elif annotation_column == text_column:
            st.warning(
                "The annotation column must be different from the text column."
            )
        elif not labels:
            st.warning("Enter at least one allowed label.")
        else:
            annotation_base = prepare_annotation_dataframe(
                annotation_dataset,
                annotation_column,
            )
            summary = get_annotation_summary(
                annotation_base,
                annotation_column,
            )

            progress1, progress2, progress3 = st.columns(3)
            progress1.metric("Total Records", summary["total"])
            progress2.metric("Annotated Records", summary["annotated"])
            progress3.metric("Remaining Records", summary["remaining"])
            st.write("Choose a label for each record.")

            current_annotations = (
                annotation_base[annotation_column]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            existing_labels = [
                value
                for value in current_annotations.unique().tolist()
                if value
            ]
            available_labels = list(dict.fromkeys(labels + existing_labels))

            annotation_view = pd.DataFrame(
                {
                    "Text": annotation_base[text_column].fillna("").astype(str),
                    "Annotation": current_annotations,
                },
                index=annotation_base.index,
            )

            edited_view = st.data_editor(
                annotation_view,
                column_config={
                    "Text": st.column_config.TextColumn("Text"),
                    "Annotation": st.column_config.SelectboxColumn(
                        "Annotation",
                        options=available_labels,
                        required=False,
                    ),
                },
                disabled=["Text"],
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                key=f"annotation_editor_{text_column}_{annotation_column}",
            )

            if st.button(
                "Save Annotations",
                type="primary",
                key="save_annotations_button",
            ):
                edited_for_save = edited_view.rename(
                    columns={"Annotation": annotation_column}
                )
                updated_dataset = save_annotations(
                    annotation_base,
                    edited_for_save,
                    annotation_column,
                )
                clear_saved_split()
                persist_dataset(
                    updated_dataset,
                    st.session_state.current_source,
                )
                st.session_state.annotation_saved = True
                st.rerun()


# ------------------------------------------------------------------
# Workspace page 5: AI-assisted annotation
# ------------------------------------------------------------------
if active_page == "05  AI Annotation":
    render_ai_annotation_page()


# ------------------------------------------------------------------
# Workspace page 6: Annotation analytics
# ------------------------------------------------------------------
if active_page == "06  Analytics":
    render_analytics_page(st.session_state.dataset)


# ------------------------------------------------------------------
# Workspace page 7: Annotation QA and inter-annotator agreement
# ------------------------------------------------------------------
if active_page == "07  Annotation QA":
    render_annotation_qa_page()


# ------------------------------------------------------------------
# Workspace page 8: Annotation adjudication
# ------------------------------------------------------------------
if active_page == "08  Adjudication":
    render_adjudication_page()


# ------------------------------------------------------------------
# Workspace page 9: Train/test split and data leakage check
# ------------------------------------------------------------------
if active_page == "09  Train / Test Split":
    st.subheader("Train/Test Split")
    st.write(
        "Prepare separate datasets for training and evaluation. Group-aware "
        "splitting can keep records with the same selected group value together."
    )

    test_percentage = st.slider(
        "Test set percentage",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        key="split_test_percentage",
    )
    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        key="split_random_seed",
    )

    group_options = ["No grouping"] + df.columns.tolist()
    default_group = "text" if "text" in df.columns else "No grouping"
    group_choice = st.selectbox(
        "Group records by (recommended for repeated text)",
        options=group_options,
        index=group_options.index(default_group),
        key="split_group_choice",
    )
    group_column = None if group_choice == "No grouping" else group_choice
    stratify_column = None

    if group_column:
        st.info(
            f"Group-aware mode is using `{group_column}`. Matching group "
            "values will stay in the same split. Stratification is disabled "
            "when group-aware mode is selected."
        )
    else:
        stratify_options = ["No stratification"] + df.columns.tolist()
        stratify_choice = st.selectbox(
            "Optional: stratify by a label column",
            options=stratify_options,
            key="split_stratify_choice",
        )
        if stratify_choice != "No stratification":
            stratify_column = stratify_choice
        st.caption(
            "Stratification tries to preserve label proportions in both sets. "
            "Each class needs enough records for the requested split."
        )

    likely_leakage_columns = [
        "text", "message", "content", "prompt", "query",
        "id", "record_id", "sample_id",
    ]
    default_check_columns = [
        column for column in likely_leakage_columns if column in df.columns
    ]
    if group_column and group_column not in default_check_columns:
        default_check_columns.append(group_column)

    leakage_columns = st.multiselect(
        "Columns to check for shared values",
        options=df.columns.tolist(),
        default=default_check_columns,
        key="split_leakage_columns",
        help=(
            "Choose identifier or text columns, not ordinary category labels "
            "such as sentiment, which are expected to repeat."
        ),
    )

    if st.button(
        "Generate Train/Test Split",
        type="primary",
        key="generate_split_button",
    ):
        clear_saved_split()
        try:
            train_df, test_df = split_dataset(
                df,
                test_size=test_percentage / 100,
                random_state=int(random_seed),
                group_column=group_column,
                stratify_column=stratify_column,
            )
            leakage_report = get_leakage_report(
                train_df,
                test_df,
                leakage_columns,
            )
            st.session_state.split_train_df = train_df
            st.session_state.split_test_df = test_df
            st.session_state.split_leakage_report = leakage_report
            st.session_state.split_settings = {
                "test_percentage": test_percentage,
                "random_seed": int(random_seed),
                "group_column": group_column,
                "stratify_column": stratify_column,
                "leakage_columns": leakage_columns,
            }
        except (ValueError, KeyError) as error:
            st.error(str(error))

    if "split_train_df" in st.session_state:
        train_df = st.session_state.split_train_df
        test_df = st.session_state.split_test_df
        leakage_report = st.session_state.split_leakage_report
        settings = st.session_state.split_settings

        st.divider()
        st.subheader("Split Results")
        result1, result2, result3 = st.columns(3)
        result1.metric("Training Records", len(train_df))
        result2.metric("Testing Records", len(test_df))
        result3.metric("Total Split Records", len(train_df) + len(test_df))

        if settings["group_column"]:
            st.caption(
                f"Last split grouped by `{settings['group_column']}`. "
                "Because groups stay together, the final test percentage may "
                "differ slightly from the requested percentage."
            )
        else:
            st.caption(
                f"Last split: {settings['test_percentage']}% test data, "
                f"random seed {settings['random_seed']}."
            )

        train_preview, test_preview = st.tabs(
            ["Training Dataset", "Testing Dataset"]
        )
        with train_preview:
            st.dataframe(train_df, use_container_width=True)
        with test_preview:
            st.dataframe(test_df, use_container_width=True)

        st.subheader("Data Leakage Check")
        if not leakage_report:
            st.info(
                "No columns were selected for leakage checking. Generate the "
                "split again after selecting text or ID columns."
            )
        else:
            leakage_rows = []
            for column, details in leakage_report.items():
                overlap_count = details["overlap_count"]
                leakage_rows.append(
                    {
                        "Column": column,
                        "Shared Values": overlap_count,
                        "Status": (
                            "Review overlap"
                            if overlap_count
                            else "No shared values found"
                        ),
                        "Examples": ", ".join(details["shared_values"]),
                    }
                )
            st.dataframe(
                pd.DataFrame(leakage_rows),
                use_container_width=True,
                hide_index=True,
            )
            overlapping_columns = [
                column
                for column, details in leakage_report.items()
                if details["overlap_count"] > 0
            ]
            if overlapping_columns:
                st.warning(
                    "Shared values were found in: "
                    + ", ".join(overlapping_columns)
                    + ". Review these records. Shared values indicate possible "
                    "leakage, not proof of leakage in every case."
                )
            else:
                st.success(
                    "No shared values were found in the selected columns."
                )

        st.subheader("Download Split Datasets")
        download1, download2, download3, download4 = st.columns(4)
        with download1:
            st.download_button(
                "Training CSV",
                data=train_df.to_csv(index=False).encode("utf-8"),
                file_name="dataforge_train.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_train_csv",
            )
        with download2:
            st.download_button(
                "Testing CSV",
                data=test_df.to_csv(index=False).encode("utf-8"),
                file_name="dataforge_test.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_test_csv",
            )
        with download3:
            st.download_button(
                "Training JSONL",
                data=train_df.to_json(
                    orient="records", lines=True, force_ascii=False
                ).encode("utf-8"),
                file_name="dataforge_train.jsonl",
                mime="application/x-ndjson",
                use_container_width=True,
                key="download_train_jsonl",
            )
        with download4:
            st.download_button(
                "Testing JSONL",
                data=test_df.to_json(
                    orient="records", lines=True, force_ascii=False
                ).encode("utf-8"),
                file_name="dataforge_test.jsonl",
                mime="application/x-ndjson",
                use_container_width=True,
                key="download_test_jsonl",
            )


# ------------------------------------------------------------------
# Workspace page 10: LLM evaluation
# ------------------------------------------------------------------
if active_page == "10  LLM Evaluation":
    render_evaluation_page()


# ------------------------------------------------------------------
# Workspace page 11: Export full dataset
# ------------------------------------------------------------------
if active_page == "11  Export Data":
    st.subheader("Download Full Dataset")
    current_df = st.session_state.dataset

    csv_data = current_df.to_csv(index=False).encode("utf-8")
    jsonl_data = current_df.to_json(
        orient="records",
        lines=True,
        force_ascii=False,
    ).encode("utf-8")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="dataforge_dataset.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_full_csv",
        )
    with col2:
        st.download_button(
            "Download JSONL",
            data=jsonl_data,
            file_name="dataforge_dataset.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
            key="download_full_jsonl",
        )

st.divider()
st.caption("AI DataForge · Data trust workspace · Local development build")
