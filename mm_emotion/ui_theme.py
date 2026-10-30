"""A restrained lab-notebook theme for the remote experiment interface."""


def apply_lab_theme():
    import streamlit as st
    st.markdown('''<style>
    :root { --lab-ink:#192b36; --lab-rule:#d4dce1; --lab-green:#315b43; }
    .stApp { background:#f8fafb; color:var(--lab-ink); }
    [data-testid="stMainBlockContainer"] { max-width:1180px; padding-top:2rem; padding-bottom:3rem; }
    h1,h2,h3 { color:var(--lab-ink); letter-spacing:-.025em; font-family:"Segoe UI",Arial,sans-serif !important; }
    h1 { font-size:2.2rem !important; font-weight:600 !important; margin-bottom:.2rem; }
    h2 { font-weight:600 !important; }
    h3 { font-size:1.25rem !important; font-weight:600 !important; margin-top:1.2rem; }
    .lab-eyebrow { font:11px/1.6 Consolas,monospace; letter-spacing:.12em; color:#586252; margin:0 0 8px; }
    [data-testid="stSidebar"] { background:#edf1f4; border-right:1px solid var(--lab-rule); }
    [data-testid="stSidebar"] h2 { font-size:1.45rem; }
    [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
      border-radius:3px !important; box-shadow:none !important;
    }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
    [data-testid="stLinkButton"] a { border-radius:3px !important; box-shadow:none !important; }
    button:focus-visible, a:focus-visible { outline:2px solid var(--lab-green) !important; outline-offset:3px; }
    [data-testid="stTabs"] [role="tablist"] { gap:28px; border-bottom:1px solid var(--lab-rule); }
    [data-testid="stTabs"] [role="tab"] { border-radius:0; padding-left:0; padding-right:0; }
    [data-testid="stDataFrame"] { border:1px solid var(--lab-rule); border-radius:2px; }
    [data-testid="stExpander"] { border-radius:2px; border-color:var(--lab-rule); box-shadow:none; }
    .stCaption, [data-testid="stCaptionContainer"] p { color:#536674 !important; opacity:1; }
    @media(max-width:700px) {
      [data-testid="stMainBlockContainer"] { padding:1.2rem 1rem 2rem; }
      h1 { font-size:2rem !important; }
      [data-testid="stHorizontalBlock"] { flex-direction:column; gap:.8rem; }
      [data-testid="stColumn"] { width:100% !important; flex:1 1 auto !important; }
    }
    </style>''',unsafe_allow_html=True)
