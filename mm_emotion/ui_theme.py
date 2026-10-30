"""A restrained lab-notebook theme for the experiment interfaces.

Motion here is feedback only: controls acknowledge a press, alerts arrive when an
action finishes, and tab panels cross-fade when the view changes. Streamlit
reruns the script on every interaction, so nothing animates on plain rerender.
"""

FONT_STACK = '"IBM Plex Sans","Segoe UI",system-ui,-apple-system,Arial,sans-serif'
MONO_STACK = '"IBM Plex Mono",ui-monospace,Consolas,monospace'

# Shared with the Plotly figures so charts and chrome read as one system.
INK = '#192b36'
INK_SOFT = '#536674'
RULE = '#d4dce1'
SURFACE = '#f8fafb'
PANEL = '#edf1f4'
ACCENT = '#315b43'

_CSS = '''<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap");
:root {
  --lab-ink:#192b36; --lab-ink-soft:#536674; --lab-rule:#d4dce1; --lab-surface:#f8fafb;
  --lab-panel:#edf1f4; --lab-green:#315b43; --lab-green-tint:#e3ece6;
  --lab-sans:__SANS__; --lab-mono:__MONO__;
  --lab-ease:cubic-bezier(.16,1,.3,1); --lab-fast:120ms; --lab-base:200ms; --lab-slow:320ms;
}

/* Base type and browser surfaces */
.stApp { background:var(--lab-surface); color:var(--lab-ink); font-family:var(--lab-sans); caret-color:var(--lab-green); }
.stApp ::selection { background:var(--lab-green-tint); color:var(--lab-ink); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button { font-family:var(--lab-sans); }
.stApp code, .stApp pre, .stApp kbd { font-family:var(--lab-mono); font-size:.9em; }
.stApp a { color:var(--lab-green); text-underline-offset:.2em; text-decoration-thickness:1px;
  transition:text-decoration-color var(--lab-base) var(--lab-ease); }
.stApp a:hover { text-decoration-thickness:2px; }
* { scrollbar-width:thin; scrollbar-color:#b8c3ca transparent; }

[data-testid="stMainBlockContainer"] { max-width:1180px; padding-top:2.25rem; padding-bottom:4rem; }
[data-testid="stMarkdownContainer"] p { text-wrap:pretty; }
[data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] p { max-width:78ch; }
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p { text-wrap:nowrap; max-width:none; }

h1, h2, h3 { color:var(--lab-ink); font-family:var(--lab-sans) !important; text-wrap:balance; }
h1 { font-size:2.25rem !important; font-weight:600 !important; letter-spacing:-.03em; line-height:1.15 !important;
  padding-bottom:.15rem !important; }
h2 { font-size:1.6rem !important; font-weight:600 !important; letter-spacing:-.02em; }
h3 { font-size:1.15rem !important; font-weight:600 !important; letter-spacing:-.01em;
  margin-top:2.1rem !important; padding-bottom:.35rem !important; }
[data-testid="stHeading"]:has(h3) { border-top:1px solid var(--lab-rule); margin-top:.6rem; }
[data-testid="stCaptionContainer"] { opacity:1 !important; }
.stCaption, [data-testid="stCaptionContainer"] p { color:var(--lab-ink-soft) !important; line-height:1.5; }

/* Numbers in data line up */
[data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stMetricValue"], .js-plotly-plot .number,
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBar"] { font-variant-numeric:tabular-nums; }
[data-testid="stMetricValue"] { font-weight:500; letter-spacing:-.02em; }

/* Sidebar */
[data-testid="stSidebar"] { background:var(--lab-panel); border-right:1px solid var(--lab-rule); }
[data-testid="stSidebar"] h2 { font-size:1.35rem !important; }

/* Inputs */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
  border-radius:3px !important; box-shadow:none !important;
  transition:border-color var(--lab-base) var(--lab-ease), background-color var(--lab-base) var(--lab-ease);
}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within, [data-baseweb="select"] > div:focus-within {
  border-color:var(--lab-green) !important;
}

/* Buttons: hover tints, press settles down a pixel */
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
[data-testid="stLinkButton"] a, [data-testid="stFormSubmitButton"] button {
  border-radius:3px !important; box-shadow:none !important; font-weight:500;
  transition:background-color var(--lab-base) var(--lab-ease), border-color var(--lab-base) var(--lab-ease),
    color var(--lab-base) var(--lab-ease), transform var(--lab-fast) var(--lab-ease);
}
[data-testid="stButton"] button:hover:not(:disabled), [data-testid="stDownloadButton"] button:hover:not(:disabled),
[data-testid="stLinkButton"] a:hover { border-color:var(--lab-green); color:var(--lab-green); }
[data-testid="stButton"] button[kind="primary"]:hover:not(:disabled) { color:#fff; background:#274a36; border-color:#274a36; }
[data-testid="stButton"] button:active:not(:disabled), [data-testid="stDownloadButton"] button:active:not(:disabled),
[data-testid="stLinkButton"] a:active { transform:translateY(1px); }
[data-testid="stButton"] button:disabled { cursor:not-allowed; }
button:focus-visible, a:focus-visible, [role="tab"]:focus-visible,
input:focus-visible { outline:2px solid var(--lab-green) !important; outline-offset:3px; }

/* Workspace switch: horizontal radio reads as a segmented control */
.st-key-lab_workspace [data-testid="stRadioGroup"] {
  display:inline-flex; border:1px solid var(--lab-rule); border-radius:4px; padding:3px; background:var(--lab-panel); gap:2px !important;
}
.st-key-lab_workspace [data-testid="stRadioOption"] {
  margin:0 !important; padding:.4rem 1rem; border-radius:3px; cursor:pointer; color:var(--lab-ink-soft);
  transition:background-color var(--lab-base) var(--lab-ease), color var(--lab-base) var(--lab-ease), box-shadow var(--lab-base) var(--lab-ease);
}
.st-key-lab_workspace [data-testid="stRadioOption"] > div > div:first-child { display:none; }
.st-key-lab_workspace [data-testid="stRadioOption"] p { font-weight:500; }
.st-key-lab_workspace [data-testid="stRadioOption"]:hover { color:var(--lab-ink); }
.st-key-lab_workspace [data-testid="stRadioOption"][data-selected="true"] {
  background:#fff; color:var(--lab-ink); box-shadow:0 1px 2px rgba(25,43,54,.14);
}
.st-key-lab_workspace [data-testid="stRadioOption"]:has(input:focus-visible) { outline:2px solid var(--lab-green); outline-offset:1px; }

/* Tabs: the underline slides in on the selected tab, panels cross-fade */
[data-testid="stTabs"] [role="tablist"] { gap:28px; border-bottom:1px solid var(--lab-rule); }
[data-testid="stTabs"] [role="tab"] { border-radius:0; padding-left:0; padding-right:0; color:var(--lab-ink-soft);
  transition:color var(--lab-base) var(--lab-ease); }
[data-testid="stTabs"] [role="tab"][aria-selected="true"], [data-testid="stTabs"] [role="tab"]:hover { color:var(--lab-ink); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--lab-green); height:2px;
  transition:transform var(--lab-slow) var(--lab-ease), width var(--lab-slow) var(--lab-ease) !important; }
[data-testid="stTabs"] [role="tabpanel"]:not([hidden]) { animation:lab-panel-in var(--lab-slow) var(--lab-ease); }

/* Containers */
[data-testid="stDataFrame"] { border:1px solid var(--lab-rule); border-radius:2px; }
[data-testid="stExpander"] details { border-radius:2px; border-color:var(--lab-rule); box-shadow:none; }
[data-testid="stExpander"] summary { transition:color var(--lab-base) var(--lab-ease); }
[data-testid="stExpander"] summary:hover { color:var(--lab-green); }
[data-testid="stExpander"] summary svg { transition:transform var(--lab-base) var(--lab-ease); }
[data-testid="stExpander"] details[open] > summary + div { animation:lab-reveal var(--lab-base) var(--lab-ease); }

/* Feedback: outcomes of an action arrive, they do not pop */
[data-testid="stAlert"] { animation:lab-arrive var(--lab-slow) var(--lab-ease); }
[data-testid="stAlert"] > div { border-radius:3px; }
[data-testid="stSpinner"] { color:var(--lab-ink-soft); }

/* Plotly modebar stays out of the way until wanted */
.js-plotly-plot .modebar { opacity:0; transition:opacity var(--lab-base) var(--lab-ease); }
.js-plotly-plot:hover .modebar, .js-plotly-plot:focus-within .modebar { opacity:1; }

@keyframes lab-arrive { from { opacity:0; transform:translateY(-4px); } to { opacity:1; transform:none; } }
@keyframes lab-panel-in { from { opacity:.25; } to { opacity:1; } }
@keyframes lab-reveal { from { opacity:0; clip-path:inset(0 0 100% 0); } to { opacity:1; clip-path:inset(0); } }

@media (prefers-reduced-motion:reduce) {
  [data-testid="stAlert"] { animation:lab-panel-in var(--lab-base) linear; }
  [data-testid="stExpander"] details[open] > summary + div { animation:lab-panel-in var(--lab-base) linear; }
  [data-testid="stButton"] button:active, [data-testid="stDownloadButton"] button:active,
  [data-testid="stLinkButton"] a:active { transform:none; }
  [data-testid="stTabs"] [data-baseweb="tab-highlight"] { transition:none !important; }
}

@media (max-width:700px) {
  [data-testid="stMainBlockContainer"] { padding:1.25rem 1rem 2.5rem; }
  h1 { font-size:1.85rem !important; }
  [data-testid="stHorizontalBlock"] { flex-direction:column; gap:.8rem; }
  [data-testid="stColumn"] { width:100% !important; flex:1 1 auto !important; }
  [data-testid="stTabs"] [role="tablist"] { gap:18px; overflow-x:auto; }
}
</style>'''


def apply_lab_theme():
    import streamlit as st
    st.markdown(_CSS.replace('__SANS__',FONT_STACK).replace('__MONO__',MONO_STACK),unsafe_allow_html=True)


def plot_font(size=12):
    """Plotly font matching the page chrome."""
    return dict(family=FONT_STACK.replace('"',''),color=INK,size=size)
