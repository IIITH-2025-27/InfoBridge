"""App-wide Streamlit CSS styles — dark & light themes."""

import streamlit as st

# ─── Dark Theme ───────────────────────────────────────────────────────────────
_DARK = """
<style>
/* ── Base ─────────────────────────────────────────────── */
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #1a1040 0%, #0d1117 60%) !important;
}
.main .block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:900px; }

/* ── Header ───────────────────────────────────────────── */
.ib-header {
    background: linear-gradient(135deg, #10153a 0%, #1e1050 40%, #0e1a3a 100%);
    border: 1px solid rgba(255,153,51,0.25);
    border-radius: 20px;
    padding: 2rem 2.5rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
    box-shadow: 0 0 60px rgba(255,153,51,0.08), 0 8px 32px rgba(0,0,0,0.5);
}
.ib-header::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px;
    background:radial-gradient(circle, rgba(255,153,51,0.15) 0%, transparent 70%);
    pointer-events:none;
}
.ib-header::after {
    content:''; position:absolute; bottom:-40px; left:30%;
    width:300px; height:100px;
    background:radial-gradient(ellipse, rgba(19,136,8,0.08) 0%, transparent 70%);
    pointer-events:none;
}
.ib-header-logo {
    font-size:2.4rem; font-weight:900;
    background: linear-gradient(135deg, #FF9933 0%, #FFD700 50%, #FF9933 100%);
    background-size:200% auto;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation: shimmer 4s linear infinite;
    display:inline-block; margin:0; letter-spacing:-1px;
}
.ib-header-subtitle { color:rgba(255,255,255,0.6); font-size:0.92rem; margin:0.3rem 0 1.2rem 0; }
.ib-pill { display:inline-flex; align-items:center; gap:0.3rem; padding:0.25rem 0.75rem;
    border-radius:20px; font-size:0.75rem; font-weight:600; margin-right:0.4rem; margin-bottom:0.3rem; }
.ib-pill-saffron { background:rgba(255,153,51,0.15); border:1px solid rgba(255,153,51,0.3); color:#FF9933; }
.ib-pill-green   { background:rgba(19,136,8,0.15);   border:1px solid rgba(19,136,8,0.35);   color:#4CAF50; }
.ib-pill-blue    { background:rgba(100,149,237,0.15); border:1px solid rgba(100,149,237,0.3); color:#90CAF9; }
.ib-header-services {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
    border-radius:12px; padding:0.75rem 1rem;
    font-size:0.78rem; color:rgba(255,255,255,0.5); line-height:2; white-space:nowrap;
}

/* ── Welcome ──────────────────────────────────────────── */
.ib-welcome-heading { font-size:1.25rem; font-weight:700; color:#ffffff; margin:0 0 0.25rem 0; }
.ib-welcome-sub { color:rgba(255,255,255,0.5); font-size:0.85rem; margin:0 0 1.25rem 0; }
.ib-service-grid {
    display:grid; grid-template-columns:repeat(5,1fr); gap:0.65rem; margin-bottom:1.5rem;
}
.ib-service-card {
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:1rem 0.5rem; text-align:center; transition:all 0.25s ease;
}
.ib-service-card:hover {
    background:rgba(255,153,51,0.07); border-color:rgba(255,153,51,0.35);
    transform:translateY(-3px); box-shadow:0 8px 20px rgba(255,153,51,0.1);
}
.ib-service-card .svc-icon { font-size:1.7rem; display:block; margin-bottom:0.45rem; }
.ib-service-card .svc-name { font-size:0.72rem; font-weight:600; color:rgba(255,255,255,0.8); line-height:1.3; }
.ib-examples-label {
    font-size:0.8rem; font-weight:600; color:rgba(255,255,255,0.4);
    text-transform:uppercase; letter-spacing:0.8px; margin-bottom:0.6rem;
}

/* ── Buttons ──────────────────────────────────────────── */
.stButton > button {
    background:rgba(255,255,255,0.04) !important; border:1px solid rgba(255,255,255,0.09) !important;
    color:rgba(255,255,255,0.7) !important; border-radius:10px !important;
    font-size:0.82rem !important; text-align:left !important;
    transition:all 0.2s ease !important; padding:0.55rem 0.9rem !important;
    white-space:normal !important; line-height:1.4 !important;
    height:auto !important; min-height:2.4rem !important;
}
.stButton > button:hover {
    background:rgba(255,153,51,0.09) !important; border-color:rgba(255,153,51,0.35) !important;
    color:#FFB347 !important; transform:translateX(4px) !important;
    box-shadow:0 4px 14px rgba(255,153,51,0.1) !important;
}
.stButton > button:active { transform:translateX(2px) !important; }

/* ── Chat Messages ────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background:rgba(22,27,42,0.7) !important; border:1px solid rgba(255,255,255,0.06) !important;
    border-radius:16px !important; backdrop-filter:blur(8px) !important; margin-bottom:0.8rem !important;
    transition:border-color 0.2s ease !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *,
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td,
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] div {
    color:rgba(255,255,255,0.92) !important;
}
[data-testid="stChatMessage"] a { color:#FFD700 !important; }
[data-testid="stChatMessage"] code { color:#FFB347 !important; background:rgba(255,153,51,0.12) !important; }
[data-testid="stChatMessage"]:hover { border-color:rgba(255,255,255,0.1) !important; }

/* ── Chat Input ───────────────────────────────────────── */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
.stChatFloatingInputContainer,
.stChatFloatingInputContainer > div,
div[class*="chatInputContainer"],
div[class*="stChatInput"],
div[class*="bottom"] {
    background:#0d1117 !important;
    border-top:1px solid rgba(255,153,51,0.08) !important;
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {
    background:rgba(22,27,42,0.9) !important; border:1px solid rgba(255,153,51,0.2) !important;
    border-radius:16px !important; transition:border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color:rgba(255,153,51,0.5) !important; box-shadow:0 0 0 3px rgba(255,153,51,0.08) !important;
}
[data-testid="stChatInput"] textarea { color:rgba(255,255,255,0.9) !important; background:transparent !important; font-size:0.95rem !important; }
[data-testid="stChatInput"] textarea::placeholder { color:rgba(255,255,255,0.3) !important; }

/* ── Expander ─────────────────────────────────────────── */
[data-testid="stExpander"] {
    background:rgba(22,27,42,0.5) !important; border:1px solid rgba(255,153,51,0.12) !important;
    border-radius:12px !important; overflow:hidden;
}
[data-testid="stExpander"] summary { color:rgba(255,255,255,0.7) !important; font-size:0.85rem !important; font-weight:500 !important; padding:0.6rem 1rem !important; }
[data-testid="stExpander"] summary:hover { color:#FFD700 !important; }
[data-testid="stExpander"] svg { fill:rgba(255,255,255,0.5) !important; }

/* ── Source Cards ─────────────────────────────────────── */
.ib-source-card {
    background:rgba(255,153,51,0.05); border:1px solid rgba(255,153,51,0.12);
    border-left:3px solid rgba(255,153,51,0.6); border-radius:0 10px 10px 0;
    padding:0.55rem 1rem; margin:0.3rem 0; font-size:0.81rem;
    color:rgba(255,255,255,0.75); line-height:1.5; transition:background 0.2s;
}
.ib-source-card:hover { background:rgba(255,153,51,0.1); }
.ib-source-card strong { color:#FFD700; font-weight:600; }
.ib-source-meta { color:rgba(255,255,255,0.45); font-size:0.76rem; }

/* ── Sidebar ──────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg, #0b0f1e 0%, #0f1428 100%) !important;
    border-right:1px solid rgba(255,153,51,0.1) !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * { color:rgba(255,255,255,0.8) !important; }
section[data-testid="stSidebar"] .stCaption p { color:rgba(255,255,255,0.4) !important; font-size:0.75rem !important; }
section[data-testid="stSidebar"] label { color:rgba(255,255,255,0.75) !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background:rgba(255,255,255,0.05) !important; border:1px solid rgba(255,153,51,0.18) !important;
    border-radius:10px !important; color:rgba(255,255,255,0.85) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill:rgba(255,255,255,0.5) !important; }
section[data-testid="stSidebar"] [data-baseweb="popover"] { background:#161b2e !important; border:1px solid rgba(255,153,51,0.15) !important; }
section[data-testid="stSidebar"] [role="option"] { color:rgba(255,255,255,0.8) !important; background:transparent !important; }
section[data-testid="stSidebar"] [role="option"]:hover,
section[data-testid="stSidebar"] [aria-selected="true"] { background:rgba(255,153,51,0.12) !important; color:#FF9933 !important; }
section[data-testid="stSidebar"] .stButton > button {
    background:rgba(239,83,80,0.08) !important; color:rgba(239,83,80,0.9) !important;
    border:1px solid rgba(239,83,80,0.2) !important; border-radius:10px !important;
    font-weight:500 !important; transform:none !important; transition:all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background:rgba(239,83,80,0.18) !important; border-color:rgba(239,83,80,0.45) !important;
    color:#ef5350 !important; transform:none !important; box-shadow:0 4px 12px rgba(239,83,80,0.12) !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] {
    display: flex !important;
    gap: 0.5rem !important;
    flex-wrap: nowrap !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] > label {
    flex: 1 1 0 !important;
    padding: 0.6rem 0.75rem !important;
    margin: 0 !important;
    border: 1px solid rgba(255,153,51,0.18) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.04) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: rgba(255,153,51,0.1) !important;
    border-color: rgba(255,153,51,0.38) !important;
    transform: translateY(-1px) !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, rgba(255,153,51,0.2), rgba(255,215,0,0.12)) !important;
    border-color: rgba(255,153,51,0.55) !important;
    box-shadow: 0 4px 14px rgba(255,153,51,0.14) !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] > label > div {
    color: rgba(255,255,255,0.88) !important;
    font-weight: 700 !important;
}

/* ── Status ───────────────────────────────────────────── */
.status-indicator { display:inline-flex; align-items:center; gap:0.4rem; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.78rem; font-weight:600; }
.status-ready { background:rgba(76,175,80,0.12); color:#66bb6a; border:1px solid rgba(76,175,80,0.22); }
.status-error { background:rgba(239,83,80,0.12); color:#ef5350; border:1px solid rgba(239,83,80,0.22); }

.ib-strip-caption {
    margin-top: 0.3rem;
    text-align: center;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.65);
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    height: 7.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: auto;
    margin-right: auto;
}

/* ── Hide native sidebar toggle controls (use custom strip toggle only) ──── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarNavCollapseButton"],
button[data-testid="stBaseButton-headerNoPadding"][kind="headerNoPadding"],
button[data-testid="baseButton-header"][aria-label*="sidebar" i],
button[aria-label*="open sidebar" i],
button[aria-label*="show sidebar" i],
button[aria-label*="collapse sidebar" i],
button[aria-label*="expand sidebar" i],
button[aria-label*="close sidebar" i],
button[title*="sidebar" i] {
    display: none !important;
}

/* ── Misc ─────────────────────────────────────────────── */
hr { border-color:rgba(255,255,255,0.07) !important; }
.stSpinner > div { border-top-color:#FF9933 !important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background:rgba(255,153,51,0.25); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:rgba(255,153,51,0.45); }
@keyframes shimmer { 0%{ background-position:0% center; } 100%{ background-position:200% center; } }
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
header { background:transparent !important; }
[data-testid="stToolbar"] {
    visibility: visible !important;
    display: block !important;
    opacity: 1 !important;
}
</style>
"""

# ─── Light Theme ──────────────────────────────────────────────────────────────
_LIGHT = """
<style>
/* ── Base ─────────────────────────────────────────────── */
.stApp { background:#edf0f7 !important; }
.main .block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:900px; }

/* ── Header ───────────────────────────────────────────── */
.ib-header {
    background:linear-gradient(135deg, #fffdf8 0%, #fff8ee 50%, #f0f8ff 100%);
    border:1px solid rgba(200,94,0,0.2);
    border-radius:20px; padding:2rem 2.5rem 1.8rem; margin-bottom:1.5rem;
    position:relative; overflow:hidden;
    box-shadow:0 4px 24px rgba(0,0,0,0.07), 0 1px 3px rgba(0,0,0,0.05);
}
.ib-header::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px;
    background:radial-gradient(circle, rgba(255,153,51,0.12) 0%, transparent 70%);
    pointer-events:none;
}
.ib-header::after {
    content:''; position:absolute; bottom:-40px; left:30%;
    width:300px; height:100px;
    background:radial-gradient(ellipse, rgba(19,136,8,0.06) 0%, transparent 70%);
    pointer-events:none;
}
.ib-header-logo {
    font-size:2.4rem; font-weight:900;
    background:linear-gradient(135deg, #c85c00 0%, #e8820a 50%, #c85c00 100%);
    background-size:200% auto;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation:shimmer 4s linear infinite;
    display:inline-block; margin:0; letter-spacing:-1px;
}
.ib-header-subtitle { color:rgba(30,41,59,0.6); font-size:0.92rem; margin:0.3rem 0 1.2rem 0; }
.ib-pill { display:inline-flex; align-items:center; gap:0.3rem; padding:0.25rem 0.75rem;
    border-radius:20px; font-size:0.75rem; font-weight:600; margin-right:0.4rem; margin-bottom:0.3rem; }
.ib-pill-saffron { background:rgba(200,92,0,0.1);  border:1px solid rgba(200,92,0,0.25);  color:#c85c00; }
.ib-pill-green   { background:rgba(46,125,50,0.1);  border:1px solid rgba(46,125,50,0.25);  color:#2e7d32; }
.ib-pill-blue    { background:rgba(25,118,210,0.1); border:1px solid rgba(25,118,210,0.25); color:#1565c0; }
.ib-header-services {
    background:rgba(0,0,0,0.04); border:1px solid rgba(0,0,0,0.08);
    border-radius:12px; padding:0.75rem 1rem;
    font-size:0.78rem; color:rgba(30,41,59,0.55); line-height:2; white-space:nowrap;
}

/* ── Welcome ──────────────────────────────────────────── */
.ib-welcome-heading { font-size:1.25rem; font-weight:700; color:#1e293b; margin:0 0 0.25rem 0; }
.ib-welcome-sub { color:rgba(30,41,59,0.5); font-size:0.85rem; margin:0 0 1.25rem 0; }
.ib-service-grid {
    display:grid; grid-template-columns:repeat(5,1fr); gap:0.65rem; margin-bottom:1.5rem;
}
.ib-service-card {
    background:#ffffff; border:1px solid rgba(0,0,0,0.08);
    border-radius:14px; padding:1rem 0.5rem; text-align:center;
    transition:all 0.25s ease; box-shadow:0 1px 4px rgba(0,0,0,0.05);
}
.ib-service-card:hover {
    background:#fff8f0; border-color:rgba(200,92,0,0.3);
    transform:translateY(-3px); box-shadow:0 8px 20px rgba(200,92,0,0.1);
}
.ib-service-card .svc-icon { font-size:1.7rem; display:block; margin-bottom:0.45rem; }
.ib-service-card .svc-name { font-size:0.72rem; font-weight:600; color:#334155; line-height:1.3; }
.ib-examples-label {
    font-size:0.8rem; font-weight:600; color:rgba(30,41,59,0.45);
    text-transform:uppercase; letter-spacing:0.8px; margin-bottom:0.6rem;
}

/* ── Buttons ──────────────────────────────────────────── */
.stButton > button {
    background:#ffffff !important; border:1px solid rgba(0,0,0,0.1) !important;
    color:#334155 !important; border-radius:10px !important;
    font-size:0.82rem !important; text-align:left !important;
    transition:all 0.2s ease !important; padding:0.55rem 0.9rem !important;
    white-space:normal !important; line-height:1.4 !important;
    height:auto !important; min-height:2.4rem !important;
    box-shadow:0 1px 3px rgba(0,0,0,0.06) !important;
}
.stButton > button:hover {
    background:#fff5eb !important; border-color:rgba(200,92,0,0.35) !important;
    color:#c85c00 !important; transform:translateX(4px) !important;
    box-shadow:0 4px 14px rgba(200,92,0,0.1) !important;
}
.stButton > button:active { transform:translateX(2px) !important; }

/* ── Chat Messages ────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background:#ffffff !important; border:1px solid rgba(0,0,0,0.07) !important;
    border-radius:16px !important; margin-bottom:0.8rem !important;
    box-shadow:0 2px 8px rgba(0,0,0,0.05) !important;
    transition:box-shadow 0.2s ease !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *,
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td,
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] div {
    color:#1e293b !important;
    cursor:text !important;
    user-select:text !important;
    -webkit-user-select:text !important;
}
[data-testid="stChatMessage"] a { color:#c85c00 !important; cursor:pointer !important; }
[data-testid="stChatMessage"] code { color:#7c2d12 !important; background:rgba(200,92,0,0.08) !important; }
[data-testid="stChatMessage"]:hover { box-shadow:0 4px 16px rgba(0,0,0,0.08) !important; }
/* Text selection highlight in light mode */
[data-testid="stChatMessage"] ::selection {
    background:rgba(200,92,0,0.18) !important;
    color:#1e293b !important;
}

/* ── Chat Input ───────────────────────────────────────── */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
.stChatFloatingInputContainer,
.stChatFloatingInputContainer > div,
div[class*="chatInputContainer"],
div[class*="stChatInput"],
div[class*="bottom"] {
    background:#edf0f7 !important;
    border-top:1px solid rgba(0,0,0,0.06) !important;
}
/* Single border on outer wrapper only — mirrors dark mode approach */
[data-testid="stChatInput"] > div {
    background:#ffffff !important;
    border:1px solid rgba(200,92,0,0.25) !important;
    border-radius:16px !important;
    box-shadow:0 2px 8px rgba(0,0,0,0.05) !important;
    transition:border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color:rgba(200,92,0,0.55) !important;
    box-shadow:0 0 0 3px rgba(200,92,0,0.08) !important;
}
/* Inner elements: no border, transparent background */
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] [data-baseweb="base-input"],
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"] > div {
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
}
/* Textarea: text colour, no border */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] [contenteditable="true"],
[data-testid="stChatInput"] [data-baseweb="base-input"] textarea,
[data-testid="stChatInput"] [data-baseweb="textarea"] textarea,
.stChatFloatingInputContainer textarea,
.stChatFloatingInputContainer [contenteditable="true"] {
    color:#1e293b !important;
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
    font-size:0.95rem !important;
    caret-color:#c85c00 !important;
    cursor:text !important;
}
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] [data-baseweb="base-input"] textarea::placeholder,
[data-testid="stChatInput"] [data-baseweb="textarea"] textarea::placeholder,
.stChatFloatingInputContainer textarea::placeholder {
    color:rgba(30,41,59,0.35) !important;
}
[data-testid="stChatInput"] textarea::selection {
    background:rgba(200,92,0,0.18) !important;
    color:#1e293b !important;
}

/* ── Expander ─────────────────────────────────────────── */
[data-testid="stExpander"] {
    background:#ffffff !important; border:1px solid rgba(0,0,0,0.08) !important;
    border-radius:12px !important; overflow:hidden;
    box-shadow:0 1px 4px rgba(0,0,0,0.05) !important;
}
[data-testid="stExpander"] summary { color:#334155 !important; font-size:0.85rem !important; font-weight:500 !important; padding:0.6rem 1rem !important; }
[data-testid="stExpander"] summary:hover { color:#c85c00 !important; }
[data-testid="stExpander"] svg { fill:rgba(51,65,85,0.5) !important; }

/* ── Source Cards ─────────────────────────────────────── */
.ib-source-card {
    background:rgba(200,92,0,0.04); border:1px solid rgba(200,92,0,0.12);
    border-left:3px solid rgba(200,92,0,0.55); border-radius:0 10px 10px 0;
    padding:0.55rem 1rem; margin:0.3rem 0; font-size:0.81rem;
    color:#334155; line-height:1.5; transition:background 0.2s;
}
.ib-source-card:hover { background:rgba(200,92,0,0.08); }
.ib-source-card strong { color:#c85c00; font-weight:600; }
.ib-source-meta { color:rgba(51,65,85,0.55); font-size:0.76rem; }

/* ── Sidebar ──────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
    border-right:1px solid rgba(0,0,0,0.08) !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * { color:#334155 !important; }
section[data-testid="stSidebar"] .stCaption p { color:rgba(51,65,85,0.5) !important; font-size:0.75rem !important; }
section[data-testid="stSidebar"] label { color:#475569 !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background:#ffffff !important; border:1px solid rgba(0,0,0,0.12) !important;
    border-radius:10px !important; color:#334155 !important;
    box-shadow:0 1px 3px rgba(0,0,0,0.06) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill:#64748b !important; }
section[data-testid="stSidebar"] [data-baseweb="popover"] { background:#ffffff !important; border:1px solid rgba(0,0,0,0.1) !important; box-shadow:0 4px 16px rgba(0,0,0,0.1) !important; }
section[data-testid="stSidebar"] [role="option"] { color:#334155 !important; background:transparent !important; }
section[data-testid="stSidebar"] [role="option"]:hover,
section[data-testid="stSidebar"] [aria-selected="true"] { background:rgba(200,92,0,0.08) !important; color:#c85c00 !important; }
section[data-testid="stSidebar"] .stButton > button {
    background:rgba(239,83,80,0.06) !important; color:#c0392b !important;
    border:1px solid rgba(239,83,80,0.2) !important; border-radius:10px !important;
    font-weight:500 !important; transform:none !important; transition:all 0.2s ease !important;
    box-shadow:none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background:rgba(239,83,80,0.12) !important; border-color:rgba(239,83,80,0.4) !important;
    color:#c0392b !important; transform:none !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] {
    display: flex !important;
    gap: 0.5rem !important;
    flex-wrap: nowrap !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] > label {
    flex: 1 1 0 !important;
    padding: 0.6rem 0.75rem !important;
    margin: 0 !important;
    border: 1px solid rgba(200,92,0,0.18) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.88) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: #fff7ee !important;
    border-color: rgba(200,92,0,0.38) !important;
    transform: translateY(-1px) !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, rgba(255,153,51,0.16), rgba(255,215,0,0.08)) !important;
    border-color: rgba(200,92,0,0.55) !important;
    box-shadow: 0 4px 14px rgba(200,92,0,0.12) !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] > label > div {
    color: #334155 !important;
    font-weight: 700 !important;
}

/* ── Status ───────────────────────────────────────────── */
.status-indicator { display:inline-flex; align-items:center; gap:0.4rem; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.78rem; font-weight:600; }
.status-ready { background:rgba(46,125,50,0.1); color:#2e7d32; border:1px solid rgba(46,125,50,0.2); }
.status-error { background:rgba(198,40,40,0.1); color:#c62828; border:1px solid rgba(198,40,40,0.2); }

.ib-strip-caption {
    margin-top: 0.3rem;
    text-align: center;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(51,65,85,0.72);
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    height: 7.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: auto;
    margin-right: auto;
}

/* ── Hide native sidebar toggle controls (use custom strip toggle only) ──── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarNavCollapseButton"],
button[data-testid="stBaseButton-headerNoPadding"][kind="headerNoPadding"],
button[data-testid="baseButton-header"][aria-label*="sidebar" i],
button[aria-label*="open sidebar" i],
button[aria-label*="show sidebar" i],
button[aria-label*="collapse sidebar" i],
button[aria-label*="expand sidebar" i],
button[aria-label*="close sidebar" i],
button[title*="sidebar" i] {
    display: none !important;
}

/* ── Misc ─────────────────────────────────────────────── */
hr { border-color:rgba(0,0,0,0.08) !important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:rgba(0,0,0,0.03); }
::-webkit-scrollbar-thumb { background:rgba(200,92,0,0.25); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:rgba(200,92,0,0.45); }
@keyframes shimmer { 0%{ background-position:0% center; } 100%{ background-position:200% center; } }
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
header { background:transparent !important; }
[data-testid="stToolbar"] {
    visibility: visible !important;
    display: block !important;
    opacity: 1 !important;
}
</style>
"""


def apply_styles(theme: str = "dark") -> None:
    st.markdown(_LIGHT if theme == "light" else _DARK, unsafe_allow_html=True)
    sidebar_open = st.session_state.get("sidebar_open", True)
    strip_bg = "#0f1428" if theme == "dark" else "#f1f5f9"
    strip_border = "rgba(255,153,51,0.22)" if theme == "dark" else "rgba(0,0,0,0.16)"
    toggle_bg = "rgba(255,153,51,0.16)" if theme == "dark" else "rgba(200,92,0,0.12)"
    toggle_text = "#FFD700" if theme == "dark" else "#c85c00"

    sidebar_layout_css = f"""
    <style>
    section[data-testid="stSidebar"] {{
        width: {'18rem' if sidebar_open else '3.6rem'} !important;
        min-width: {'18rem' if sidebar_open else '3.6rem'} !important;
        max-width: {'18rem' if sidebar_open else '3.6rem'} !important;
        flex: 0 0 {'18rem' if sidebar_open else '3.6rem'} !important;
        left: 0 !important;
        transform: translateX(0) !important;
        background: {strip_bg} !important;
        border-right: 1px solid {strip_border} !important;
    }}

    section[data-testid="stSidebar"] > div:first-child,
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding-top: 0.6rem !important;
        padding-left: {'0.75rem' if sidebar_open else '0.32rem'} !important;
        padding-right: {'0.75rem' if sidebar_open else '0.32rem'} !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
        justify-content: center !important;
        text-align: center !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] > button {{
        background: {('#2a3045' if theme == 'dark' else '#fff1df')} !important;
        color: {('#FFD700' if theme == 'dark' else '#9a4300')} !important;
        border: 1px solid {('#9ca3af' if theme == 'dark' else '#c85c00')} !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
        font-weight: 800 !important;
        letter-spacing: 0.01em !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] > button:hover {{
        background: {('#39405a' if theme == 'dark' else '#ffe7cc')} !important;
        color: {('#FFE083' if theme == 'dark' else '#7f3600')} !important;
        border-color: {('#c4c9d4' if theme == 'dark' else '#a34700')} !important;
        transform: none !important;
    }}

    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarNavCollapseButton"],
    button[data-testid="stBaseButton-headerNoPadding"][kind="headerNoPadding"],
    button[data-testid="stBaseButton-headerNoPadding"],
    button[data-testid="baseButton-header"][aria-label*="sidebar" i],
    button[aria-label*="open sidebar" i],
    button[aria-label*="show sidebar" i],
    button[aria-label*="collapse sidebar" i],
    button[aria-label*="expand sidebar" i],
    button[aria-label*="close sidebar" i],
    button[title*="sidebar" i] {{
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }}
    </style>
    """
    st.markdown(sidebar_layout_css, unsafe_allow_html=True)
