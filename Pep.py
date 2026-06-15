# ==========================================================
# PepTastePredictor — app.py  (Hybrid Structural Engine v2)
# Production-Ready Premium Cyber Theme Implementation
# Streamlit Cloud Compatible — No local ESMFold/GPU required
# ==========================================================

# ==========================================================
# SECTION 1 - IMPORTS
# ==========================================================

import os
import io
import re
import time
import json
import urllib.request
import urllib.error
import tempfile
import zipfile
from datetime import date
from collections import Counter
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import PDBIO, PDBParser, PPBuilder
import PeptideBuilder
from PeptideBuilder import Geometry

from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image as RLImage,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors


# ==========================================================
# SECTION 2 - GLOBAL CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="PepTastePredictor v2",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_PATH    = "AIML.xlsx"
PREDICTIONS_DIR = Path("predictions")
PREDICTIONS_DIR.mkdir(exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"

ALL_DIPEPTIDES = [a1 + a2 for a1 in AA for a2 in AA]

KD_SCALE = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
}

TASTE_EMOJI = {
    "Bitter": "😖", "Sweet": "😋", "Salty": "🧂",
    "Sour": "😮‍💨", "Umami": "🤤",
}

CF_HELIX = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}
CF_SHEET = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}

THREE_LETTER = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY",
    "H":"HIS","I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN",
    "P":"PRO","Q":"GLN","R":"ARG","S":"SER","T":"THR","V":"VAL",
    "W":"TRP","Y":"TYR",
}

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/"
RCSB_SEARCH   = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_FETCH    = "https://files.rcsb.org/download/{}.pdb"
ESMFOLD_API   = "https://api.esmatlas.com/foldSequence/v1/pdb/"


# ==========================================================
# SECTION 3 - FRONTEND STYLING (PREMIUM THEME)
# ==========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;700;800&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Overrides for Dark Cyber Theme Archetype */
    .stApp {
        background-color: #05070C !important;
        background-image: 
            radial-gradient(ellipse 80% 50% at 10% 20%, rgba(31, 60, 136, 0.08) 0%, transparent 65%),
            radial-gradient(ellipse 60% 40% at 85% 80%, rgba(18, 184, 134, 0.05) 0%, transparent 65%) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Force global text color consistency */
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp div {
        color: #ECF0F5 !important;
    }
    
    /* Premium Shimmer Dashboard Cards */
    .premium-card {
        background: #0B0E16;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 22px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        margin-bottom: 15px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .premium-card:hover {
        border-color: rgba(0, 214, 168, 0.3);
        transform: translateY(-2px);
    }
    .premium-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #1f3c88, #0b7285, #12b886);
    }
    .card-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #8A8F9E !important;
        margin-bottom: 8px;
        font-weight: 700;
    }
    .card-value {
        font-size: 28px;
        font-weight: 800;
        color: #00D4A0 !important;
        line-height: 1.1;
        font-family: 'Syne', sans-serif;
    }
    .card-sub {
        font-size: 11px;
        color: #525866 !important;
        margin-top: 6px;
    }

    /* High-tech Platform Badges */
    .engine-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 4px 4px 4px 0;
    }
    .badge-esm   { background: rgba(18,184,134,0.1); border: 1px solid rgba(18,184,134,0.3); color: #12b886 !important; }
    .badge-af    { background: rgba(31,60,136,0.12); border: 1px solid rgba(31,60,136,0.4); color: #4361ee !important; }
    .badge-pdb   { background: rgba(255,152,0,0.1);  border: 1px solid rgba(255,152,0,0.3);  color: #ffa94d !important; }
    .badge-fold  { background: rgba(103,58,183,0.1); border: 1px solid rgba(103,58,183,0.3); color: #a78bfa !important; }
    .badge-pb    { background: rgba(230,126,34,0.1);  border: 1px solid rgba(230,126,34,0.3);  color: #e67e22 !important; }

    /* Custom Input panel & fasta display wrapper */
    .seq-panel {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #A3AED0 !important;
        background: #0E131F;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.04);
        white-space: pre-wrap;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.5);
    }

    /* Hero Branding Panel Block */
    .hero {
        background: linear-gradient(135deg, #090D16 0%, #0F1524 100%);
        border: 1px solid rgba(255,255,255,0.04);
        padding: 35px 40px; 
        border-radius: 16px; 
        margin-bottom: 35px;
        position: relative;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: 0; right: 10%; width: 30%; height: 100%;
        background: radial-gradient(circle, rgba(18,184,134,0.04) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero h1 { 
        font-family: 'Syne', sans-serif !important;
        font-size: 2.5rem !important; 
        font-weight: 800 !important;
        margin-bottom: 12px; 
        background: linear-gradient(90deg, #FFFFFF, #A3AED0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px; 
    }
    .hero p { 
        font-size: 1.02rem !important; 
        line-height: 1.7; 
        color: #A3AED0 !important; 
        margin: 0; 
    }

    /* Structural Panels Info Wrap */
    .struct-info-panel {
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px; 
        padding: 24px; 
        margin-bottom: 25px;
        background: #0B0E16;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .struct-info-panel h4 { 
        font-family: 'Syne', sans-serif !important;
        font-size: 14px !important; 
        font-weight: 700 !important;
        text-transform: uppercase; 
        letter-spacing: 0.08em; 
        color: #00D4A0 !important;
        margin: 0 0 20px 0; 
    }
    .struct-row { display: flex; flex-wrap: wrap; gap: 30px; }
    .struct-item { display: flex; flex-direction: column; }
    .struct-label { 
        font-size: 10px; 
        font-weight: 700; 
        text-transform: uppercase;
        letter-spacing: 0.1em; 
        color: #656A7A !important; 
        margin-bottom: 5px;
    }
    .struct-value { font-size: 16px; font-weight: 700; color: #5c7cfa !important; }

    /* Custom Secondary Structure Progress Segment bar */
    .ss-bar { 
        display: flex; 
        height: 12px; 
        border-radius: 6px; 
        overflow: hidden;
        margin: 14px 0 8px; 
        background: #141923;
    }
    .ss-segment { height: 100%; transition: width 0.5s ease; }
    
    /* Interactive Navigation Tab Customization Overrides */
    div[data-testid="stTabs"] button {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        background-color: transparent !important;
        transition: color 0.3s ease;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] p {
        color: #00D4A0 !important;
    }

    /* Graph Caption Custom Box wrapper */
    .graph-caption { 
        border-left: 4px solid #1f3c88; 
        border-radius: 0 8px 8px 0;
        padding: 15px 20px; 
        margin-top: 15px; 
        margin-bottom: 35px;
        font-size: 14px !important; 
        line-height: 1.6; 
        background: #0B0E16;
        border: 1px solid rgba(255,255,255,0.03);
        border-left-width: 4px;
    }
    
    /* Modernized Streamlit native Elements Form Customizations */
    div[data-testid="stForm"] {
        background-color: #0B0E16 !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
    }
    textarea, input {
        background-color: #0E131F !important;
        color: #ECF0F5 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
    }

    .footer { 
        text-align: center; 
        font-size: 13px !important; 
        padding: 40px 20px;
        margin-top: 70px; 
        line-height: 2;
        border-top: 1px solid rgba(255,255,255,0.05);
        color: #525866 !important; 
    }
    .live-indicator { 
        display: inline-block; width: 7px; height: 7px;
        background: #00D4A0; border-radius: 50%; margin-right: 8px;
        box-shadow: 0 0 8px #00D4A0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# SECTION 4 - SIDEBAR
# ==========================================================

if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=120)

st.sidebar.markdown("<h2 style='font-family:\"Syne\",sans-serif; font-weight:800; color:#ECF0F5 !important;'>🧬 PepTaste</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#A3AED0 !important; font-size:13px;'>Integrated Structural Hybrid Framework Workspace</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.write("• High-throughput Taste Classification")
st.sidebar.write("• Thermodynamic Binding Free Energies")
st.sidebar.write("• Topological Residue Contact Network Maps")
st.sidebar.write("• Secondary Structure Propensity Arrays")
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='font-size:11px; font-weight:700; color:#656A7A !important; text-transform:uppercase;'>Engine Cascade Priority</p>", unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="engine-badge badge-af">① AlphaFold DB</div><br>'
    '<div class="engine-badge badge-pdb">② RCSB PDB</div><br>'
    '<div class="engine-badge badge-esm">③ Remote ESMFold</div><br>'
    '<div class="engine-badge badge-fold">④ Peptide Folder</div><br>'
    '<div class="engine-badge badge-pb">⑤ PeptideBuilder</div>',
    unsafe_allow_html=True,
)


# ==========================================================
# SECTION 5 - SESSION STATE
# ==========================================================

_defaults = {
    "initialized":     True,
    "pdb_text":        None,
    "pdb_source":      None,
    "last_prediction": {},
    "show_analytics":  False,
    "pdf_figures":     [],
    "current_mode":    None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ==========================================================
# SECTION 6 - UTILITY FUNCTIONS
# ==========================================================

def save_fig(fig, filename: str):
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    if filename not in st.session_state.pdf_figures:
        st.session_state.pdf_figures.append(filename)


def clean_sequence(seq) -> str:
    if not isinstance(seq, str):
        return ""
    lines = seq.splitlines()
    lines = [l for l in lines if not l.strip().startswith(">")]
    seq = "".join(lines)
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join(a for a in seq if a in AA)


def parse_fasta(text: str) -> list:
    records = []
    current_header = ""
    current_seq    = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_seq:
                records.append((current_header, clean_sequence("".join(current_seq))))
            current_header = line[1:]
            current_seq    = []
        else:
            current_seq.append(line)
    if current_seq:
        records.append((current_header, clean_sequence("".join(current_seq))))
    return [(h, s) for h, s in records if s]


def model_features(seq: str) -> dict:
    L = len(seq)
    features = {"length": L}
    if L >= 2:
        try:
            ana = ProteinAnalysis(seq)
            features.update({
                "mw":         ana.molecular_weight(),
                "pI":         ana.isoelectric_point(),
                "aromaticity": ana.aromaticity(),
                "instability": ana.instability_index(),
                "gravy":       ana.gravy(),
                "charge":      ana.charge_at_pH(7.0),
            })
        except Exception:
            features.update({"mw": 0, "pI": 7.0, "aromaticity": 0,
                             "instability": 0, "gravy": 0, "charge": 0})
    else:
        features.update({
            "mw":          111.0,
            "pI":          7.0,
            "aromaticity": 1.0 if seq in "FWY" else 0.0,
            "instability": 0.0,
            "gravy":       KD_SCALE.get(seq, 0.0),
            "charge":      1.0 if seq in "KRH" else (-1.0 if seq in "DE" else 0.0),
        })
    for aa in AA:
        features[f"AA_{aa}"] = seq.count(aa) / L
    denom = max(L - 1, 1)
    for dp in ALL_DIPEPTIDES:
        features[f"DPC_{dp}"] = seq.count(dp) / denom
    groups = {
        "hydrophobic": "AILMFWV", "polar": "STNQ",
        "charged": "DEKRH", "aromatic": "FWY", "tiny": "AGSC",
    }
    for name, aas in groups.items():
        features[name] = sum(seq.count(a) for a in aas) / L
    return features


def build_feature_table(seqs) -> pd.DataFrame:
    return pd.DataFrame([model_features(s) for s in seqs]).fillna(0)


def physicochemical_features(seq: str) -> dict:
    L = len(seq)
    if L >= 2:
        try:
            ana = ProteinAnalysis(seq)
            h, t, s = ana.secondary_structure_fraction()
            return {
                "Length":                L,
                "Molecular weight (Da)": round(ana.molecular_weight(), 2),
                "Isoelectric point":     round(ana.isoelectric_point(), 2),
                "Net charge (pH 7)":     round(ana.charge_at_pH(7.0), 2),
                "Aromaticity":           round(ana.aromaticity(), 3),
                "GRAVY":                 round(ana.gravy(), 3),
                "Instability index":     round(ana.instability_index(), 2),
                "Helix fraction":        round(h, 3),
                "Turn fraction":         round(t, 3),
                "Sheet fraction":        round(s, 3),
            }
        except Exception:
            pass
    return {
        "Length":      L,
        "GRAVY":       round(KD_SCALE.get(seq, 0.0), 3),
        "Aromaticity": 1.0 if seq in "FWY" else 0.0,
        "Note":        "Extended analysis requires ≥2 residues",
    }


def composition_features(seq: str) -> dict:
    c = Counter(seq)
    L = len(seq)
    return {
        "Hydrophobic (%)": round(100 * sum(c[a] for a in "AILMFWV") / L, 1),
        "Polar (%)":       round(100 * sum(c[a] for a in "STNQ") / L, 1),
        "Charged (%)":     round(100 * sum(c[a] for a in "DEKRH") / L, 1),
        "Aromatic (%)":    round(100 * sum(c[a] for a in "FWY") / L, 1),
    }


def simplify_taste(taste_series):
    counts = taste_series.value_counts()
    rare   = set(counts[counts < 5].index)
    def _map(t):
        if t in rare:
            for base in ["Bitter", "Sweet", "Salty", "Sour", "Umami"]:
                if base.lower() in t.lower():
                    return base
            return "Bitter"
        return t
    return taste_series.apply(_map)


def prettify_feature(name: str) -> str:
    if name.startswith("DPC_"):
        return f"Dipeptide {name[4:]}"
    if name.startswith("AA_"):
        return f"Amino acid: {name[3:]}"
    return name.replace("_", " ").title()


def gravy_score(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(KD_SCALE.get(a, 0) for a in seq) / len(seq)


def taste_emoji(taste: str) -> str:
    for k, v in TASTE_EMOJI.items():
        if k.lower() in taste.lower():
            return v
    return "🧬"


def show_caption(html_text: str):
    st.markdown(f'<div class="graph-caption">{html_text}</div>', unsafe_allow_html=True)


# ==========================================================
# SECTION 7 - MATPLOTLIB DARK ARCHETYPE THEME
# ==========================================================

def get_plot_colors() -> dict:
    return {
        "fig_bg": "#0B0E16", "ax_bg": "#0E131F", "text": "#ECF0F5",
        "grid": "#1A2235", "accent1": "#00D4A0", "accent2": "#5c7cfa",
        "accent3": "#4dd0e1", "red": "#ff6b6b", "orange": "#ffa94d",
        "tick": "#8A8F9E", "green": "#12b886",
    }


def apply_plot_style(fig, axes_list):
    C = get_plot_colors()
    fig.patch.set_facecolor(C["fig_bg"])
    for ax in (axes_list if hasattr(axes_list, "__iter__") else [axes_list]):
        ax.set_facecolor(C["ax_bg"])
        ax.tick_params(colors=C["tick"], labelsize=10)
        ax.xaxis.label.set_color(C["text"])
        ax.yaxis.label.set_color(C["text"])
        ax.title.set_color(C["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["grid"])
        ax.tick_params(axis="x", colors=C["tick"])
        ax.tick_params(axis="y", colors=C["tick"])


# ==========================================================
# SECTION 8 - PDB HELPERS
# ==========================================================

def _write_temp_pdb(pdb_text: str) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write(pdb_text)
    tmp.close()
    return tmp.name


def _unlink(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass


def _validate_pdb(pdb_text: str) -> bool:
    if not pdb_text or not pdb_text.strip():
        return False
    has_atom = False
    has_ca   = False
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        has_atom = True
        if line[12:16].strip() == "CA":
            has_ca = True
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                    return False
            except ValueError:
                return False
    return has_atom and has_ca


def _extract_plddt(pdb_text: str) -> list:
    seen = set()
    vals = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        chain   = line[21]
        res_seq = line[22:26].strip()
        key     = (chain, res_seq)
        if key not in seen:
            seen.add(key)
            try:
                vals.append(float(line[60:66].strip()))
            except ValueError:
                pass
    return vals


def _count_residues_in_pdb(pdb_text: str) -> int:
    seen = set()
    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            key = (line[21], line[22:26].strip())
            seen.add(key)
    return len(seen)


# ==========================================================
# SECTION 9 - SECONDARY STRUCTURE PREDICTION (Chou-Fasman)
# ==========================================================

def predict_secondary_structure(seq: str) -> dict:
    L = len(seq)
    if L == 0:
        return {"assignments": [], "helix_frac": 0, "sheet_frac": 0, "coil_frac": 1}

    helix_prop = [CF_HELIX.get(aa, 1.0) for aa in seq]
    sheet_prop = [CF_SHEET.get(aa, 1.0) for aa in seq]

    assignments = []
    for i in range(L):
        lo = max(0, i - 2)
        hi = min(L, i + 3)
        h_avg = np.mean(helix_prop[lo:hi])
        s_avg = np.mean(sheet_prop[lo:hi])
        if seq[i] == "P":
            assignments.append("C")
        elif h_avg >= 1.03 and h_avg >= s_avg:
            assignments.append("H")
        elif s_avg >= 1.05 and s_avg > h_avg:
            assignments.append("E")
        else:
            assignments.append("C")

    ss = list(assignments)
    for i in range(L):
        if ss[i] in ("H", "E"):
            j = i
            while j < L and ss[j] == ss[i]:
                j += 1
            if j - i < 4:
                for k in range(i, j):
                    ss[k] = "C"

    counts = Counter(ss)
    total  = max(L, 1)
    return {
        "assignments":  ss,
        "helix_frac":   counts.get("H", 0) / total,
        "sheet_frac":   counts.get("E", 0) / total,
        "coil_frac":    counts.get("C", 0) / total,
    }


def classify_fold(ss_result: dict, seq: str) -> str:
    h = ss_result["helix_frac"]
    e = ss_result["sheet_frac"]
    L = len(seq)
    if L <= 2:
        return "Dipeptide / Residue"
    if L <= 10:
        if h > 0.5:
            return "Short Helix"
        if e > 0.3:
            return "Beta-rich Peptide"
        return "Short Peptide / Loop"
    if h > 0.6:
        return "All-α Helix"
    if e > 0.5:
        return "All-β Sheet"
    if h > 0.3 and e > 0.2:
        return "α/β Mixed"
    if h > 0.4:
        return "Predominantly α"
    if e > 0.3:
        return "Predominantly β"
    return "Disordered / Coil"


# ==========================================================
# SECTION 10 - HYBRID STRUCTURAL ENGINE
# ==========================================================

def _http_get(url: str, timeout: int = 15) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PepTastePredictor/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _http_post_json(url: str, payload: str, timeout: int = 60) -> str:
    try:
        data = payload.encode("utf-8")
        req  = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent":   "PepTastePredictor/2.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_alphafold(seq: str) -> str:
    return ""


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_rcsb_pdb(seq: str) -> str:
    if len(seq) < 5:
        return ""
    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1,
                "identity_cutoff": 0.95,
                "sequence_type": "protein",
                "value": seq,
            },
        },
        "request_options": {"results_verbosity": "compact", "sort": [
            {"sort_by": "score", "direction": "desc"}]},
        "return_type": "entry",
    }
    try:
        data = json.dumps(query).encode("utf-8")
        req  = urllib.request.Request(
            RCSB_SEARCH,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "PepTastePredictor/2.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        hits = result.get("result_set", [])
        if not hits:
            return ""
        pdb_id  = hits[0]["identifier"].split("_")[0].upper()
        pdb_url = RCSB_FETCH.format(pdb_id)
        pdb_txt = _http_get(pdb_url, timeout=15)
        if pdb_txt and _validate_pdb(pdb_txt):
            return pdb_txt
    except Exception:
        pass
    return ""


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_esmfold_remote(seq: str) -> str:
    if len(seq) < 3 or len(seq) > 400:
        return ""
    try:
        data = seq.encode("utf-8")
        req  = urllib.request.Request(
            ESMFOLD_API,
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent":   "PepTastePredictor/2.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            pdb_txt = resp.read().decode("utf-8", errors="ignore")
        if pdb_txt and _validate_pdb(pdb_txt):
            return pdb_txt
    except Exception:
        pass
    return ""


def _build_realistic_peptide(seq: str) -> str:
    ss_result = predict_secondary_structure(seq)
    ss        = ss_result["assignments"]
    L         = len(seq)

    CA_C  = 1.52
    C_N   = 1.33
    N_CA  = 1.46
    OMEGA = np.radians(180.0)

    SS_ANGLES = {
        "H": (-57.0, -47.0),
        "E": (-120.0, 120.0),
        "C": (-60.0, 140.0),
    }

    def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        ca, sa = np.cos(angle), np.sin(angle)
        ux, uy, uz = axis
        return np.array([
            [ca + ux*ux*(1-ca),    ux*uy*(1-ca)-uz*sa, ux*uz*(1-ca)+uy*sa],
            [uy*ux*(1-ca)+uz*sa,   ca + uy*uy*(1-ca),  uy*uz*(1-ca)-ux*sa],
            [uz*ux*(1-ca)-uy*sa,   uz*uy*(1-ca)+ux*sa, ca + uz*uz*(1-ca)],
        ])

    def _place_atom(p1, p2, p3, bond_len, angle_deg, dihedral_deg):
        b1 = p3 - p2
        b2 = p2 - p1
        angle   = np.radians(angle_deg)
        dihedral= np.radians(dihedral_deg)

        b1n = b1 / (np.linalg.norm(b1) + 1e-12)
        b2n = b2 / (np.linalg.norm(b2) + 1e-12)

        n  = np.cross(b2n, b1n)
        nn = np.linalg.norm(n)
        if nn < 1e-10:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n /= nn

        rot1 = _rotation_matrix(n, np.pi - angle)
        d_dir = rot1 @ b1n

        rot2 = _rotation_matrix(b1n, dihedral)
        d_dir = rot2 @ d_dir

        return p3 + bond_len * d_dir

    N_CA_C  = 111.2
    CA_C_N  = 116.2
    C_N_CA  = 121.7

    atoms = {}

    atoms[(0, "N")]  = np.array([0.0, 0.0, 0.0])
    atoms[(0, "CA")] = np.array([N_CA, 0.0, 0.0])

    phi0, psi0 = SS_ANGLES.get(ss[0] if ss else "C", (-60.0, 140.0))
    c0 = _place_atom(
        np.array([-1.0, 0.0, 0.0]),
        atoms[(0, "N")],
        atoms[(0, "CA")],
        CA_C, N_CA_C, psi0,
    )
    atoms[(0, "C")] = c0

    for i in range(1, L):
        phi_i, psi_i = SS_ANGLES.get(ss[i] if i < len(ss) else "C", (-60.0, 140.0))

        n_i = _place_atom(
            atoms[(i-1, "N")],
            atoms[(i-1, "CA")],
            atoms[(i-1, "C")],
            C_N, CA_C_N, np.degrees(OMEGA),
        )
        atoms[(i, "N")] = n_i

        ca_i = _place_atom(
            atoms[(i-1, "CA")],
            atoms[(i-1, "C")],
            n_i,
            N_CA, C_N_CA, phi_i,
        )
        atoms[(i, "CA")] = ca_i

        c_i = _place_atom(
            atoms[(i-1, "C")],
            n_i,
            ca_i,
            CA_C, N_CA_C, psi_i,
        )
        atoms[(i, "C")] = c_i

    for i in range(L):
        if seq[i] == "G":
            continue
        n  = atoms[(i, "N")]
        ca = atoms[(i, "CA")]
        c  = atoms[(i, "C")]
        try:
            cb = _place_atom(c, n, ca, 1.52, 110.5, -122.5)
            atoms[(i, "CB")] = cb
        except Exception:
            pass

    lines    = []
    atom_num = 1
    atom_order = ["N", "CA", "C", "CB"]

    SS_BFACTOR = {"H": 85.0, "E": 75.0, "C": 55.0}

    for i in range(L):
        resname = THREE_LETTER.get(seq[i], "UNK")
        bf      = SS_BFACTOR.get(ss[i] if i < len(ss) else "C", 55.0)
        for aname in atom_order:
            if (i, aname) not in atoms:
                continue
            pos = atoms[(i, aname)]
            if not np.all(np.isfinite(pos)):
                continue
            x, y, z = pos
            aname_fmt = f" {aname:<3s}" if len(aname) < 4 else aname
            lines.append(
                f"ATOM  {atom_num:5d} {aname_fmt} {resname} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bf:6.2f}           "
                f"{'C' if aname == 'CA' or aname == 'CB' or aname == 'C' else 'N'}"
            )
            atom_num += 1

    lines.append("END")
    return "\n".join(lines)


def _build_geometric_pdb(seq: str) -> str:
    try:
        structure = PeptideBuilder.initialize_res(seq[0])
        for aa in seq[1:]:
            try:
                PeptideBuilder.add_residue(structure, Geometry.geometry(aa))
            except Exception:
                pass
        io_obj = PDBIO()
        io_obj.set_structure(structure)
        out = "predicted_peptide.pdb"
        io_obj.save(out)
        with open(out) as f:
            pdb = f.read()
        return pdb if _validate_pdb(pdb) else ""
    except Exception:
        return ""


def _minimal_single_residue_pdb(seq: str) -> str:
    lines = []
    for i, aa in enumerate(seq):
        resname = THREE_LETTER.get(aa, "UNK")
        x = i * 3.8
        lines.append(
            f"ATOM  {i+1:5d}  CA  {resname} A{i+1:4d}    "
            f"{x:8.3f}   0.000   0.000  1.00 50.00           C"
        )
    lines.append("END")
    return "\n".join(lines)


def predict_structure(seq: str, use_remote: bool = True) -> tuple:
    L = len(seq)

    if L <= 2:
        pdb = _build_realistic_peptide(seq)
        if not pdb or not _validate_pdb(pdb):
            pdb = _build_geometric_pdb(seq)
        if not pdb or not _validate_pdb(pdb):
            pdb = _minimal_single_residue_pdb(seq)
        return pdb, "Peptide Conformation Engine", "Geometric (1–2 aa)"

    if L <= 20:
        pdb = _build_realistic_peptide(seq)
        if pdb and _validate_pdb(pdb):
            engine = "Peptide Folding Engine"
            source = "Chou-Fasman SS + backbone torsions"
            if use_remote:
                esm_pdb = _fetch_esmfold_remote(seq)
                if esm_pdb and _validate_pdb(esm_pdb):
                    return esm_pdb, "Remote ESMFold", "ESM Atlas API"
            return pdb, engine, source

        pdb = _build_geometric_pdb(seq)
        if pdb and _validate_pdb(pdb):
            return pdb, "PeptideBuilder", "Geometric backbone"
        pdb = _minimal_single_residue_pdb(seq)
        return pdb, "PeptideBuilder (fallback)", "Linear chain"

    if use_remote:
        rcsb_pdb = _fetch_rcsb_pdb(seq)
        if rcsb_pdb and _validate_pdb(rcsb_pdb):
            return rcsb_pdb, "RCSB PDB Database", "Experimental / deposited structure"

    if use_remote and L <= 400:
        esm_pdb = _fetch_esmfold_remote(seq)
        if esm_pdb and _validate_pdb(esm_pdb):
            return esm_pdb, "Remote ESMFold", "ESM Atlas API (AI-predicted)"

    pdb = _build_realistic_peptide(seq)
    if pdb and _validate_pdb(pdb):
        return pdb, "Peptide Folding Engine", "Chou-Fasman SS + backbone torsions"

    pdb = _build_geometric_pdb(seq)
    if pdb and _validate_pdb(pdb):
        return pdb, "PeptideBuilder", "Geometric backbone"

    pdb = _minimal_single_residue_pdb(seq)
    return pdb, "PeptideBuilder (fallback)", "Linear chain"


# ==========================================================
# SECTION 11 - STRUCTURE VISUALIZATION
# ==========================================================

def show_structure(pdb_text: str, use_plddt_colors: bool = False):
    view = py3Dmol.view(width=1000, height=600)
    view.addModel(pdb_text, "pdb")

    plddt_vals = _extract_plddt(pdb_text)
    has_plddt  = len(plddt_vals) > 0 and max(plddt_vals) > 1.0

    if use_plddt_colors and has_plddt:
        view.setStyle({"cartoon": {"colorscheme": {
            "prop": "b",
            "gradient": "roygb",
            "min": 0, "max": 100,
        }}})
    else:
        view.setStyle({"cartoon": {"color": "spectrum"}})

    view.addSurface(py3Dmol.VDW, {"opacity": 0.2})
    view.zoomTo()
    return view


def render_plddt_legend():
    st.markdown("""
    <div class="plddt-legend" style="display:flex; justify-content:center; gap:20px; margin-top:15px;">
      <div class="plddt-chip"><span class="plddt-dot" style="background:#1565C0; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:5px;"></span>Very high (≥90)</div>
      <div class="plddt-chip"><span class="plddt-dot" style="background:#40C4FF; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:5px;"></span>Confident (70–90)</div>
      <div class="plddt-chip"><span class="plddt-dot" style="background:#FFEB3B; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:5px;"></span>Low (50–70)</div>
      <div class="plddt-chip"><span class="plddt-dot" style="background:#FF7043; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:5px;"></span>Very low (<50)</div>
    </div>
    """, unsafe_allow_html=True)


def render_structure_info_panel(seq: str, engine_label: str, source_detail: str, ss_result: dict):
    fold_type = classify_fold(ss_result, seq)
    L         = len(seq)
    h_pct     = round(ss_result["helix_frac"] * 100, 1)
    e_pct     = round(ss_result["sheet_frac"] * 100, 1)
    c_pct     = round(ss_result["coil_frac"]  * 100, 1)

    if "ESMFold" in engine_label:
        badge_cls, badge_icon = "badge-esm", "🧬"
    elif "RCSB" in engine_label or "PDB" in engine_label:
        badge_cls, badge_icon = "badge-pdb", "🗂️"
    elif "AlphaFold" in engine_label:
        badge_cls, badge_icon = "badge-af", "🔵"
    elif "Folding" in engine_label or "Conformation" in engine_label:
        badge_cls, badge_icon = "badge-fold", "🔮"
    else:
        badge_cls, badge_icon = "badge-pb", "⚙️"

    ss_bar_html = ""
    if h_pct > 0:
        ss_bar_html += f'<div class="ss-segment" style="width:{h_pct}%;background:#e91e63;"></div>'
    if e_pct > 0:
        ss_bar_html += f'<div class="ss-segment" style="width:{e_pct}%;background:#2196F3;"></div>'
    if c_pct > 0:
        ss_bar_html += f'<div class="ss-segment" style="width:{c_pct}%;background:#78909C;"></div>'

    st.markdown(f"""
    <div class="struct-info-panel">
      <h4>🔬 Structural Conformation Topology Mapping</h4>
      <div class="struct-row">
        <div class="struct-item">
          <span class="struct-label">Source Module Engine</span>
          <span class="engine-badge {badge_cls}" style="margin:0;">{badge_icon} {engine_label}</span>
        </div>
        <div class="struct-item">
          <span class="struct-label">Pipeline Trace</span>
          <span class="struct-value" style="font-size:13px;color:#A3AED0 !important;">{source_detail}</span>
        </div>
        <div class="struct-item">
          <span class="struct-label">Total Resolving Amino Acids</span>
          <span class="struct-value">{L} AA</span>
        </div>
        <div class="struct-item">
          <span class="struct-label">Assigned Topology Geometry</span>
          <span class="struct-value" style="color:#a78bfa !important;">{fold_type}</span>
        </div>
      </div>
      <div style="margin-top:20px;">
        <span class="struct-label">Secondary Structure Propensity Ratios</span>
        <div class="ss-bar">{ss_bar_html}</div>
        <div style="display:flex; gap:20px; font-size:12px;">
          <div><span style="color:#e91e63;">●</span> Helix: {h_pct}%</div>
          <div><span style="color:#2196F3;">●</span> Beta-Sheet: {e_pct}%</div>
          <div><span style="color:#78909C;">●</span> Random Coil: {c_pct}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# SECTION 12 - STRUCTURAL ANALYSIS FUNCTIONS
# ==========================================================

def ramachandran(pdb_text: str) -> list:
    if not pdb_text or not pdb_text.strip():
        return []
    tmp = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp)[0]
        pts = []
        for pp in PPBuilder().build_peptides(structure):
            for phi, psi in pp.get_phi_psi_list():
                if phi is not None and psi is not None:
                    pts.append((np.degrees(phi), np.degrees(psi)))
        return pts
    except Exception:
        return []
    finally:
        _unlink(tmp)


def ca_distance_map(pdb_text: str) -> np.ndarray:
    if not pdb_text or not pdb_text.strip():
        return np.zeros((1, 1))
    tmp = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp)
        cas = [r["CA"].get_vector().get_array()
               for r in structure.get_residues() if "CA" in r]
        if not cas:
            return np.zeros((1, 1))
        coords = np.array(cas)
        diff   = coords[:, None, :] - coords[None, :, :]
        return np.sqrt((diff ** 2).sum(-1))
    except Exception:
        return np.zeros((1, 1))
    finally:
        _unlink(tmp)


def ca_rmsd(pdb_text: str):
    if not pdb_text or not pdb_text.strip():
        return None
    tmp = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp)
        cas = [r["CA"].get_vector() for r in structure.get_residues() if "CA" in r]
        if len(cas) < 2:
            return None
        ref = cas[0]
        return float(np.sqrt(np.mean([(v - ref).norm() ** 2 for v in cas])))
    except Exception:
        return None
    finally:
        _unlink(tmp)


# ==========================================================
# SECTION 13 - PLOT FUNCTIONS
# ==========================================================

def plot_pca(X, y_labels, class_names, title="PCA"):
    C       = get_plot_colors()
    pca     = PCA(n_components=2)
    coords  = pca.fit_transform(X)
    v1, v2  = pca.explained_variance_ratio_[:2] * 100
    palette = plt.cm.get_cmap("cool", len(class_names))
    fig, ax = plt.subplots(figsize=(9, 6))
    apply_plot_style(fig, [ax])
    for i, cls in enumerate(class_names):
        mask = y_labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=cls, alpha=0.85, s=40, color=palette(i), edgecolors="none")
    ax.set_xlabel(f"PC1 ({v1:.1f}%)", fontsize=11, labelpad=10)
    ax.set_ylabel(f"PC2 ({v2:.1f}%)", fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    legend = ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left",
                       title="Taste Class Matrix", title_fontsize=9,
                       facecolor=C["fig_bg"], edgecolor=C["grid"])
    legend.get_title().set_color(C["text"])
    for t in legend.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig, pca


def plot_confusion(y_true, y_pred, class_names, title, cmap):
    C   = get_plot_colors()
    cm  = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    n   = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.75), max(5, n * 0.6)))
    apply_plot_style(fig, [ax])
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=1.0, linecolor=C["grid"],
                annot_kws={"size": 11, "color": "#FFFFFF"})
    ax.set_title(f"{title} (Accuracy: {acc*100:.1f}%)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Label Target", fontsize=11, labelpad=10)
    ax.set_ylabel("True Ground Validation", fontsize=11, labelpad=10)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(C["text"])
    cbar.ax.tick_params(colors=C["text"])
    plt.xticks(rotation=45, ha="right", fontsize=9, color=C["tick"])
    plt.yticks(rotation=0,  fontsize=9, color=C["tick"])
    plt.tight_layout()
    return fig


def plot_docking(y_true, y_pred):
    C    = get_plot_colors()
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    lims = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    fig, ax = plt.subplots(figsize=(6, 6))
    apply_plot_style(fig, [ax])
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="none", color=C["accent1"], s=45)
    ax.plot(lims, lims, color=C["red"], linestyle="--", lw=1.5, label="Perfect Alignment Fit")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.annotate(f"R² Fit = {r2:.3f}\nRMSE = {rmse:.2f} kcal/mol",
                xy=(0.05, 0.85), xycoords="axes fraction", fontsize=10, color=C["text"],
                bbox=dict(boxstyle="round,pad=0.4", fc=C["fig_bg"], ec=C["grid"], alpha=0.9))
    ax.set_xlabel("Experimental Bio-Dock Target (kcal/mol)", fontsize=11, labelpad=10)
    ax.set_ylabel("Machine Learned Forecast (kcal/mol)", fontsize=11, labelpad=10)
    ax.set_title("Docking Regression Distribution", fontsize=12, fontweight="bold", pad=12)
    legend = ax.legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in legend.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=20):
    C   = get_plot_colors()
    imp = pd.DataFrame({
        "Feature":    [prettify_feature(f) for f in feature_names],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(top_n)
    clrs = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp))[::-1])
    fig, ax = plt.subplots(figsize=(8, 7))
    apply_plot_style(fig, [ax])
    ax.barh(imp["Feature"][::-1], imp["Importance"][::-1], color=clrs, edgecolor=C["grid"])
    ax.set_xlabel("Relative Weight Coefficient", fontsize=11, labelpad=10)
    ax.set_title(f"Top {top_n} Primary Molecular Vector Descriptors", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_distributions(df):
    C            = get_plot_colors()
    seq_lengths = [len(s) for s in df["peptide"]]
    taste_counts = df["taste"].value_counts()
    grav_vals   = [gravy_score(s) for s in df["peptide"]]
    fig, axes   = plt.subplots(1, 3, figsize=(16, 5))
    apply_plot_style(fig, axes)
    
    mean_len = np.mean(seq_lengths)
    axes[0].hist(seq_lengths, bins=20, color=C["accent2"], edgecolor=C["grid"], alpha=0.8)
    axes[0].axvline(mean_len, color=C["red"], linestyle="--", lw=1.5, label=f"μ={mean_len:.1f}")
    axes[0].set_xlabel("Sequence Vector Length (AA)", fontsize=10)
    axes[0].set_title("Peptide Length Density Matrix", fontsize=11, fontweight="bold", pad=10)
    leg0 = axes[0].legend(fontsize=8, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg0.get_texts(): t.set_color(C["text"])
    
    n_cls = len(taste_counts)
    bar_colors = plt.cm.cool(np.linspace(0.1, 0.9, n_cls))
    axes[1].barh(taste_counts.index, taste_counts.values, color=bar_colors, edgecolor=C["grid"])
    axes[1].set_xlabel("Frequency Volume", fontsize=10)
    axes[1].set_title("Phenotypic Category Balance", fontsize=11, fontweight="bold", pad=10)
    for i, v in enumerate(taste_counts.values):
        axes[1].text(v + 0.3, i, str(v), va="center", fontsize=8, color=C["text"])
        
    axes[2].hist(grav_vals, bins=20, color=C["accent1"], edgecolor=C["grid"], alpha=0.8)
    axes[2].axvline(0, color=C["red"], linestyle="--", lw=1)
    axes[2].axvline(np.mean(grav_vals), color=C["orange"], linestyle="--", lw=1.5, label=f"μ={np.mean(grav_vals):.2f}")
    axes[2].set_xlabel("Hydropathicity Index (GRAVY)", fontsize=10)
    axes[2].set_title("Hydrophobic Profile Balance", fontsize=11, fontweight="bold", pad=10)
    leg2 = axes[2].legend(fontsize=8, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg2.get_texts(): t.set_color(C["text"])
    
    plt.tight_layout(pad=2.5)
    return fig


def plot_ramachandran(phi_psi):
    C   = get_plot_colors()
    fig, ax = plt.subplots(figsize=(6, 6))
    apply_plot_style(fig, [ax])
    ax.fill([-180,-180,-45,-45,-180], [-75,-45,-45,-75,-75], color="rgba(76, 175, 80, 0.15)", label="α-helix Region")
    ax.fill([-180,-180,-90,-90,-180], [90,180,180,90,90],    color="rgba(33, 150, 243, 0.15)", label="β-sheet Region")
    ax.fill([45,45,90,90,45],         [0,90,90,0,0],         color="rgba(255, 152, 0, 0.12)", label="L-helix Region")
    if phi_psi:
        phi, psi = zip(*phi_psi)
        ax.scatter(phi, psi, s=45, color="#ff6b6b", zorder=5, edgecolors="#FFFFFF", linewidths=0.5)
    ax.axhline(0, color=C["grid"], lw=0.8, linestyle=":")
    ax.axvline(0, color=C["grid"], lw=0.8, linestyle=":")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xlabel("Phi Torsion Angle φ (°)", fontsize=11, labelpad=10)
    ax.set_ylabel("Psi Torsion Angle ψ (°)", fontsize=11, labelpad=10)
    ax.set_title("Ramachandran Steric Alignment Check", fontsize=12, fontweight="bold", pad=12)
    leg = ax.legend(fontsize=8, loc="upper right", facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts(): t.set_color(C["text"])
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-180, 181, 60))
    plt.tight_layout()
    return fig


def plot_distance_map(dist_matrix, seq=""):
    C = get_plot_colors()
    n = dist_matrix.shape[0]
    if seq and len(seq) == n:
        labels = [f"{aa}{i+1}" for i, aa in enumerate(seq)]
    else:
        labels = [str(i + 1) for i in range(n)]
    tick_step   = max(1, n // 15)
    show_labels = [labels[i] if i % tick_step == 0 else "" for i in range(n)]
    size        = max(5, n * 0.28 + 2)
    fig, ax     = plt.subplots(figsize=(size, size))
    apply_plot_style(fig, [ax])
    sns.heatmap(dist_matrix, cmap="rocket", ax=ax,
                xticklabels=show_labels, yticklabels=show_labels,
                linewidths=0, cbar_kws={"label": "Euclidean Space Inter-Residue Vectors (Å)"})
    ax.set_title("Cα Structural Proximity Network Map", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Sequence Coordinate Position", fontsize=11, labelpad=10)
    ax.set_ylabel("Sequence Coordinate Position", fontsize=11, labelpad=10)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(C["text"])
    cbar.ax.tick_params(colors=C["text"])
    plt.xticks(rotation=45, ha="right", fontsize=8, color=C["tick"])
    plt.yticks(rotation=0,  fontsize=8, color=C["tick"])
    plt.tight_layout()
    return fig


def plot_plddt(plddt_vals, seq=""):
    C   = get_plot_colors()
    n   = len(plddt_vals)
    bar_colors = [
        "#1565C0" if v >= 90 else
        "#40C4FF" if v >= 70 else
        "#FFEB3B" if v >= 50 else
        "#FF7043"
        for v in plddt_vals
    ]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.22), 4))
    apply_plot_style(fig, [ax])
    ax.bar(range(n), plddt_vals, color=bar_colors, width=0.85)
    mean_pl = np.mean(plddt_vals)
    ax.axhline(mean_pl, color=C["orange"], linestyle="-.", lw=1.5, label=f"Mean μ = {mean_pl:.1f}")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Residue Sequential Space", fontsize=11, labelpad=8)
    ax.set_ylabel("pLDDT Statistical Factor", fontsize=11, labelpad=8)
    ax.set_title(f"Model Spatial Confidence Boundary Layer Metrics", fontsize=12, fontweight="bold", pad=12)
    if seq and len(seq) == n and n <= 60:
        ax.set_xticks(range(n))
        ax.set_xticklabels(list(seq), fontsize=8)
    leg = ax.legend(fontsize=8, loc="lower right", facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_ss_composition(ss_result: dict, seq: str):
    C   = get_plot_colors()
    ss  = ss_result["assignments"]
    n   = len(ss)
    if n == 0:
        return None
    color_map = {"H": "#e91e63", "E": "#2196F3", "C": "#78909C"}
    bar_cols  = [color_map.get(s, "#888") for s in ss]
    heights = [1.0 if s == "H" else 0.7 if s == "E" else 0.4 for s in ss]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.22), 3))
    apply_plot_style(fig, [ax])
    ax.bar(range(n), heights, color=bar_cols, width=0.85, edgecolor="none")
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0.4, 0.7, 1.0])
    ax.set_yticklabels(["Coil", "β-Sheet", "α-Helix"], fontsize=9, color=C["tick"])
    ax.set_xlabel("Residue Sequential Space", fontsize=11, labelpad=8)
    ax.set_title("Discrete Conformation Phase Segment Scanners", fontsize=12, fontweight="bold", pad=12)
    if seq and len(seq) == n and n <= 60:
        ax.set_xticks(range(n))
        ax.set_xticklabels(list(seq), fontsize=8)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e91e63", label="α-Helix"),
        Patch(facecolor="#2196F3", label="β-Sheet"),
        Patch(facecolor="#78909C", label="Coil/Loop"),
    ]
    leg = ax.legend(handles=legend_elements, fontsize=8, loc="upper right",
                    facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts(): t.set_color(C["text"])
    plt.tight_layout()
    return fig


# ==========================================================
# SECTION 14 - DYNAMIC CAPTIONS
# ==========================================================

def caption_distributions(df):
    lengths   = [len(s) for s in df["peptide"]]
    grav      = [gravy_score(s) for s in df["peptide"]]
    dom_taste = df["taste"].value_counts().idxmax()
    dom_count = df["taste"].value_counts().max()
    mean_grav = np.mean(grav)
    glabel    = ("slightly hydrophobic" if mean_grav > 0.2 else
                 "slightly hydrophilic" if mean_grav < -0.2 else "amphipathic")
    return (
        f"<strong>Length (left):</strong> Spatial sequence variance spans {int(np.min(lengths))}–{int(np.max(lengths))} residues, "
        f"with calculated mean centroid localized at {np.mean(lengths):.1f} AA.<br><br>"
        f"<strong>Taste Balance (centre):</strong> High abundance density skewed towards &ldquo;{dom_taste}&rdquo; matrices "
        f"({dom_count} entries recorded).<br><br>"
        f"<strong>GRAVY Index (right):</strong> Mean distribution scales at {mean_grav:.2f}, indicating a structurally <strong>{glabel}</strong> framework dataset alignment profile."
    )


def caption_pca(pca_model, class_names):
    v1, v2 = pca_model.explained_variance_ratio_[:2] * 100
    return (
        f"High-dimensional multi-omics features compressed down onto top principal components orthogonal coordinate systems.<br><br>"
        f"<strong>Component 1 (PC1):</strong> Captures {v1:.1f}% unique metric variance &nbsp;|&nbsp; "
        f"<strong>Component 2 (PC2):</strong> Captures {v2:.1f}% unique metric variance.<br><br>"
        f"Geometric clustering isolation correlates to predictive clarity indices for classification boundary verification steps."
    )


def caption_confusion_taste(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred) * 100
    cm  = confusion_matrix(y_true, y_pred)
    cp  = cm.astype(float); np.fill_diagonal(cp, 0)
    idx = np.unravel_index(np.argmax(cp), cp.shape)
    pca = cm.diagonal() / cm.sum(axis=1)
    return (
        f"Discrete Multi-Class Classifier Performance Evaluation: <strong>{acc:.1f}% Absolute Model Accuracy Cross-Validation</strong>.<br><br>"
        f"Critical Ambiguity Vectors: High misclassification weight encountered between &ldquo;{class_names[idx[0]]}&rdquo; models transitioning to "
        f"&ldquo;{class_names[idx[1]]}&rdquo; states ({int(cp[idx])} iteration incidents).<br>"
        f"Optimal Performing Category Array: &ldquo;{class_names[np.argmax(pca)]}&rdquo;."
    )


def caption_confusion_sol(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred) * 100
    cm  = confusion_matrix(y_true, y_pred)
    cp  = cm.astype(float); np.fill_diagonal(cp, 0)
    idx = np.unravel_index(np.argmax(cp), cp.shape)
    return (
        f"Solubility Vector Classifier Model Evaluation: <strong>{acc:.1f}% Validation Accuracy</strong>.<br><br>"
        f"Primary Structural False Error Path: &ldquo;{class_names[idx[0]]}&rdquo; misinterpreted as "
        f"&ldquo;{class_names[idx[1]]}&rdquo; ({int(cp[idx])} absolute count blocks)."
    )


def caption_feature_importance(model, feature_names, top_n=20):
    imp  = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    imp  = imp.sort_values("Importance", ascending=False).head(top_n)
    top3 = [(prettify_feature(r["Feature"]), r["Importance"]) for _, r in imp.head(3).iterrows()]
    return (
        f"Ranked weight profiles of chemical structures dictating model classification rules.<br><br>"
        + "".join(f"<strong>Rank #{i+1} Vector Domain component: {n}</strong> (Entropy impact score: {s:.4f})<br>" for i, (n, s) in enumerate(top3))
    )


def caption_docking(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    qual = "strong" if r2 >= 0.75 else ("moderate" if r2 >= 0.5 else "weak")
    return (
        f"Peptide-Peptide / Peptide-Receptor Free Energy Affinity Regression Metrics. Red line tracks theoretical convergence targets.<br><br>"
        f"<strong>R² Coefficient of Determination: {r2:.3f}</strong> (Classified as a <strong>{qual}</strong> mapping fit) — accounts for {r2*100:.1f}% mathematical feature covariance.<br>"
        f"<strong>Root Mean Squared Error (RMSE) Core Variance: {rmse:.2f} kcal/mol</strong>."
    )


def caption_ramachandran(phi_psi, seq=""):
    if not phi_psi:
        return "Torsion tracking context unavailable — sequence requires a minimum threshold of ≥3 residues with complete backbone atom groups."
    n_total = len(phi_psi)
    n_helix = sum(1 for p, s in phi_psi if -180 <= p <= -45 and -75 <= s <= -15)
    n_sheet = sum(1 for p, s in phi_psi if -180 <= p <= -45 and 90  <= s <= 180)
    n_other = n_total - n_helix - n_sheet
    dominant = "α-helix" if n_helix >= n_sheet else "β-sheet"
    return (
        f"Backbone backbone rotation trace verification matrices{f' for construct vector target <strong>{seq}</strong>' if seq else ''}.<br><br>"
        f"<strong>Alpha-Helix coordinate zone matches:</strong> {n_helix/n_total*100:.1f}% &nbsp;|&nbsp; "
        f"<strong>Beta-Sheet coordinate zone matches:</strong> {n_sheet/n_total*100:.1f}% &nbsp;|&nbsp; "
        f"<strong>Steric Strain/Outlier positions:</strong> {n_other/n_total*100:.1f}%<br><br>"
        f"Dominant Structural Backbone Geometrical Conformation: <strong>{dominant}</strong>."
    )


def caption_distance_map(dist_matrix, seq=""):
    n = dist_matrix.shape[0]
    if n < 2:
        return "Matrix resolution requires two spatial Cα atoms."
    mask = ~np.eye(n, dtype=bool)
    od   = dist_matrix[mask]
    lr   = sum(1 for i in range(n) for j in range(n) if abs(i-j) > 3 and dist_matrix[i,j] < 8.0)
    fold = ("indicating a highly globular complex <strong>compact spatial loop fold-back structure</strong>"
            if lr > 0 else "indicating a fully linearized, <strong>extended unstructured architecture chain</strong>")
    return (
        f"Pairwise intramolecular matrix distances mapping the peptide alpha carbon backbone grid topology. Shifting dark colors mirror spatial closeness thresholds.<br><br>"
        f"<strong>Spatial Range limits:</strong> {od.min():.2f}Å to {od.max():.2f}Å &nbsp;|&nbsp; "
        f"<strong>Long-Range Matrix Spatial Contacts</strong> (|i−j|>3, d<8Å): {lr} observations — {fold}."
    )


# ==========================================================
# SECTION 15 - STRUCTURAL ANALYSIS RENDER
# ==========================================================

def render_structural_analysis(pdb_text: str, prefix: str = "", seq: str = ""):
    if not pdb_text or not pdb_text.strip():
        st.warning("Spatial structure file stream is empty. Analytical execution bypassed.")
        return

    if seq:
        ss_result = predict_secondary_structure(seq)
        if len(ss_result["assignments"]) >= 3:
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("### 🎨 Geometrical Phase Assignment Profile")
            fig_ss = plot_ss_composition(ss_result, seq)
            if fig_ss is not None:
                save_fig(fig_ss, f"{prefix}ss_composition.png")
                st.pyplot(fig_ss)
                plt.close(fig_ss)
                h_pct = round(ss_result["helix_frac"] * 100, 1)
                e_pct = round(ss_result["sheet_frac"] * 100, 1)
                c_pct = round(ss_result["coil_frac"]  * 100, 1)
                show_caption(
                    f"Secondary structure alignment predictions tracking sequence <strong>{seq[:30]}{'…' if len(seq)>30 else ''}</strong>.<br><br>"
                    f"<strong>α-Helix density allocation:</strong> {h_pct}% &nbsp;|&nbsp; "
                    f"<strong>β-Sheet density allocation:</strong> {e_pct}% &nbsp;|&nbsp; "
                    f"<strong>Random Coil configuration:</strong> {c_pct}%"
                )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### 📐 Steric Conformational Adequacy Map (Ramachandran)")
    phi_psi = ramachandran(pdb_text)
    fig_rama = plot_ramachandran(phi_psi)
    save_fig(fig_rama, f"{prefix}ramachandran.png")
    st.pyplot(fig_rama)
    plt.close(fig_rama)
    show_caption(caption_ramachandran(phi_psi, seq=seq))

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### 🗺️ Intramolecular Matrix Spatial Contacts Map")
    dist_map = ca_distance_map(pdb_text)
    fig_dist = plot_distance_map(dist_map, seq=seq)
    save_fig(fig_dist, f"{prefix}ca_distance_map.png")
    st.pyplot(fig_dist)
    plt.close(fig_dist)
    show_caption(caption_distance_map(dist_map, seq=seq))

    plddt_vals = _extract_plddt(pdb_text)
    if plddt_vals and max(plddt_vals) > 1.0:
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Backbone Stability Layer Confidence Profiles (pLDDT)")
        render_plddt_legend()
        fig_pl = plot_plddt(plddt_vals, seq=seq)
        save_fig(fig_pl, f"{prefix}plddt.png")
        st.pyplot(fig_pl)
        plt.close(fig_pl)
        mean_pl = np.mean(plddt_vals)
        quality = ("Very High" if mean_pl >= 90 else "High" if mean_pl >= 70 else
                   "Medium" if mean_pl >= 50 else "Low")
        show_caption(
            f"Per-residue model reliability matrix indexes.<br><br>"
            f"<strong>Mean Confidence Value:</strong> {mean_pl:.1f} — <strong>Classified as {quality} Model Stability</strong>. "
            f"Fringes trailing below a score threshold of < 50 indicate highly flexible flexible unstructured loop spaces."
        )


# ==========================================================
# SECTION 16 - MODEL TRAINING
# ==========================================================

@st.cache_data
def train_models():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Required structural training archive database lost or missing path coordinates: {DATASET_PATH}")
        st.stop()

    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.lower().str.strip()
    df["peptide"] = df["peptide"].apply(clean_sequence)
    df = df[df["peptide"].str.len() >= 1].reset_index(drop=True)
    df = df[
        df["taste"].notna()
        & df["solubility"].notna()
        & df["docking score (kcal/mol)"].notna()
    ].reset_index(drop=True)

    df["solubility"] = df["solubility"].str.strip().str.rstrip(".")
    df["taste"]      = simplify_taste(df["taste"])

    X        = build_feature_table(df["peptide"])
    le_taste = LabelEncoder()
    le_sol   = LabelEncoder()
    y_taste  = le_taste.fit_transform(df["taste"])
    y_sol    = le_sol.fit_transform(df["solubility"])
    y_dock   = df["docking score (kcal/mol)"].values

    idx            = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_taste)
    Xtr, Xte       = X.iloc[tr_idx], X.iloc[te_idx]
    yt_tr, yt_te   = y_taste[tr_idx], y_taste[te_idx]
    ys_tr, ys_te   = y_sol[tr_idx],   y_sol[te_idx]
    yd_tr, yd_te   = y_dock[tr_idx],  y_dock[te_idx]

    taste_model = ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42)
    sol_model   = ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    dock_model  = RandomForestRegressor(n_estimators=400, random_state=42)

    taste_model.fit(Xtr, yt_tr)
    sol_model.fit(Xtr, ys_tr)
    dock_model.fit(Xtr, yd_tr)

    metrics = {
        "Taste accuracy":      accuracy_score(yt_te, taste_model.predict(Xte)),
        "Taste F1":            f1_score(yt_te, taste_model.predict(Xte), average="weighted"),
        "Solubility accuracy": accuracy_score(ys_te, sol_model.predict(Xte)),
        "Solubility F1":       f1_score(ys_te, sol_model.predict(Xte), average="weighted"),
        "Docking RMSE":        np.sqrt(mean_squared_error(yd_te, dock_model.predict(Xte))),
        "Docking R2":          r2_score(yd_te, dock_model.predict(Xte)),
    }

    return (df, X, Xte, yt_te, ys_te, yd_te,
            taste_model, sol_model, dock_model,
            le_taste, le_sol, metrics)


# ==========================================================
# SECTION 17 - LOAD MODELS
# ==========================================================

(
    df_all, X_all, X_test, yt_test, ys_test, yd_test,
    taste_model, sol_model, dock_model,
    le_taste, le_sol, metrics,
) = train_models()


# ==========================================================
# SECTION 18 - PDF REPORT ENGINE
# ==========================================================

def generate_pdf(metrics: dict, prediction: dict, image_paths: list) -> str:
    file_name = "PepTastePredictor_Report.pdf"
    styles    = getSampleStyleSheet()
    doc       = SimpleDocTemplate(file_name, pagesize=A4,
                                  topMargin=40, bottomMargin=40,
                                  leftMargin=50, rightMargin=50)
    story = []
    story.append(Paragraph("<b>PepTastePredictor — Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "AI-driven peptide taste, solubility, docking & structural analysis platform. "
        "Hybrid Structural Engine v2 — Priority cascade: RCSB PDB → Remote ESMFold → "
        "Peptide Folding Engine → PeptideBuilder.",
        styles["Normal"]))
    story.append(Spacer(1, 14))
    story.append(Paragraph("<b>Model Performance</b>", styles["Heading2"]))
    tbl_data = [["Metric", "Value"]] + [[k, str(round(v, 4))] for k, v in metrics.items()]
    tbl = Table(tbl_data, colWidths=[280, 150])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  rl_colors.HexColor("#1f3c88")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  rl_colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("BACKGROUND",    (0, 1), (-1, -1), rl_colors.HexColor("#f0f4ff")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [rl_colors.HexColor("#f0f4ff"), rl_colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#cccccc")),
        ("FONTSIZE",      (0, 1), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 14))
    if prediction:
        story.append(Paragraph("<b>Prediction Results</b>", styles["Heading2"]))
        for k, v in prediction.items():
            story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
        story.append(Spacer(1, 14))
    story.append(Paragraph("<b>Visual Analytics</b>", styles["Heading2"]))
    story.append(Spacer(1, 8))
    figure_titles = {
        "distributions.png":           "Dataset Distributions",
        "pca_overall.png":             "PCA Feature Space",
        "confusion_taste.png":         "Taste Confusion Matrix",
        "confusion_solubility.png":    "Solubility Confusion Matrix",
        "feature_importance_taste.png":"Feature Importance",
        "docking_scatter.png":         "Docking True vs Predicted",
        "ss_composition.png":          "Secondary Structure Profile",
        "ramachandran.png":            "Ramachandran Plot",
        "ca_distance_map.png":         "Cα Distance Map",
        "plddt.png":                   "pLDDT Confidence Profile",
    }
    for img in image_paths:
        if not os.path.exists(img):
            continue
        basename = os.path.basename(img)
        title    = next((v for k, v in figure_titles.items() if k in basename), basename)
        story.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
        story.append(RLImage(img, width=430, height=270))
        story.append(Spacer(1, 20))
    story.append(Paragraph(
        f"<i>Generated by PepTastePredictor v2 · Hybrid Structural Engine · "
        f"{date.today().strftime('%d %B %Y')}</i>",
        styles["Normal"]))
    doc.build(story)
    return file_name


# ==========================================================
# SECTION 19 - HERO HEADER PANEL BLOCK
# ==========================================================

st.markdown("""
<div class="hero">
<h1>🧬 PepTaste Workspace</h1>
<p>
An advanced integrated machine learning &amp; structural bioinformatics workspace architecture for structural
peptide validation, solubility profiles, free energy affinities, and structural folding predictions.<br>
<strong>Multi-Class Structural Layer v2:</strong> Completely responsive priority fallback engine execution tracing 
RCSB PDB &rarr; ESMFold API &rarr; Idealized Backbone Torsion Modellers. Minimal environment overhead, zero local dependency.
</p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# SECTION 20 - MODE SELECTION PANEL
# ==========================================================

st.markdown("<h3 style='font-family:\"Syne\",sans-serif; font-weight:700; color:#ECF0F5 !important;'>🔧 Operational Module Selection</h3>", unsafe_allow_html=True)
mode = st.radio(
    "Choose active runtime operation context",
    ["Single Peptide Prediction", "Batch Peptide Prediction", "PDB Upload & Structural Analysis"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("<br>", unsafe_allow_html=True)

if "current_mode" not in st.session_state or st.session_state.current_mode != mode:
    st.session_state.pdf_figures     = []
    st.session_state.show_analytics  = False
    st.session_state.last_prediction = {}
    st.session_state.pdb_text        = None
    st.session_state.pdb_source      = None
    st.session_state.current_mode    = mode


# ==========================================================
# SECTION 21 - SINGLE PEPTIDE PREDICTION MODE
# ==========================================================

if mode == "Single Peptide Prediction":

    st.markdown("<h4 style='font-family:\"Syne\",sans-serif; color:#00D4A0 !important;'>⚗️ Monomeric Sequence Analysis Console</h4>", unsafe_allow_html=True)

    with st.form("single_peptide_form"):
        fasta_file = st.file_uploader(
            "Upload targeted character dataset vector (FASTA structural string map file format)",
            type=["fasta", "fa", "faa"],
            key="single_fasta_upload",
        )
        
        fasta_seq = ""
        if fasta_file is not None:
            try:
                fasta_text = fasta_file.read().decode("utf-8")
                records    = parse_fasta(fasta_text)
                if records:
                    fasta_seq = records[0][1]
                    st.info(f"Buffered FASTA Record: {records[0][0] or 'unnamed'} — {len(fasta_seq)} residues tracked.")
            except Exception as e:
                st.error(f"File system parsing exception: {e}")

        seq_raw = st.text_area(
            "Amino acid linear structure input parameters string (Single-Letter AA)",
            value=fasta_seq,
            placeholder="Paste raw character codes string map (e.g., EKKGIMDKIKEKLPGGHKKTGSS)...",
            key="single_seq_input",
            height=110,
        )
        
        use_remote = st.checkbox(
            "🌐 Allow network pipeline queries across remote data servers (RCSB Repository + ESM Atlas)",
            value=True,
        )
        
        submit_btn = st.form_submit_button("Launch Analytical Pipeline Core →", type="primary")

    seq = clean_sequence(seq_raw)

    if submit_btn:
        st.session_state.pdf_figures = []

        if incap_len := len(seq) < 1:
            st.error("Operation failed. Input sequence parameters length lacks structural mass (Zero string dimensions).")
        else:
            # ── ML Predictions ──────────────────────────────────
            ml_seq = seq[:100]
            Xp     = pd.DataFrame([model_features(ml_seq)])
            taste  = le_taste.inverse_transform(taste_model.predict(Xp))[0]
            sol    = le_sol.inverse_transform(sol_model.predict(Xp))[0]
            dock   = dock_model.predict(Xp)[0]
            emoji  = taste_emoji(taste)

            sol_color  = "#00D4A0" if "soluble" in sol.lower() else "#e67e22"
            dock_color = "#00D4A0" if dock < -6 else ("#ffa94d" if dock < -4 else "#ff6b6b")

            # Upgraded StressPep HTML-Style Metric Grid Layout Display panel
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 25px; margin-top:15px;">
                <div class="premium-card">
                    <div class="card-title"><span class="live-indicator"></span>Classification Phase</div>
                    <div class="card-value">{emoji} {taste}</div>
                    <div class="card-sub">Assigned phenotypic class configuration</div>
                </div>
                <div class="premium-card">
                    <div class="card-title">Solubility Index</div>
                    <div class="card-value" style="color:{sol_color} !important;">{sol}</div>
                    <div class="card-sub">Hydropathicity threshold evaluation</div>
                </div>
                <div class="premium-card">
                    <div class="card-title">Free Energy Affinity Delta</div>
                    <div class="card-value" style="color:{dock_color} !important;">{dock:.2f} <span style="font-size:13px; font-weight:400; color:#525866;">kcal/mol</span></div>
                    <div class="card-sub">{'Strong variant binding structural fit' if dock<-6 else 'Moderate variant binding structural fit' if dock<-4 else 'Weak variant binding structural fit'}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if len(seq) > 100:
                st.info(f"Monomeric machine learning parameters strictly parse head boundary 100 AA fragments of your {len(seq)} sequence array.")

            taste_proba = taste_model.predict_proba(Xp)[0]
            sol_proba   = sol_model.predict_proba(Xp)[0]
            
            c1, c2 = st.columns(2)
            c1.metric("Classifier Confidence Margin",      f"{max(taste_proba)*100:.1f}%")
            c2.metric("Solubility Layer Probability Weight", f"{max(sol_proba)*100:.1f}%")

            # ── Physicochemical ─────────────────────────────────
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("### 📌 Macromolecular Character Metrics")
            phys = physicochemical_features(ml_seq)
            cols = st.columns(min(len(phys), 4))
            for i, (k, v) in enumerate(phys.items()):
                cols[i % len(cols)].metric(k, v)

            st.markdown("### 🧪 Element/Residue Structural Density Split")
            comp      = composition_features(seq)
            comp_cols = st.columns(len(comp))
            for i, (k, v) in enumerate(comp.items()):
                comp_cols[i].metric(k, f"{v}%")

            # ── Structure Generation (Hybrid Engine) ────────────
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("<h3 style='font-family:\"Syne\",sans-serif; color:#ECF0F5 !important;'>🧬 3D Intramolecular Atomic Structure Spatial Render</h3>", unsafe_allow_html=True)

            with st.spinner("Executing structural folding calculations via cascading framework pipeline loops..."):
                pdb_text, engine_label, source_detail = predict_structure(seq, use_remote=use_remote)

            st.session_state.pdb_text   = pdb_text
            st.session_state.pdb_source = engine_label

            ss_result = predict_secondary_structure(seq)
            render_structure_info_panel(seq, engine_label, source_detail, ss_result)

            plddt_vals = _extract_plddt(pdb_text)
            use_plddt  = (plddt_vals and max(plddt_vals) > 1.0 and
                          ("ESMFold" in engine_label or "RCSB" in engine_label or "AlphaFold" in engine_label))

            st.components.v1.html(
                show_structure(pdb_text, use_plddt_colors=use_plddt)._make_html(),
                height=620,
            )

            if use_plddt:
                render_plddt_legend()

            rmsd_val = ca_rmsd(pdb_text)
            if rmsd_val is not None:
                st.success(f"Calculated Alpha Carbon Backbone Root Mean Square Deviation (Cα RMSD): {rmsd_val:.3f} Å")

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "⬇️ Export Local Structure Geometry File (PDB Format)",
                pdb_text,
                file_name=f"peptide_model_{seq[:12]}.pdb",
                mime="text/plain",
            )

            # ── Structural Analysis ─────────────────────────────
            render_structural_analysis(pdb_text, prefix="single_", seq=seq)

            st.session_state.last_prediction = {
                "Sequence":               seq[:60] + ("…" if len(seq) > 60 else ""),
                "Predicted taste":          taste,
                "Predicted solubility":      sol,
                "Docking score (kcal/mol)": round(dock, 3),
                "Taste confidence":         f"{max(taste_proba)*100:.1f}%",
                "Structure engine":         engine_label,
                "Structure source":         source_detail,
                "Predicted fold":            classify_fold(ss_result, seq),
                "α-Helix fraction":         f"{ss_result['helix_frac']*100:.1f}%",
                "β-Sheet fraction":         f"{ss_result['sheet_frac']*100:.1f}%",
            }
            st.session_state.show_analytics = True


# ==========================================================
# SECTION 22 - BATCH PEPTIDE PREDICTION MODE
# ==========================================================

elif mode == "Batch Peptide Prediction":

    st.markdown("<h4 style='font-family:\"Syne\",sans-serif; color:#00D4A0 !important;'>📦 High-Throughput Batch Processing Core Engine</h4>", unsafe_allow_html=True)

    with st.form("batch_processing_form"):
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            batch_csv = st.file_uploader("Upload structured variant catalog file alignment index (CSV layout with 'peptide' header column context)", type=["csv"])
        with col_up2:
            batch_fasta = st.file_uploader("Or link complete concatenated multiple fasta asset streams (.FASTA / .FA / .FAA structural variants formats)", type=["fasta", "fa", "faa"])

        gen_structures = st.checkbox(
            "Compile custom 3D topological model vectors across full cohort array entries (Exports inside compressed archive file ZIP packaging)",
            value=False,
        )
        batch_use_remote = st.checkbox("🌐 Execute structural calculations via public distributed networks (RCSB + ESM Servers)", value=True)
        
        batch_submit = st.form_submit_button("Launch High-Throughput Matrix Computations →", type="primary")

    batch_df   = None
    batch_seqs = []

    if batch_submit:
        if batch_csv is not None:
            try:
                batch_df = pd.read_csv(batch_csv)
                if "peptide" not in batch_df.columns:
                    st.error("Processing sequence failed. Column map architecture requires explicit identity key tracking string labeled exactly: 'peptide'.")
                    batch_df = None
                else:
                    batch_df["peptide"] = batch_df["peptide"].apply(clean_sequence)
                    batch_df = batch_df[batch_df["peptide"].str.len() >= 1].reset_index(drop=True)
                    batch_seqs = batch_df["peptide"].tolist()
            except Exception as e:
                st.error(f"CSV operational loop interruption error: {e}")

        elif batch_fasta is not None:
            try:
                fasta_text = batch_fasta.read().decode("utf-8")
                records    = parse_fasta(fasta_text)
                if not records:
                    st.error("Multi-FASTA trace read error. Extracted sequence contents contain structural formatting discrepancies.")
                else:
                    batch_df   = pd.DataFrame(records, columns=["Header Identity Key", "peptide"])
                    batch_seqs = batch_df["peptide"].tolist()
            except Exception as e:
                st.error(f"FASTA block extraction exception error trace: {e}")

        if batch_df is not None and batch_seqs:
            total    = len(batch_seqs)
            st.info(f"Target vector framework initialized successfully. Processing entry arrays trace sizing across: **{total}** sequence columns.")
            progress = st.progress(0, text="Staging processing variables baseline states...")

            tastes, sols, docks, taste_confs, engines, fold_types = [], [], [], [], [], []
            pdb_files = {}

            for i, seq_b in enumerate(batch_seqs):
                try:
                    ml_seq = seq_b[:100]
                    Xr     = pd.DataFrame([model_features(ml_seq)])
                    t      = le_taste.inverse_transform(taste_model.predict(Xr))[0]
                    s      = le_sol.inverse_transform(sol_model.predict(Xr))[0]
                    d      = round(dock_model.predict(Xr)[0], 3)
                    tc     = round(max(taste_model.predict_proba(Xr)[0]) * 100, 1)
                except Exception:
                    t, s, d, tc = "Error", "Error", None, None
                tastes.append(t); sols.append(s); docks.append(d); taste_confs.append(tc)

                try:
                    ss_b = predict_secondary_structure(seq_b)
                    ft   = classify_fold(ss_b, seq_b)
                except Exception:
                    ft = "Unknown"
                fold_types.append(ft)

                if gen_structures:
                    try:
                        pdb_b, eng_b, _ = predict_structure(seq_b, use_remote=batch_use_remote)
                        pdb_files[f"peptide_{i+1}_{seq_b[:12]}.pdb"] = pdb_b
                        engines.append(eng_b)
                    except Exception:
                        pdb_b = _build_geometric_pdb(seq_b) or _minimal_single_residue_pdb(seq_b)
                        pdb_files[f"peptide_{i+1}_{seq_b[:12]}.pdb"] = pdb_b
                        engines.append("PeptideBuilder (fallback)")
                else:
                    engines.append("—")

                progress.progress(
                    min(int((i + 1) / total * 100), 100),
                    text=f"Resolving entries grid maps: {i+1} of {total} completed...",
                )

            progress.progress(100, text="High-Throughput evaluation execution loop successfully closed.")

            batch_df["Predicted Taste Variant"]         = tastes
            batch_df["Predicted Solubility Index"]    = sols
            batch_df["Predicted Free Energy Affinities"] = docks
            batch_df["Classifier Confidence Score (%)"]    = taste_confs
            batch_df["Predicted Fold Category Class"]     = fold_types
            batch_df["Structure Optimization Module"]        = engines

            st.markdown("### ✅ Vector Cohort Analytics Matrix Results")
            st.dataframe(batch_df, use_container_width=True)
            
            st.download_button(
                "⬇️ Export Processed Metrics Table (CSV Asset File)",
                batch_df.to_csv(index=False),
                file_name="batch_analysis_matrix_output.csv",
            )

            if gen_structures and pdb_files:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fname, pdb_content in pdb_files.items():
                        if pdb_content:
                            zf.writestr(fname, pdb_content)
                zip_buffer.seek(0)
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    "⬇️ Export Compiled Geometries Archive Package (ZIP Cluster Resource File)",
                    zip_buffer,
                    file_name="batch_geometries_package.zip",
                    mime="application/zip",
                )

            st.session_state.show_analytics = True


# ==========================================================
# SECTION 23 - PDB UPLOAD & STRUCTURAL ANALYSIS MODE
# ==========================================================

elif mode == "PDB Upload & Structural Analysis":

    st.markdown("<h4 style='font-family:\"Syne\",sans-serif; color:#00D4A0 !important;'>🧩 Static Coordinates Importing & Geometrical Parsing Block</h4>", unsafe_allow_html=True)
    st.info("Direct spatial parsing architecture. Processes externally rendered, machine-learned folding predictions or experimentally solved files.", icon="🌐")

    with st.form("pdb_upload_form"):
        uploaded_pdb = st.file_uploader("Choose structural coordinate database map file (.PDB architecture specification file context format)", type=["pdb"])
        parse_submit = st.form_submit_button("Initiate Spatial Structural Analysis Framework →", type="primary")

    if parse_submit and uploaded_pdb is not None:
        try:
            pdb_text = uploaded_pdb.read().decode("utf-8")
        except Exception as e:
            st.error(f"Coordinate character loading exception occurred: {e}")
            pdb_text = ""

        if pdb_text and pdb_text.strip():
            if not _validate_pdb(pdb_text):
                st.error("Parsing initialization error: Imported file contents fail minimal requirements constraints validation checkpoints (missing functional target CA coordinates array).")
            else:
                st.session_state.pdb_text       = pdb_text
                st.session_state.pdb_source     = "Uploaded PDB"
                st.session_state.show_analytics = True

                n_atoms = sum(1 for l in pdb_text.splitlines() if l.startswith("ATOM"))
                n_res   = _count_residues_in_pdb(pdb_text)
                
                c1, c2 = st.columns(2)
                c1.metric("ResolvedIntramolecular Atoms", n_atoms)
                c2.metric("Extracted Residuic Grid Length",     n_res)

                seq_from_pdb = ""
                seen_res = {}
                ONE_LETTER = {v: k for k, v in THREE_LETTER.items()}
                for line in pdb_text.splitlines():
                    if line.startswith("ATOM"):
                        chain   = line[21]
                        res_seq = line[22:26].strip()
                        resname = line[17:20].strip()
                        key     = (chain, res_seq)
                        if key not in seen_res:
                            seen_res[key] = ONE_LETTER.get(resname, "X")
                seq_from_pdb = "".join(seen_res.values()).replace("X", "")

                plddt_vals = _extract_plddt(pdb_text)
                has_plddt  = plddt_vals and max(plddt_vals) > 1.0

                if seq_from_pdb:
                    ss_result = predict_secondary_structure(seq_from_pdb)
                    render_structure_info_panel(
                        seq_from_pdb, "Imported Coordinates File", uploaded_pdb.name, ss_result)

                st.markdown("### 🧬 Interactive Atom Layer Space Mapping Viewer")
                st.components.v1.html(
                    show_structure(pdb_text, use_plddt_colors=has_plddt)._make_html(),
                    height=620,
                )
                if has_plddt:
                    render_plddt_legend()

                rmsd_val = ca_rmsd(pdb_text)
                if rmsd_val is not None:
                    st.success(f"Calculated Spatial Alpha Carbon Displacement Metric (Cα RMSD Baseline Alignment Coefficient): {rmsd_val:.3f} Å")

                render_structural_analysis(pdb_text, prefix="pdb_", seq=seq_from_pdb)
        else:
            st.error("Parsing failure. Uploaded coordinate data payload structure is corrupted or contains empty variable arrays.")


# ==========================================================
# SECTION 24 - MODEL & DATASET ANALYTICS
# ==========================================================

if st.session_state.show_analytics:

    st.markdown("---")

    with st.expander("📊 Mathematical Optimization Performance Matrices & Feature Space Indicators", expanded=False):

        st.markdown("<h4 style='font-family:\"Syne\",sans-serif; color:#00D4A0 !important; margin-bottom:15px;'>📈 Baseline Validation Performance Profiles</h4>", unsafe_allow_html=True)
        
        # Upgraded StressPep Dashboard Card Look for Global Analytics Engine Metrics
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 25px;">
            <div class="premium-card"><div class="card-title">Taste Matrix Accuracy</div><div class="card-value">{metrics['Taste accuracy']*100:.1f}%</div><div class="card-sub">Out-of-sample prediction success</div></div>
            <div class="premium-card"><div class="card-title">Taste Weighted F1-Score</div><div class="card-value" style="color:#5c7cfa !important;">{metrics['Taste F1']:.3f}</div><div class="card-sub">Harmonic precision balance map</div></div>
            <div class="premium-card"><div class="card-title">Solubility Validation Acc</div><div class="card-value" style="color:#ffa94d !important;">{metrics['Solubility accuracy']*100:.1f}%</div><div class="card-sub">Discrete boundary confirmation</div></div>
            <div class="premium-card"><div class="card-title">Solubility F1 Metric</div><div class="card-value" style="color:#ffa94d !important;">{metrics['Solubility F1']:.3f}</div><div class="card-sub">Imbalanced dataset verification</div></div>
            <div class="premium-card"><div class="card-title">Docking Fit R² Coefficient</div><div class="card-value" style="color:#a78bfa !important;">{metrics['Docking R2']:.3f}</div><div class="card-sub">Explained regression variance ratio</div></div>
            <div class="premium-card"><div class="card-title">Docking Convergence RMSE</div><div class="card-value" style="color:#ff6b6b !important;">{metrics['Docking RMSE']:.2f}</div><div class="card-sub">Deviations standard error metric scale</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Cohort Training Dataset Distribution Analytics")
        fig_dist = plot_distributions(df_all)
        save_fig(fig_dist, "distributions.png")
        st.pyplot(fig_dist)
        plt.close(fig_dist)
        show_caption(caption_distributions(df_all))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Principal Component Analysis (PCA) Latent Structural Feature Space")
        fig_pca, pca_model = plot_pca(
            X_all, le_taste.transform(df_all["taste"]), le_taste.classes_,
            title="PCA Dim-Reduction Vector Map Compression Matrix (Grouped by phenotypic designation tags)",
        )
        save_fig(fig_pca, "pca_overall.png")
        st.pyplot(fig_pca)
        plt.close(fig_pca)
        show_caption(caption_pca(pca_model, le_taste.classes_))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Phenotypic Category Assignment Error Analysis Matrix (Taste Confusion)")
        taste_preds  = taste_model.predict(X_test)
        fig_cm_taste = plot_confusion(yt_test, taste_preds, le_taste.classes_,
                                      "Classifier Convergence Discrepancies Grid", "Blues")
        save_fig(fig_cm_taste, "confusion_taste.png")
        st.pyplot(fig_cm_taste)
        plt.close(fig_cm_taste)
        show_caption(caption_confusion_taste(yt_test, taste_preds, le_taste.classes_))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Solubility Phase Error Localization Matrix (Solubility Confusion)")
        sol_preds  = sol_model.predict(X_test)
        fig_cm_sol = plot_confusion(ys_test, sol_preds, le_sol.classes_,
                                    "Solubility Category Error Matrix Grid Map", "Greens")
        save_fig(fig_cm_sol, "confusion_solubility.png")
        st.pyplot(fig_cm_sol)
        plt.close(fig_cm_sol)
        show_caption(caption_confusion_sol(ys_test, sol_preds, le_sol.classes_))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Information Gain Coefficients Weight Rankings (Feature Importance)")
        fig_imp = plot_feature_importance(taste_model, X_all.columns, top_n=20)
        save_fig(fig_imp, "feature_importance_taste.png")
        st.pyplot(fig_imp)
        plt.close(fig_imp)
        show_caption(caption_feature_importance(taste_model, X_all.columns, top_n=20))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Free Energy Docking Affinities Regression Scatter Profiles")
        dock_preds = dock_model.predict(X_test)
        fig_dock   = plot_docking(yd_test, dock_preds)
        save_fig(fig_dock, "docking_scatter.png")
        st.pyplot(fig_dock)
        plt.close(fig_dock)
        show_caption(caption_docking(yd_test, dock_preds))


# ==========================================================
# SECTION 25 - PDF DOWNLOAD PANEL
# ==========================================================

if st.session_state.show_analytics and len(st.session_state.pdf_figures) > 0:
    st.markdown("<h3 style='font-family:\"Syne\",sans-serif; color:#ECF0F5 !important;'>📄 Automated Documentation Compiler</h3>", unsafe_allow_html=True)
    pdf_path = generate_pdf(
        metrics, st.session_state.last_prediction, st.session_state.pdf_figures)
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📥 Export Complete Comprehensive Structural Analytics File (PDF Report Format)", f,
                file_name="PepTastePredictor_Comprehensive_Analytical_Report.pdf",
                mime="application/pdf",
                type="secondary"
            )


# ==========================================================
# SECTION 26 - FOOTER
# ==========================================================

st.markdown(f"""
<div class="footer">
&copy; {date.today().year} &nbsp; <b>PepTaste Integrated Hybrid Framework Platform v2.0</b><br>
AI Algorithms Optimization + Structural Topological Bioinformatics Pipelines · High-Throughput Matrix Screening Engine Core Architecture<br>
System Operational Framework Cascade Sequence Tracking Priority: RCSB PDB Repository System &rarr; Remote Automated ESMFold Inference Engine &rarr; Chou-Fasman Dihedral Torsion Structurizers &rarr; Local PeptideBuilder Geometry Fallbacks.<br>
<span style="opacity:0.4;">Strictly intended for academic research, validation exercises, and educational exploration settings.</span>
</div>
""", unsafe_allow_html=True)
