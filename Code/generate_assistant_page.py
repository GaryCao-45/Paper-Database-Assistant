"""
Paper Database Assistant -- HTML Tool Generator
================================================
Generates a local HTML page with two tabs:
  1. PDF batch download (via browser tabs)
  2. LLM-based feature extraction (Google Gemini / OpenRouter)

Usage:
    python generate_assistant_page.py

Then open the generated paper_assistant.html in Chrome.
"""

import re
import os
import sys
import json

DOI_LIST_FILE = "non_first_tier_dois_2020_2026.txt"
OUTPUT_DIR    = "./downloaded_pdfs"


def parse_doi_list(filepath):
    entries = []
    year = "unknown"
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"[\u3010](\d{4})\u5e74[\u3011]", line)
            if m:
                year = m.group(1)
                continue
            m = re.match(r"(\d+)\.\s*(https://doi\.org/.+)", line)
            if m:
                url = m.group(2).strip()
                doi = url.replace("https://doi.org/", "")
                pub = detect_publisher(doi)
                pdf_url = construct_pdf_url(doi, pub)
                entries.append({
                    "year": year,
                    "index": int(m.group(1)),
                    "doi": doi,
                    "doi_url": url,
                    "pdf_url": pdf_url if pdf_url else url,
                    "has_direct_pdf": pdf_url is not None,
                    "publisher": pub,
                })
    return entries


def detect_publisher(doi):
    if doi.startswith("10.1002/"): return "wiley"
    if doi.startswith("10.1021/"): return "acs"
    if doi.startswith("10.1016/"): return "elsevier"
    if doi.startswith("10.1038/"): return "nature"
    if doi.startswith("10.1126/"): return "science"
    return "unknown"


def construct_pdf_url(doi, publisher):
    if publisher == "wiley":
        return f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}"
    elif publisher == "acs":
        return f"https://pubs.acs.org/doi/pdf/{doi}"
    elif publisher == "science":
        return f"https://www.science.org/doi/pdf/{doi}"
    return None


# ======================== EXTRACTION CONFIG ========================

_SYSTEM_PROMPT = """You are an expert data extractor for perovskite solar cell research. Your task is to extract precise fabrication and performance parameters from research papers provided as page images.

You can see the FULL paper: text, tables, figures, cross-section SEM, device schematics -- use ALL visual information.

=== EXTRACTION RULES ===

1. EVERY distinct device CONFIGURATION = one entry in "devices" array. Extract ALL configurations: control, target, different additives/concentrations, different HTLs, etc.
2. For each configuration, extract the CHAMPION (best) device performance, NOT average+/-std.
3. Use null ONLY when a parameter is truly not reported anywhere in the paper.
4. Extract BOTH reverse and forward scan J-V data when available.
5. Shared parameters (same substrate, ETL, electrode across all devices) MUST be repeated in every device entry.
6. ONLY extract standard small-area devices (typically 0.04-0.2 cm2). If the paper ALSO reports large-area devices (>1 cm2) or modules, note them in paper_notes but do NOT create separate device entries for them.

=== DEVICE ARCHITECTURE ===

This database targets p-i-n (inverted) perovskite solar cells.
- p-i-n stack (bottom to top): Substrate -> HTL -> Perovskite -> ETL -> Electrode
  - Common HTL (bottom): NiOx, PTAA, SAMs (Me-4PACz, 2PACz, MeO-2PACz), PEDOT:PSS, P3CT-N
  - Common ETL (top): C60, PCBM, BCP, LiF, SnO2 (by ALD)
- n-i-p stack (bottom to top): Substrate -> ETL -> Perovskite -> HTL -> Electrode
  - Common bottom ETL: TiO2, SnO2, ZnO
  - Common top HTL: Spiro-OMeTAD, PTAA, P3HT

CRITICAL: Identify the architecture FIRST by looking at the device schematic or layer description. Then assign HTL_material and ETL_material according to their FUNCTION, not position.
- If the paper is n-i-p: still fill HTL_material with the hole-transport material and ETL_material with the electron-transport material, but write "n-i-p structure" in extraction_notes.

=== UNITS (MANDATORY) ===

All numerical values MUST be converted to these standard units before reporting:
- Thickness (HTL, ETL, BCP, Electrode): nm
- Annealing temperature: C
- Annealing time, stirring time: minutes (if paper says "30 s", convert to 0.5; if "2 h", convert to 120)
- Spin coating speed: rpm
- Spin coating time: seconds
- Antisolvent volume: uL (if paper says "1 mL", convert to 1000)
- Antisolvent drip delay: seconds from start of spin step
- Cell active area: cm2
- Jsc: mA/cm2
- Voc: V
- FF: percentage (82.5, NOT 0.825)
- PCE: percentage
- Bandgap: eV

=== PRECISION REQUIREMENTS ===

NUMERICAL VALUES:
- Copy numbers EXACTLY as printed in tables/text. Do NOT round. If the table says 18.06, write 18.06 -- not 18.1.
- Read performance data from summary tables first (most accurate), then cross-check with text.
- After copying, convert to the standard units listed above if needed.

CATEGORICAL VALUES -- BE MAXIMALLY DESCRIPTIVE:
- HTL_material: if the HTL includes additives, dopants, or mixed materials, include them ALL.
  Examples: "PTAA+PBDB-T-SF", "Eu-doped NiOx", "NiOx/Me-4PACz bilayer", "Li-doped Spiro-OMeTAD"
- HTL_subtype: for SAMs -> exact molecule name; for NiOx -> preparation method (nanoparticle/sputtered/sol-gel/combustion); for doped materials -> dopant identity
- Perovskite_additive: list ALL additives with their roles if multiple are used, separated by commas.
  Example: "MACl, PEAI" not just "MACl"
- Antisolvent_type: if gas quenching (N2 knife, air knife) is used instead of traditional antisolvent, write the technique name (e.g., "N2 gas quenching")
- Deposition_procedure: "one-step", "two-step (sequential)", "blade coating", "slot-die coating", "vapor deposition", etc. Always specify.
- Fab_atmosphere: be specific -- "glovebox N2", "ambient air (RH 30-40%)", "dry air glovebox", "N2-filled glovebox"

=== MANDATORY PERFORMANCE DATA ===

- Every device MUST have Jsc, Voc, FF, and PCE as explicit numbers from a TABLE or TEXT.
- Do NOT estimate or read approximate values from figures, scatter plots, bar charts, or J-V curves.
- If a device's Jsc/Voc/FF/PCE values exist ONLY in a figure (no numerical table or text), SKIP that device entirely.
- If ALL devices in a paper only have graphical performance data, return an empty "devices" array and explain in paper_notes.

=== TABLE READING ===

- Performance summary tables are the PRIMARY source for Jsc, Voc, FF, PCE values.
- If a table shows "champion" or "best" device data, extract those CHAMPION values.
- If a table shows both average+/-std and champion values, use the CHAMPION values only.
- If both reverse and forward scan columns exist in the table, extract BOTH.
- If the table has a "stabilized PCE" or "SPO" column, note it in paper_notes.

=== FIGURE READING ===

- Cross-section SEM images: extract layer thicknesses if labeled.
- Device architecture schematics: use to CONFIRM the layer stack and architecture type (p-i-n vs n-i-p).
- Do NOT extract Jsc/Voc/FF/PCE from figures -- only use figures for layer thicknesses and architecture confirmation.

=== OTHER NOTES ===

- Precursor_filtration: 1 if filtered before use, 0 if explicitly not filtered, null if not mentioned.
- BCP_present: 1 if BCP is used, 0 if explicitly not used, null if not mentioned.
- If only one scan direction is reported without specifying which, assume reverse scan.
- If a "stabilized" or "steady-state" PCE is reported, note it in paper_notes but put J-V scan values in the main fields.
- Perovskite_formula: write the FULL stoichiometric formula exactly as in the paper, e.g., "Cs0.05FA0.79MA0.16Pb(I0.83Br0.17)3"."""

_USER_PROMPT = """Extract ALL device configuration data from this perovskite solar cell paper.

STEP-BY-STEP:
1. Identify the device architecture (p-i-n or n-i-p) from schematics or layer descriptions.
2. Read the Experimental/Methods section for fabrication parameters.
3. Read ALL performance summary tables -- for each configuration, extract CHAMPION values (not average+/-std).
4. Check cross-section SEM and schematics for layer thicknesses.
5. Convert all values to standard units (nm, minutes, rpm, uL, cm2, etc. -- see system prompt).
6. For each distinct device configuration, fill in all fields below.

Return a JSON object with this EXACT structure (showing 2 devices as example -- extract as many as the paper reports):
{
  "paper_notes": "e.g. n-i-p structure; large-area (1 cm2) device also reported with 22.1% PCE",
  "devices": [
    {
      "device_label": "control",
      "DOI": "10.1016/j.cej.2024.xxxxx",
      "Year": 2024,
      "Substrate_type": "ITO glass",
      "HTL_material": "NiOx",
      "HTL_subtype": "sol-gel",
      "HTL_deposition_method": "spin coating",
      "HTL_concentration": "20 mg/mL",
      "HTL_solvent": "2-methoxyethanol",
      "HTL_annealing_temp": 300,
      "HTL_annealing_time": 60,
      "HTL_additive": null,
      "HTL_thickness": 30,
      "Buried_interface_treatment": null,
      "Buried_interface_concentration": null,
      "Perovskite_formula": "Cs0.05FA0.79MA0.16Pb(I0.83Br0.17)3",
      "Bandgap": 1.62,
      "Precursor_solvents": "DMF+DMSO",
      "Precursor_solvent_ratio": "4:1",
      "Precursor_concentration": "1.2 M",
      "Perovskite_additive": "MACl",
      "Additive_concentration": "30 mol%",
      "Precursor_stirring_time": 120,
      "Precursor_filtration": 1,
      "Deposition_procedure": "one-step spin coating",
      "Spin_step1_speed": 1000,
      "Spin_step1_time": 10,
      "Spin_step2_speed": 5000,
      "Spin_step2_time": 30,
      "Spin_acceleration": 2000,
      "Antisolvent_type": "chlorobenzene",
      "Antisolvent_volume": 200,
      "Antisolvent_drip_delay": 25,
      "Anneal_1_temp": 100,
      "Anneal_1_time": 10,
      "Anneal_2_temp": null,
      "Anneal_2_time": null,
      "Annealing_atmosphere": "N2",
      "Fab_atmosphere": "N2-filled glovebox",
      "Fab_humidity": null,
      "Top_passivation_agent": null,
      "Top_passivation_concentration": null,
      "Top_passivation_method": null,
      "ETL_material": "C60",
      "ETL_deposition_method": "thermal evaporation",
      "ETL_thickness": 20,
      "ETL_additive": null,
      "BCP_present": 1,
      "BCP_thickness": 8,
      "Alt_interlayer": null,
      "Electrode_material": "Ag",
      "Electrode_thickness": 100,
      "Cell_active_area": 0.08,
      "JV_scan_direction": "reverse",
      "Jsc_reverse": 22.85,
      "Voc_reverse": 1.142,
      "FF_reverse": 79.33,
      "PCE_reverse": 20.71,
      "Jsc_forward": null,
      "Voc_forward": null,
      "FF_forward": null,
      "PCE_forward": null
    },
    {
      "device_label": "target (with PEAI passivation)",
      "DOI": "10.1016/j.cej.2024.xxxxx",
      "Year": 2024,
      "Substrate_type": "ITO glass",
      "HTL_material": "NiOx",
      "HTL_subtype": "sol-gel",
      "HTL_deposition_method": "spin coating",
      "HTL_concentration": "20 mg/mL",
      "HTL_solvent": "2-methoxyethanol",
      "HTL_annealing_temp": 300,
      "HTL_annealing_time": 60,
      "HTL_additive": null,
      "HTL_thickness": 30,
      "Buried_interface_treatment": "PEAI",
      "Buried_interface_concentration": "2 mg/mL in IPA",
      "Perovskite_formula": "Cs0.05FA0.79MA0.16Pb(I0.83Br0.17)3",
      "Bandgap": 1.62,
      "Precursor_solvents": "DMF+DMSO",
      "Precursor_solvent_ratio": "4:1",
      "Precursor_concentration": "1.2 M",
      "Perovskite_additive": "MACl",
      "Additive_concentration": "30 mol%",
      "Precursor_stirring_time": 120,
      "Precursor_filtration": 1,
      "Deposition_procedure": "one-step spin coating",
      "Spin_step1_speed": 1000,
      "Spin_step1_time": 10,
      "Spin_step2_speed": 5000,
      "Spin_step2_time": 30,
      "Spin_acceleration": 2000,
      "Antisolvent_type": "chlorobenzene",
      "Antisolvent_volume": 200,
      "Antisolvent_drip_delay": 25,
      "Anneal_1_temp": 100,
      "Anneal_1_time": 10,
      "Anneal_2_temp": null,
      "Anneal_2_time": null,
      "Annealing_atmosphere": "N2",
      "Fab_atmosphere": "N2-filled glovebox",
      "Fab_humidity": null,
      "Top_passivation_agent": "PEAI",
      "Top_passivation_concentration": "2 mg/mL in IPA",
      "Top_passivation_method": "spin coating",
      "ETL_material": "C60",
      "ETL_deposition_method": "thermal evaporation",
      "ETL_thickness": 20,
      "ETL_additive": null,
      "BCP_present": 1,
      "BCP_thickness": 8,
      "Alt_interlayer": null,
      "Electrode_material": "Ag",
      "Electrode_thickness": 100,
      "Cell_active_area": 0.08,
      "JV_scan_direction": "both",
      "Jsc_reverse": 25.13,
      "Voc_reverse": 1.182,
      "FF_reverse": 82.47,
      "PCE_reverse": 24.51,
      "Jsc_forward": 24.92,
      "Voc_forward": 1.175,
      "FF_forward": 81.23,
      "PCE_forward": 23.82
    }
  ]
}

CRITICAL REMINDERS:
- Copy ALL numbers with FULL decimal precision from tables. Never round.
- Convert to standard units: thickness->nm, annealing time->min, spin time->s, volume->uL, area->cm2.
- For each configuration, use CHAMPION values only, not average+/-std.
- For categorical fields, be MAXIMALLY descriptive (include dopants, additives, bilayer info).
- Deposition_procedure MUST always be filled if the paper describes how the film was made.
- If gas quenching is used instead of antisolvent, write it in Antisolvent_type (e.g., "N2 gas quenching").
- Skip large-area (>1 cm2) or module devices -- just note them in paper_notes.
- Return ONLY valid JSON. No markdown code blocks, no explanation text."""

_COLUMNS = [
    "filename", "device_label", "device_index", "needs_SI", "extraction_notes",
    "DOI", "Year",
    "Substrate_type",
    "HTL_material", "HTL_subtype", "HTL_deposition_method", "HTL_concentration",
    "HTL_solvent", "HTL_annealing_temp", "HTL_annealing_time", "HTL_additive",
    "HTL_thickness",
    "Buried_interface_treatment", "Buried_interface_concentration",
    "Perovskite_formula", "Bandgap",
    "Precursor_solvents", "Precursor_solvent_ratio", "Precursor_concentration",
    "Perovskite_additive", "Additive_concentration", "Precursor_stirring_time",
    "Precursor_filtration",
    "Deposition_procedure", "Spin_step1_speed", "Spin_step1_time",
    "Spin_step2_speed", "Spin_step2_time", "Spin_acceleration",
    "Antisolvent_type", "Antisolvent_volume", "Antisolvent_drip_delay",
    "Anneal_1_temp", "Anneal_1_time", "Anneal_2_temp", "Anneal_2_time",
    "Annealing_atmosphere",
    "Fab_atmosphere", "Fab_humidity",
    "Top_passivation_agent", "Top_passivation_concentration", "Top_passivation_method",
    "ETL_material", "ETL_deposition_method", "ETL_thickness", "ETL_additive",
    "BCP_present", "BCP_thickness", "Alt_interlayer",
    "Electrode_material", "Electrode_thickness",
    "Cell_active_area", "JV_scan_direction",
    "Jsc_reverse", "Voc_reverse", "FF_reverse", "PCE_reverse",
    "Jsc_forward", "Voc_forward", "FF_forward", "PCE_forward",
]


def generate_html(entries):
    papers_json = json.dumps(entries, ensure_ascii=False, indent=2)
    sys_prompt_js = json.dumps(_SYSTEM_PROMPT)
    usr_prompt_js = json.dumps(_USER_PROMPT)
    columns_js = json.dumps(_COLUMNS)

    html = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Paper Database Assistant</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif;
    background: #0f0f1a; color: #e0e0e0; padding: 20px;
}
.container { max-width: 1200px; margin: 0 auto; }
h1 { color: #60a5fa; margin-bottom: 8px; font-size: 24px; }
.subtitle { color: #888; margin-bottom: 20px; }

/* Tabs */
.tabs { display: flex; gap: 0; margin-bottom: 20px; border-bottom: 2px solid #333; }
.tab-btn {
    padding: 10px 24px; background: transparent; color: #888; border: none;
    border-bottom: 2px solid transparent; cursor: pointer; font-size: 15px;
    margin-bottom: -2px; transition: all 0.2s;
}
.tab-btn:hover { color: #ccc; transform: none; }
.tab-btn.active { color: #60a5fa; border-bottom-color: #60a5fa; }

/* Stat cards */
.stats { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.stat-card {
    background: #1a1a2e; border-radius: 8px; padding: 12px 20px;
    border-left: 3px solid #60a5fa; min-width: 120px;
}
.stat-card .num { font-size: 28px; font-weight: bold; color: #60a5fa; }
.stat-card .label { font-size: 12px; color: #888; }
.stat-card.success { border-color: #4ade80; }
.stat-card.success .num { color: #4ade80; }
.stat-card.fail { border-color: #f87171; }
.stat-card.fail .num { color: #f87171; }
.stat-card.pending { border-color: #fbbf24; }
.stat-card.pending .num { color: #fbbf24; }

/* Panels & controls */
.panel, .controls {
    background: #1a1a2e; border-radius: 8px; padding: 16px; margin-bottom: 16px;
}
.panel-header {
    font-size: 15px; font-weight: 600; color: #60a5fa;
    margin-bottom: 12px; display: flex; align-items: center; gap: 12px;
}
.controls p { margin-bottom: 8px; line-height: 1.6; }
.controls .hint { color: #888; font-size: 13px; }
.btn-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }

/* Buttons */
button {
    padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer;
    font-size: 14px; font-weight: 500; transition: all 0.2s;
}
button:hover { transform: translateY(-1px); }
button:active { transform: translateY(0); }
button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.btn-primary { background: #3b82f6; color: white; }
.btn-primary:hover:not(:disabled) { background: #2563eb; }
.btn-success { background: #22c55e; color: white; }
.btn-success:hover:not(:disabled) { background: #16a34a; }
.btn-danger { background: #ef4444; color: white; }
.btn-danger:hover:not(:disabled) { background: #dc2626; }
.btn-secondary { background: #374151; color: #d1d5db; }
.btn-secondary:hover:not(:disabled) { background: #4b5563; }
.btn-sm { padding: 4px 10px; font-size: 12px; }

/* Inputs */
select, input[type="number"], input[type="text"], input[type="password"] {
    background: #374151; color: white; border: 1px solid #555;
    border-radius: 4px; padding: 6px 10px; font-size: 14px;
}
input:focus, select:focus { border-color: #60a5fa; outline: none; }
.form-row {
    display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 10px;
}
.form-group { display: flex; flex-direction: column; gap: 4px; min-width: 100px; }
.form-group.wide { flex: 2; min-width: 200px; }
.form-group label { font-size: 12px; color: #888; }
.form-group input, .form-group select { width: 100%; }

/* Prompts */
.prompt-area {
    width: 100%; min-height: 120px;
    background: #0d1117; color: #6e7681; border: 1px solid #30363d; border-radius: 4px;
    font-family: Consolas, Monaco, 'Courier New', monospace; font-size: 12px;
    padding: 12px; resize: vertical; line-height: 1.5;
    transition: color 0.3s, border-color 0.3s;
}
.prompt-area:focus, .prompt-area.edited { color: #c9d1d9; border-color: #58a6ff; outline: none; }
.prompt-hint { font-size: 11px; color: #555; margin-bottom: 6px; font-style: italic; }

/* File list */
.file-list { max-height: 200px; overflow-y: auto; margin-top: 8px; font-size: 13px; }
.file-item {
    display: flex; justify-content: space-between; padding: 4px 8px;
    border-bottom: 1px solid #1e1e30;
}
.file-item.processing { background: #1e293b; }
.file-item .fname { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.file-item .fstatus { min-width: 90px; text-align: right; }

/* Progress bar */
.progress-bar {
    width: 100%; height: 6px; background: #374151; border-radius: 3px;
    margin-top: 8px; overflow: hidden;
}
.progress-fill {
    height: 100%; background: linear-gradient(90deg, #3b82f6, #22c55e);
    border-radius: 3px; transition: width 0.3s;
}
#statusBar, #extStatusBar {
    margin-top: 8px; padding: 8px 12px; border-radius: 4px;
    background: #1e293b; font-size: 13px; min-height: 32px;
}

/* Log */
.log-area {
    background: #0d1117; color: #8b949e; border-radius: 4px; padding: 12px;
    font-family: Consolas, Monaco, 'Courier New', monospace; font-size: 12px;
    max-height: 300px; overflow-y: auto; line-height: 1.5;
    white-space: pre-wrap; word-wrap: break-word;
}

/* Tables */
table { width: 100%; border-collapse: collapse; margin-top: 8px; }
th {
    background: #1a1a2e; padding: 10px 12px; text-align: left; font-size: 13px;
    color: #888; border-bottom: 2px solid #333; position: sticky; top: 0; z-index: 10;
}
td { padding: 8px 12px; border-bottom: 1px solid #1e1e30; font-size: 13px; }
tr:hover { background: #1a1a2e; }
a { color: #60a5fa; text-decoration: none; }
a:hover { text-decoration: underline; }

/* Tags */
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
.tag-wiley { background: #1e3a5f; color: #60a5fa; }
.tag-acs { background: #1e3a2f; color: #4ade80; }
.tag-elsevier { background: #3a2f1e; color: #fbbf24; }
.tag-nature { background: #3a1e2f; color: #f472b6; }
.tag-science { background: #2f1e3a; color: #a78bfa; }
.tag-unknown { background: #333; color: #999; }
.status-icon { font-size: 16px; }

/* Download filters */
.filters { display: flex; gap: 8px; align-items: center; margin-top: 12px; flex-wrap: wrap; }
.filter-btn {
    padding: 4px 12px; border-radius: 12px; font-size: 12px;
    background: #374151; color: #d1d5db; border: none; cursor: pointer;
}
.filter-btn.active { background: #3b82f6; color: white; }

/* Results overflow */
.results-scroll { overflow-x: auto; }
.results-scroll td, .results-scroll th { white-space: nowrap; }
.results-scroll td { max-width: 220px; overflow: hidden; text-overflow: ellipsis; }
</style>
</head>
<body>
<div class="container">
    <h1>Paper Database Assistant</h1>
    <p class="subtitle">PDF batch download + LLM feature extraction | All API calls run directly from your browser</p>

    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('download',this)">PDF Download</button>
        <button class="tab-btn" onclick="switchTab('extract',this)">Feature Extraction</button>
    </div>

    <!-- ==================== DOWNLOAD TAB ==================== -->
    <div id="tab-download" class="tab-content">
        <div class="stats">
            <div class="stat-card"><div class="num" id="totalCount">0</div><div class="label">Total</div></div>
            <div class="stat-card success"><div class="num" id="openedCount">0</div><div class="label">Opened</div></div>
            <div class="stat-card pending"><div class="num" id="pendingCount">0</div><div class="label">Pending</div></div>
            <div class="stat-card fail"><div class="num" id="skippedCount">0</div><div class="label">Skipped</div></div>
        </div>

        <div class="controls">
            <p><strong>Usage:</strong></p>
            <p>1. Make sure you are on <strong>campus network</strong> or connected to <strong>institutional VPN</strong></p>
            <p>2. Select batch size, click "Start Batch"</p>
            <p>3. PDFs open in new tabs automatically every few seconds</p>
            <p class="hint">If a CAPTCHA appears, complete it manually then click "Resume". Recommend 5-10 per batch.</p>

            <div class="btn-row">
                <span style="line-height:32px;">Batch:</span>
                <select id="batchSize">
                    <option value="3">3</option>
                    <option value="5" selected>5</option>
                    <option value="10">10</option>
                    <option value="20">20</option>
                </select>
                <span style="line-height:32px;">Interval:</span>
                <select id="interval">
                    <option value="1500">1.5s</option>
                    <option value="2000">2s</option>
                    <option value="3000" selected>3s</option>
                    <option value="5000">5s</option>
                </select>
                <button class="btn-primary" onclick="startBatch()">Start Batch</button>
                <button class="btn-success" onclick="resumeBatch()">Resume</button>
                <button class="btn-danger" onclick="stopBatch()">Pause</button>
                <button class="btn-secondary" onclick="resetAll()">Reset</button>
            </div>

            <div class="filters">
                <span style="font-size:12px;color:#888;">Filter:</span>
                <button class="filter-btn active" onclick="setFilter('all')">All</button>
                <button class="filter-btn" onclick="setFilter('wiley')">Wiley</button>
                <button class="filter-btn" onclick="setFilter('acs')">ACS</button>
                <button class="filter-btn" onclick="setFilter('elsevier')">Elsevier</button>
                <button class="filter-btn" onclick="setFilter('nature')">Nature</button>
                <button class="filter-btn" onclick="setFilter('science')">Science</button>
                <button class="filter-btn" onclick="setFilter('pending')">Pending</button>
            </div>

            <div id="statusBar">Ready...</div>
            <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
        </div>

        <table>
            <thead><tr>
                <th style="width:50px">#</th><th style="width:50px">Year</th>
                <th>DOI</th><th style="width:80px">Publisher</th>
                <th style="width:100px">Action</th><th style="width:60px">Status</th>
            </tr></thead>
            <tbody id="tbody"></tbody>
        </table>
    </div>

    <!-- ==================== EXTRACTION TAB ==================== -->
    <div id="tab-extract" class="tab-content" style="display:none">

        <!-- API Settings -->
        <div class="panel">
            <div class="panel-header">API Settings</div>
            <div class="form-row">
                <div class="form-group">
                    <label>Provider</label>
                    <select id="extProvider" onchange="onProviderChange()">
                        <option value="google">Google Gemini</option>
                        <option value="openrouter">OpenRouter</option>
                    </select>
                </div>
                <div class="form-group wide">
                    <label>API Key</label>
                    <input type="password" id="extApiKey" placeholder="Enter your API key here">
                </div>
                <div class="form-group">
                    <label>Model</label>
                    <input type="text" id="extModel" value="gemini-2.5-flash">
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>Image DPI</label>
                    <input type="number" id="extDpi" value="200" min="72" max="400" step="50">
                </div>
                <div class="form-group">
                    <label>Max Pages</label>
                    <input type="number" id="extMaxPages" value="30" min="1" max="100">
                </div>
                <div class="form-group">
                    <label>Delay (sec)</label>
                    <input type="number" id="extDelay" value="5" min="0" max="120">
                </div>
                <div class="form-group">
                    <label>Max Retries</label>
                    <input type="number" id="extRetries" value="3" min="1" max="10">
                </div>
            </div>
        </div>

        <!-- Prompts -->
        <div class="panel">
            <div class="panel-header">
                Prompts
                <button class="btn-secondary btn-sm" onclick="togglePrompts()">Show / Hide</button>
                <button class="btn-secondary btn-sm" onclick="resetPrompts()">Reset Defaults</button>
            </div>
            <div id="promptSection" style="display:none">
                <p class="prompt-hint">Default prompts are pre-filled below (dimmed). Click to edit. These define how the LLM extracts data from papers.</p>
                <label style="font-size:13px;color:#aaa;margin-top:8px;display:block">System Prompt</label>
                <textarea id="extSysPrompt" class="prompt-area" rows="12"></textarea>
                <label style="font-size:13px;color:#aaa;margin-top:12px;display:block">User Prompt</label>
                <textarea id="extUsrPrompt" class="prompt-area" rows="12"></textarea>
            </div>
        </div>

        <!-- File Upload -->
        <div class="panel">
            <div class="panel-header">PDF Files</div>
            <div class="form-row" style="align-items:center">
                <button class="btn-secondary" onclick="document.getElementById('extFileInput').click()">Select Files</button>
                <button class="btn-secondary" onclick="document.getElementById('extFolderInput').click()">Select Folder</button>
                <input type="file" id="extFileInput" multiple accept=".pdf" style="display:none" onchange="onFilesSelected(this)">
                <input type="file" id="extFolderInput" webkitdirectory multiple style="display:none" onchange="onFilesSelected(this)">
                <span id="extFileCount" style="color:#888;font-size:13px">No files selected</span>
                <button class="btn-secondary btn-sm" onclick="clearExtFiles()" style="margin-left:auto">Clear Files</button>
            </div>
            <div id="extFileList" class="file-list"></div>
        </div>

        <!-- Controls -->
        <div class="panel">
            <div class="btn-row" style="margin-top:0">
                <button class="btn-primary" id="extStartBtn" onclick="startExtraction()">Start Extraction</button>
                <button class="btn-danger" id="extStopBtn" onclick="stopExtraction()" disabled>Pause</button>
                <button class="btn-success" onclick="exportCSV()">Export CSV</button>
                <button class="btn-secondary" onclick="clearExtResults()">Clear Results</button>
            </div>
            <div class="stats" style="margin-top:12px">
                <div class="stat-card"><div class="num" id="extTotal">0</div><div class="label">Files</div></div>
                <div class="stat-card success"><div class="num" id="extDone">0</div><div class="label">Done</div></div>
                <div class="stat-card fail"><div class="num" id="extErrors">0</div><div class="label">Errors</div></div>
                <div class="stat-card pending"><div class="num" id="extEntries">0</div><div class="label">Entries</div></div>
            </div>
            <div class="progress-bar"><div class="progress-fill" id="extProgress" style="width:0%"></div></div>
            <div id="extStatusBar">Ready. Select PDF files and configure API to begin.</div>
        </div>

        <!-- Results -->
        <div class="panel">
            <div class="panel-header">Results (<span id="extResultCount">0</span> entries)</div>
            <div class="results-scroll">
                <table id="extResultsTable">
                    <thead id="extResultsHead"><tr><th style="color:#555">Extracted data will appear here automatically</th></tr></thead>
                    <tbody id="extResultsBody"></tbody>
                </table>
            </div>
        </div>

        <!-- Log -->
        <div class="panel">
            <div class="panel-header">
                Processing Log
                <button class="btn-secondary btn-sm" onclick="document.getElementById('extLog').textContent=''">Clear Log</button>
            </div>
            <pre id="extLog" class="log-area"></pre>
        </div>
    </div>
</div>

<script>
/* ==================== TAB SWITCHING ==================== */
function switchTab(name, btn) {
    document.querySelectorAll('.tab-content').forEach(function(el) { el.style.display = 'none'; });
    document.getElementById('tab-' + name).style.display = 'block';
    document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
}

/* ==================== DOWNLOAD TAB ==================== */
var papers = PAPERS_JSON_PLACEHOLDER;
var dlState = papers.map(function() { return 'pending'; });

var saved = localStorage.getItem('paper_dl_state');
if (saved) {
    try {
        var arr = JSON.parse(saved);
        arr.forEach(function(s, i) { if (i < dlState.length) dlState[i] = s; });
    } catch(e) {}
}

var currentIndex = dlState.findIndex(function(s) { return s === 'pending'; });
if (currentIndex === -1) currentIndex = papers.length;
var batchTimer = null;
var currentFilter = 'all';
var batchRunning = false;

function saveDlState() { localStorage.setItem('paper_dl_state', JSON.stringify(dlState)); }

function updateDlStats() {
    var total = papers.length;
    var opened = dlState.filter(function(s) { return s === 'opened'; }).length;
    var skipped = dlState.filter(function(s) { return s === 'skipped'; }).length;
    var pending = total - opened - skipped;
    document.getElementById('totalCount').textContent = total;
    document.getElementById('openedCount').textContent = opened;
    document.getElementById('pendingCount').textContent = pending;
    document.getElementById('skippedCount').textContent = skipped;
    var pct = ((opened + skipped) / Math.max(total, 1) * 100).toFixed(1);
    document.getElementById('progressFill').style.width = pct + '%';
}

function setDlStatus(msg) { document.getElementById('statusBar').textContent = msg; }

function renderDl() {
    var tbody = document.getElementById('tbody');
    tbody.innerHTML = papers.map(function(p, i) {
        if (currentFilter === 'pending' && dlState[i] !== 'pending') return '';
        if (currentFilter !== 'all' && currentFilter !== 'pending' && p.publisher !== currentFilter) return '';
        var tagClass = 'tag-' + p.publisher;
        var icon = dlState[i] === 'opened' ? '&#9989;' : dlState[i] === 'skipped' ? '&#9197;' : '&#9203;';
        return '<tr id="row-' + i + '">' +
            '<td>' + p.year + '-' + String(p.index).padStart(3,'0') + '</td>' +
            '<td>' + p.year + '</td>' +
            '<td><a href="' + p.doi_url + '" target="_blank" title="' + p.doi + '">' + p.doi + '</a>' +
            (!p.has_direct_pdf ? '<span style="color:#f87171;font-size:11px;"> (manual)</span>' : '') + '</td>' +
            '<td><span class="tag ' + tagClass + '">' + p.publisher + '</span></td>' +
            '<td><a href="' + p.pdf_url + '" target="_blank" onclick="markOpened(' + i + ')" style="color:#4ade80;">Download</a>' +
            '&nbsp;<a href="#" onclick="markSkipped(' + i + ');return false;" style="color:#888;font-size:11px;">Skip</a></td>' +
            '<td class="status-icon">' + icon + '</td></tr>';
    }).join('');
    updateDlStats();
}

function markOpened(i) { dlState[i] = 'opened'; saveDlState(); setTimeout(renderDl, 100); }
function markSkipped(i) { dlState[i] = 'skipped'; saveDlState(); renderDl(); }

function findNextPending() {
    for (var i = 0; i < papers.length; i++) { if (dlState[i] === 'pending') return i; }
    return -1;
}

function openOne() {
    var i = findNextPending();
    if (i === -1) { stopBatch(); setDlStatus('All done!'); return false; }
    var p = papers[i];
    var a = document.createElement('a');
    a.href = p.pdf_url; a.target = '_blank'; a.rel = 'noopener';
    a.style.display = 'none'; document.body.appendChild(a); a.click(); document.body.removeChild(a);
    markOpened(i);
    var opened = dlState.filter(function(s) { return s === 'opened'; }).length;
    setDlStatus('Opening #' + opened + ': ' + p.doi);
    return true;
}

function startBatch() {
    stopBatch();
    var testWin = window.open('about:blank', '_blank');
    if (!testWin || testWin.closed) {
        setDlStatus('Popup blocked! Allow popups for this page and retry.');
        return;
    }
    testWin.close();
    var size = parseInt(document.getElementById('batchSize').value);
    var intv = parseInt(document.getElementById('interval').value);
    batchRunning = true;
    var count = 0;
    setDlStatus('Batch opening... (' + size + ' papers, ' + (intv/1000) + 's interval)');
    function next() {
        if (!batchRunning) return;
        if (count >= size || !openOne()) {
            batchRunning = false;
            var rem = dlState.filter(function(s) { return s === 'pending'; }).length;
            if (rem > 0) setDlStatus('Batch done (' + count + '). ' + rem + ' remaining, click Resume.');
            return;
        }
        count++;
        setTimeout(next, intv);
    }
    next();
}
function resumeBatch() { startBatch(); }
function stopBatch() { batchRunning = false; if (batchTimer) { clearInterval(batchTimer); batchTimer = null; } }
function resetAll() {
    if (!confirm('Reset all download progress?')) return;
    dlState.fill('pending'); saveDlState(); currentIndex = 0; renderDl(); setDlStatus('Reset.');
}
function setFilter(f) {
    currentFilter = f;
    document.querySelectorAll('.filter-btn').forEach(function(btn) {
        btn.classList.toggle('active', btn.textContent.toLowerCase().includes(f) ||
            (f === 'all' && btn.textContent === 'All') ||
            (f === 'pending' && btn.textContent === 'Pending'));
    });
    renderDl();
}

/* ==================== EXTRACTION TAB ==================== */

/* PDF.js setup */
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

/* Default prompts & columns */
var DEFAULT_SYS_PROMPT = "__SYS_PROMPT__";
var DEFAULT_USR_PROMPT = "__USR_PROMPT__";
var EXT_COLUMNS = "__COLUMNS__";

/* State */
var extFiles = [];
var extResults = [];
var extRunning = false;
var extAborted = false;

/* ---- Settings persistence ---- */
function saveExtSettings() {
    var s = {
        provider: document.getElementById('extProvider').value,
        apiKey: document.getElementById('extApiKey').value,
        model: document.getElementById('extModel').value,
        dpi: document.getElementById('extDpi').value,
        maxPages: document.getElementById('extMaxPages').value,
        delay: document.getElementById('extDelay').value,
        retries: document.getElementById('extRetries').value,
    };
    localStorage.setItem('ext_settings', JSON.stringify(s));
}
function loadExtSettings() {
    try {
        var s = JSON.parse(localStorage.getItem('ext_settings'));
        if (!s) return;
        if (s.provider) document.getElementById('extProvider').value = s.provider;
        if (s.apiKey) document.getElementById('extApiKey').value = s.apiKey;
        if (s.model) document.getElementById('extModel').value = s.model;
        if (s.dpi) document.getElementById('extDpi').value = s.dpi;
        if (s.maxPages) document.getElementById('extMaxPages').value = s.maxPages;
        if (s.delay) document.getElementById('extDelay').value = s.delay;
        if (s.retries) document.getElementById('extRetries').value = s.retries;
    } catch(e) {}
}
document.addEventListener('change', function(e) {
    if (e.target.closest('#tab-extract')) saveExtSettings();
});

/* ---- Provider change ---- */
function onProviderChange() {
    var p = document.getElementById('extProvider').value;
    var m = document.getElementById('extModel');
    if (p === 'google' && m.value.startsWith('google/')) {
        m.value = m.value.replace('google/', '');
    } else if (p === 'openrouter' && !m.value.startsWith('google/')) {
        m.value = 'google/' + m.value;
    }
}

/* ---- Prompt management ---- */
function togglePrompts() {
    var s = document.getElementById('promptSection');
    s.style.display = s.style.display === 'none' ? 'block' : 'none';
}
function resetPrompts() {
    document.getElementById('extSysPrompt').value = DEFAULT_SYS_PROMPT;
    document.getElementById('extUsrPrompt').value = DEFAULT_USR_PROMPT;
    document.querySelectorAll('.prompt-area').forEach(function(el) { el.classList.remove('edited'); });
}
document.addEventListener('input', function(e) {
    if (e.target.classList.contains('prompt-area')) e.target.classList.add('edited');
});

/* ---- File management ---- */
function onFilesSelected(input) {
    var files = Array.from(input.files).filter(function(f) {
        return f.name.toLowerCase().endsWith('.pdf');
    });
    if (files.length === 0) { addExtLog('No PDF files found in selection.'); return; }
    extFiles = files.map(function(f) {
        return { file: f, name: f.name, status: 'pending', error: null, devices: [] };
    });
    updateExtFileList();
    document.getElementById('extFileCount').textContent = extFiles.length + ' PDF file(s) selected';
    document.getElementById('extTotal').textContent = extFiles.length;
    addExtLog('Selected ' + extFiles.length + ' PDF files.');
    input.value = '';
}
function clearExtFiles() {
    extFiles = [];
    updateExtFileList();
    document.getElementById('extFileCount').textContent = 'No files selected';
    document.getElementById('extTotal').textContent = '0';
}
function updateExtFileList() {
    var el = document.getElementById('extFileList');
    el.innerHTML = extFiles.map(function(f, i) {
        var cls = f.status === 'processing' ? ' processing' : '';
        var icon = f.status === 'done' ? '&#9989;' :
                   f.status === 'error' ? '&#10060;' :
                   f.status === 'processing' ? '&#9881;' : '&#9203;';
        var extra = f.status === 'error' ? ' <span style="color:#f87171">' + (f.error||'').substring(0,40) + '</span>' : '';
        return '<div class="file-item' + cls + '"><span class="fname" title="' + f.name + '">' +
            (i+1) + '. ' + f.name + '</span><span class="fstatus">' + icon + ' ' + f.status + extra + '</span></div>';
    }).join('');
}

/* ---- PDF to images via PDF.js ---- */
async function pdfToImages(file, dpi, maxPages) {
    var data = new Uint8Array(await file.arrayBuffer());
    var pdf = await pdfjsLib.getDocument({ data: data }).promise;
    var n = Math.min(pdf.numPages, maxPages);
    var images = [];
    for (var i = 1; i <= n; i++) {
        var page = await pdf.getPage(i);
        var scale = dpi / 72;
        var vp = page.getViewport({ scale: scale });
        var canvas = document.createElement('canvas');
        canvas.width = vp.width;
        canvas.height = vp.height;
        var ctx = canvas.getContext('2d');
        await page.render({ canvasContext: ctx, viewport: vp }).promise;
        var b64 = canvas.toDataURL('image/jpeg', 0.85).split(',')[1];
        images.push(b64);
    }
    return images;
}

/* ---- LLM API call ---- */
async function callLLM(images, sysPrompt, usrPrompt, provider, apiKey, model, maxRetries) {
    var parts = [];
    for (var i = 0; i < images.length; i++) {
        parts.push({ type: 'text', text: '[Page ' + (i+1) + ' of ' + images.length + ']' });
        parts.push({ type: 'image_url', image_url: { url: 'data:image/jpeg;base64,' + images[i] } });
    }
    parts.push({ type: 'text', text: usrPrompt });

    var body = {
        model: model,
        messages: [
            { role: 'system', content: sysPrompt },
            { role: 'user', content: parts }
        ],
        max_tokens: 24000,
        temperature: 0
    };

    var url, headers;
    if (provider === 'google') {
        url = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions?key=' + encodeURIComponent(apiKey);
        headers = { 'Content-Type': 'application/json' };
    } else {
        url = 'https://openrouter.ai/api/v1/chat/completions';
        headers = { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + apiKey };
    }

    var lastError = null;
    for (var attempt = 0; attempt < maxRetries; attempt++) {
        try {
            var resp = await fetch(url, { method: 'POST', headers: headers, body: JSON.stringify(body) });
            if (!resp.ok) {
                var errText = await resp.text();
                lastError = 'HTTP ' + resp.status + ': ' + errText.substring(0, 200);
                if (resp.status === 429 && attempt < maxRetries - 1) {
                    var wait = (attempt + 1) * 15;
                    addExtLog('    Rate limited, waiting ' + wait + 's before retry...');
                    await sleep(wait * 1000);
                    continue;
                }
                throw new Error(lastError);
            }
            var data = await resp.json();
            if (!data.choices || !data.choices[0] || !data.choices[0].message || !data.choices[0].message.content) {
                lastError = 'Empty response from model';
                if (attempt < maxRetries - 1) {
                    addExtLog('    Empty response, retrying...');
                    await sleep((attempt + 1) * 5000);
                    continue;
                }
                throw new Error(lastError);
            }
            return data.choices[0].message.content;
        } catch(e) {
            lastError = e.message;
            if (attempt < maxRetries - 1 && !e.message.startsWith('HTTP 4')) {
                var w = (attempt + 1) * 10;
                addExtLog('    Error: ' + lastError.substring(0, 80) + ', retrying in ' + w + 's...');
                await sleep(w * 1000);
                continue;
            }
            throw e;
        }
    }
    throw new Error(lastError);
}

function sleep(ms) { return new Promise(function(r) { setTimeout(r, ms); }); }

/* ---- JSON parsing with repair ---- */
function tryParseJSON(raw) {
    raw = raw.replace(/^```(?:json)?\s*/m, '').replace(/\s*```\s*$/m, '');

    try { return JSON.parse(raw); } catch(e) {}

    var start = raw.indexOf('{');
    if (start === -1) return null;

    var depth = 0, inStr = false, esc = false, end = -1;
    for (var i = start; i < raw.length; i++) {
        var c = raw[i];
        if (esc) { esc = false; continue; }
        if (c === '\\' && inStr) { esc = true; continue; }
        if (c === '"') { inStr = !inStr; continue; }
        if (inStr) continue;
        if (c === '{') depth++;
        else if (c === '}') { depth--; if (depth === 0) { end = i; break; } }
    }

    if (end > start) {
        var s = raw.substring(start, end + 1);
        try { return JSON.parse(s); } catch(e) {}
        var cleaned = s.replace(/,\s*([}\]])/g, '$1');
        try { return JSON.parse(cleaned); } catch(e) {}
    }

    var sub = raw.substring(start);
    var repaired = repairJSON(sub);
    try { return JSON.parse(repaired); } catch(e) {}
    var cleaned2 = repaired.replace(/,\s*([}\]])/g, '$1');
    try { return JSON.parse(cleaned2); } catch(e) {}

    return null;
}

function repairJSON(raw) {
    var lastBrace = raw.lastIndexOf('}');
    if (lastBrace === -1) return raw;
    for (var i = lastBrace; i > 0; i--) {
        if (raw[i] === '}') {
            var candidate = raw.substring(0, i + 1);
            var openB = (candidate.match(/\[/g) || []).length - (candidate.match(/\]/g) || []).length;
            var openC = (candidate.match(/\{/g) || []).length - (candidate.match(/\}/g) || []).length;
            candidate += ']'.repeat(Math.max(0, openB)) + '}'.repeat(Math.max(0, openC));
            try { JSON.parse(candidate); return candidate; } catch(e) { continue; }
        }
    }
    return raw;
}

/* ---- Main extraction loop ---- */
async function startExtraction() {
    if (extRunning) return;

    var provider = document.getElementById('extProvider').value;
    var apiKey = document.getElementById('extApiKey').value.trim();
    var model = document.getElementById('extModel').value.trim();
    var dpi = parseInt(document.getElementById('extDpi').value) || 200;
    var maxPages = parseInt(document.getElementById('extMaxPages').value) || 30;
    var delay = parseInt(document.getElementById('extDelay').value);
    if (isNaN(delay)) delay = 5;
    var maxRetries = parseInt(document.getElementById('extRetries').value) || 3;
    var sysPrompt = document.getElementById('extSysPrompt').value;
    var usrPrompt = document.getElementById('extUsrPrompt').value;

    if (!apiKey) { addExtLog('ERROR: Please enter an API key.'); return; }
    if (!model) { addExtLog('ERROR: Please enter a model name.'); return; }
    if (extFiles.length === 0) { addExtLog('ERROR: No PDF files selected.'); return; }

    extRunning = true;
    extAborted = false;
    document.getElementById('extStartBtn').disabled = true;
    document.getElementById('extStopBtn').disabled = false;
    addExtLog('=== Starting extraction (' + provider + ' / ' + model + ') ===');
    addExtLog('DPI=' + dpi + ', MaxPages=' + maxPages + ', Delay=' + delay + 's, Retries=' + maxRetries);

    for (var idx = 0; idx < extFiles.length; idx++) {
        if (extAborted) break;
        var f = extFiles[idx];
        if (f.status === 'done') continue;

        f.status = 'processing';
        f.error = null;
        updateExtFileList();
        addExtLog('\n[' + (idx+1) + '/' + extFiles.length + '] ' + f.name);
        setExtStatus('Processing ' + f.name + '...');

        try {
            addExtLog('  Converting PDF to images (DPI=' + dpi + ')...');
            var images = await pdfToImages(f.file, dpi, maxPages);
            addExtLog('  Converted ' + images.length + ' page(s)');

            addExtLog('  Calling ' + provider + ' API (' + model + ')...');
            var raw = await callLLM(images, sysPrompt, usrPrompt, provider, apiKey, model, maxRetries);

            var parsed = tryParseJSON(raw);
            if (!parsed) {
                f.status = 'error';
                f.error = 'JSON parse failed';
                addExtLog('  ERROR: Failed to parse JSON response');
                addExtLog('  Raw (first 300 chars): ' + raw.substring(0, 300));
                updateExtStats(); updateExtFileList();
                if (delay > 0 && idx < extFiles.length - 1 && !extAborted) await sleep(delay * 1000);
                continue;
            }

            var devices = parsed.devices || [];
            var notes = parsed.paper_notes || '';
            f.devices = devices;
            f.status = 'done';

            if (devices.length === 0) {
                addExtLog('  No devices extracted. Notes: ' + notes);
                extResults.push({
                    filename: f.name, device_label: '', device_index: 0,
                    needs_SI: '', extraction_notes: notes || 'No devices extracted'
                });
            } else {
                addExtLog('  Extracted ' + devices.length + ' device(s)');
                for (var di = 0; di < devices.length; di++) {
                    var dev = devices[di];
                    var row = { filename: f.name, device_index: di + 1, needs_SI: 'No' };
                    row.extraction_notes = notes;
                    for (var key in dev) { if (dev.hasOwnProperty(key)) row[key] = dev[key]; }
                    extResults.push(row);
                }
            }
            updateExtResults();

        } catch(e) {
            f.status = 'error';
            f.error = e.message;
            addExtLog('  ERROR: ' + e.message);
        }

        updateExtStats();
        updateExtFileList();

        if (delay > 0 && idx < extFiles.length - 1 && !extAborted) {
            addExtLog('  Waiting ' + delay + 's...');
            await sleep(delay * 1000);
        }
    }

    extRunning = false;
    document.getElementById('extStartBtn').disabled = false;
    document.getElementById('extStopBtn').disabled = true;
    if (extAborted) {
        addExtLog('\n=== Extraction paused. Click "Start" to resume pending files. ===');
        setExtStatus('Paused. ' + extFiles.filter(function(f){return f.status==='pending';}).length + ' files remaining.');
    } else {
        var doneN = extFiles.filter(function(f){return f.status==='done';}).length;
        var errN = extFiles.filter(function(f){return f.status==='error';}).length;
        addExtLog('\n=== Extraction complete! Done: ' + doneN + ', Errors: ' + errN + ', Devices: ' + extResults.length + ' ===');
        setExtStatus('Complete! ' + doneN + ' files processed, ' + extResults.length + ' devices extracted.');
    }
}

function stopExtraction() {
    extAborted = true;
    addExtLog('Pausing after current file...');
}

/* ---- UI Updates ---- */
function addExtLog(msg) {
    var log = document.getElementById('extLog');
    var time = new Date().toLocaleTimeString();
    log.textContent += '[' + time + '] ' + msg + '\n';
    log.scrollTop = log.scrollHeight;
}
function setExtStatus(msg) { document.getElementById('extStatusBar').textContent = msg; }

function updateExtStats() {
    var total = extFiles.length;
    var done = extFiles.filter(function(f){return f.status==='done';}).length;
    var errs = extFiles.filter(function(f){return f.status==='error';}).length;
    document.getElementById('extTotal').textContent = total;
    document.getElementById('extDone').textContent = done;
    document.getElementById('extErrors').textContent = errs;
    document.getElementById('extEntries').textContent = extResults.length;
    var pct = total > 0 ? ((done + errs) / total * 100).toFixed(1) : 0;
    document.getElementById('extProgress').style.width = pct + '%';
}

function updateExtResults() {
    document.getElementById('extResultCount').textContent = extResults.length;
    document.getElementById('extEntries').textContent = extResults.length;
    if (extResults.length === 0) return;

    /* Dynamically discover all columns from actual data */
    var colSet = {};
    var colOrder = [];
    extResults.forEach(function(r) {
        for (var k in r) {
            if (r.hasOwnProperty(k) && !colSet[k]) {
                colSet[k] = true;
                colOrder.push(k);
            }
        }
    });

    /* Build dynamic header */
    var thead = document.getElementById('extResultsHead');
    thead.innerHTML = '<tr>' + colOrder.map(function(c) {
        return '<th>' + c + '</th>';
    }).join('') + '</tr>';

    /* Build rows (last 100 for performance) */
    var tbody = document.getElementById('extResultsBody');
    var show = extResults.slice(-100);
    tbody.innerHTML = show.map(function(r) {
        return '<tr>' + colOrder.map(function(c) {
            var v = r[c];
            if (v === null || v === undefined) v = '';
            v = String(v);
            var display = v.length > 50 ? v.substring(0, 47) + '...' : v;
            return '<td title="' + v.replace(/"/g, '&quot;') + '">' + display + '</td>';
        }).join('') + '</tr>';
    }).join('');
}

function clearExtResults() {
    if (extResults.length > 0 && !confirm('Clear all extracted results?')) return;
    extResults = [];
    updateExtResults();
    updateExtStats();
    addExtLog('Results cleared.');
}

/* ---- CSV Export ---- */
function exportCSV() {
    if (extResults.length === 0) { addExtLog('No results to export.'); return; }

    /* Discover all columns dynamically from data */
    var colSet = {};
    var headers = [];
    extResults.forEach(function(r) {
        for (var k in r) {
            if (r.hasOwnProperty(k) && !colSet[k]) {
                colSet[k] = true;
                headers.push(k);
            }
        }
    });

    var lines = [headers.join(',')];
    for (var i = 0; i < extResults.length; i++) {
        var r = extResults[i];
        var row = headers.map(function(h) {
            var v = r[h];
            if (v === null || v === undefined) v = '';
            v = String(v).replace(/"/g, '""');
            if (v.indexOf(',') !== -1 || v.indexOf('"') !== -1 || v.indexOf('\n') !== -1) {
                v = '"' + v + '"';
            }
            return v;
        });
        lines.push(row.join(','));
    }
    var csv = '\ufeff' + lines.join('\n');
    var blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'extracted_features.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
    addExtLog('Exported ' + extResults.length + ' rows (' + headers.length + ' columns) to CSV');
}

/* ==================== INIT ==================== */
renderDl();
resetPrompts();
loadExtSettings();
</script>
</body>
</html>"""

    html = html.replace('PAPERS_JSON_PLACEHOLDER', papers_json)
    html = html.replace('"__SYS_PROMPT__"', sys_prompt_js)
    html = html.replace('"__USR_PROMPT__"', usr_prompt_js)
    html = html.replace('"__COLUMNS__"', columns_js)

    return html


def main():
    if os.path.exists(DOI_LIST_FILE):
        entries = parse_doi_list(DOI_LIST_FILE)
        print(f"Parsed {len(entries)} papers from DOI list")

        pubs = {}
        for e in entries:
            pubs[e["publisher"]] = pubs.get(e["publisher"], 0) + 1
        for p, c in sorted(pubs.items(), key=lambda x: -x[1]):
            direct = "direct" if p in ("wiley", "acs", "science") else "manual"
            print(f"  {p}: {c} ({direct})")
    else:
        entries = []
        print(f"DOI list not found ({DOI_LIST_FILE}), download tab will be empty")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    html = generate_html(entries)
    outpath = os.path.join(OUTPUT_DIR, "paper_assistant.html")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nGenerated: {outpath}")
    print("Open in Chrome to use.")

    if sys.platform == "win32":
        os.startfile(os.path.abspath(outpath))
        print("Opened in browser!")


if __name__ == "__main__":
    main()
