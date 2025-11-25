"""
Fingerprint Grouping Report Generator with LLM Insights

This script processes a CSV file containing device fingerprint data,
generates AI-powered insights using Ollama (with intelligent fallback),
and creates a professional Word document report.

Usage:
    python llm_supported_report.py --input your_data.csv --output report_with_insights.docx
    
    Or with sample data for testing:
    python llm_supported_report.py --sample --output report_with_insights.docx
    
    Specify Ollama model:
    python llm_supported_report.py --input data.csv --model llama3.2

Required columns in CSV:
    - fingerprint
    - same_device_fingerprints  
    - priv_base_apps
    - priv_var_apps
    - build_type
    - device
"""

import pandas as pd
import json
import subprocess
import argparse
import os
from collections import Counter
from typing import Dict, List, Any
from datetime import datetime


# ============================================================================
# FINGERPRINT PARSING
# ============================================================================

def parse_fingerprint(fingerprint_value: str) -> str:
    """
    Parse fingerprint to extract the word between first and second "/",
    take last 3 characters, and capitalize them.
    
    Example: "google/sunfish/sunfish:11/RQ3A.210805.001.A1" -> "ISH"
             (sunfish -> ish -> ISH)
    """
    if not isinstance(fingerprint_value, str):
        return str(fingerprint_value)
    
    parts = fingerprint_value.split('/')
    
    if len(parts) >= 2:
        word = parts[1]
        if len(word) >= 3:
            return word[-3:].upper()
        else:
            return word.upper()
    
    return str(fingerprint_value)


# ============================================================================
# DATA PROCESSING
# ============================================================================

def get_unique_values(df: pd.DataFrame, column: str) -> List[Any]:
    """Get unique non-null values from a column."""
    return df[column].dropna().unique().tolist()


def flatten_apps(apps_list: List) -> List[str]:
    """Flatten list of apps, handling string representations of lists."""
    flattened = []
    for item in apps_list:
        if isinstance(item, str):
            try:
                parsed = json.loads(item.replace("'", '"'))
                if isinstance(parsed, list):
                    flattened.extend(parsed)
                else:
                    flattened.append(str(parsed))
            except (json.JSONDecodeError, ValueError):
                flattened.append(item)
        elif isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(str(item))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_flattened = []
    for item in flattened:
        if item not in seen:
            seen.add(item)
            unique_flattened.append(item)
    
    return unique_flattened


def process_fingerprint_data(df: pd.DataFrame) -> Dict:
    """
    Main processing function that implements the grouping and filtering logic.
    """
    results = {}
    
    unique_devices = get_unique_values(df, 'device')
    print(f"Found {len(unique_devices)} unique devices: {unique_devices}")
    
    for device in unique_devices:
        results[device] = {}
        device_data = df[df['device'] == device]
        unique_build_types = get_unique_values(device_data, 'build_type')
        print(f"  Device {device}: Found {len(unique_build_types)} build types: {unique_build_types}")
        
        for build_type in unique_build_types:
            build_data = device_data[device_data['build_type'] == build_type]
            
            unique_fingerprints = get_unique_values(build_data, 'fingerprint')
            unique_same_device_fps = get_unique_values(build_data, 'same_device_fingerprints')
            
            unique_fingerprints_parsed = [parse_fingerprint(fp) for fp in unique_fingerprints]
            unique_same_device_fps_parsed = [parse_fingerprint(sdp) for sdp in unique_same_device_fps]
            
            print(f"    Build Type {build_type}: {len(unique_fingerprints)} fingerprints, {len(unique_same_device_fps)} same_device_fps")
            
            results[device][build_type] = {
                'unique_fingerprints': unique_fingerprints,
                'unique_fingerprints_parsed': unique_fingerprints_parsed,
                'unique_same_device_fps': unique_same_device_fps,
                'unique_same_device_fps_parsed': unique_same_device_fps_parsed,
                'combinations': []
            }
            
            for fp_idx, fp in enumerate(unique_fingerprints):
                fp_data = build_data[build_data['fingerprint'] == fp]
                fp_parsed = unique_fingerprints_parsed[fp_idx]
                
                for sdp_idx, sdp in enumerate(unique_same_device_fps):
                    filtered_rows = fp_data[fp_data['same_device_fingerprints'] == sdp]
                    sdp_parsed = unique_same_device_fps_parsed[sdp_idx]
                    
                    if not filtered_rows.empty:
                        priv_base_apps = flatten_apps(filtered_rows['priv_base_apps'].dropna().tolist())
                        priv_var_apps = flatten_apps(filtered_rows['priv_var_apps'].dropna().tolist())
                        
                        combination = {
                            'fingerprint': fp,
                            'fingerprint_parsed': fp_parsed,
                            'same_device_fingerprint': sdp,
                            'same_device_fingerprint_parsed': sdp_parsed,
                            'priv_base_apps': priv_base_apps,
                            'priv_var_apps': priv_var_apps
                        }
                        
                        results[device][build_type]['combinations'].append(combination)
    
    return results


# ============================================================================
# LLM INSIGHT GENERATION
# ============================================================================

def analyze_app_distribution(apps_list: List[str]) -> Dict:
    """Analyze the distribution and patterns in an apps list."""
    counter = Counter(apps_list)
    
    return {
        'total_unique': len(counter),
        'total_occurrences': len(apps_list),
        'most_common': counter.most_common(5),
        'frequency_distribution': dict(counter),
        'single_occurrence_apps': [app for app, count in counter.items() if count == 1],
        'dominant_apps': [app for app, count in counter.items() if count > 1]
    }


def call_ollama(prompt: str, model: str = "llama3.2") -> str:
    """Call Ollama API to generate text."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"    Ollama error: {result.stderr[:100]}...")
            return None
            
    except subprocess.TimeoutExpired:
        print("    Ollama timed out")
        return None
    except FileNotFoundError:
        print("    Ollama not found, using intelligent fallback")
        return None
    except Exception as e:
        print(f"    Ollama error: {e}")
        return None


def build_ollama_prompt(device: str, build_type: str, data: Dict) -> str:
    """Build a comprehensive prompt for LLM analysis."""
    combinations = data.get('combinations', [])
    
    all_base_apps = []
    all_var_apps = []
    combination_details = []
    
    for combo in combinations:
        fp = combo.get('fingerprint_parsed', 'Unknown')
        sdp = combo.get('same_device_fingerprint_parsed', 'Unknown')
        base_apps = combo.get('priv_base_apps', [])
        var_apps = combo.get('priv_var_apps', [])
        
        all_base_apps.extend(base_apps)
        all_var_apps.extend(var_apps)
        
        combination_details.append({
            'fingerprint': fp,
            'same_device_fp': sdp,
            'base_apps': base_apps,
            'var_apps': var_apps
        })
    
    base_analysis = analyze_app_distribution(all_base_apps)
    var_analysis = analyze_app_distribution(all_var_apps)
    
    prompt = f"""You are a senior Android system analyst reviewing privileged application data.

TASK: Analyze data for Device "{device}" with Build Type "{build_type}".

=== DATA SUMMARY ===
Total Combinations: {len(combinations)}
Unique Fingerprints: {data.get('unique_fingerprints_parsed', [])}
Unique Same-Device FPs: {data.get('unique_same_device_fps_parsed', [])}

=== COMBINATIONS ===
"""
    
    for i, detail in enumerate(combination_details, 1):
        prompt += f"""
{i}. FP: {detail['fingerprint']} | SDP: {detail['same_device_fp']}
   Base Apps ({len(detail['base_apps'])}): {', '.join(detail['base_apps'][:5])}{'...' if len(detail['base_apps']) > 5 else ''}
   Var Apps ({len(detail['var_apps'])}): {', '.join(detail['var_apps'][:5])}{'...' if len(detail['var_apps']) > 5 else ''}
"""

    prompt += f"""
=== STATISTICS ===
PRIV BASE APPS: {base_analysis['total_unique']} unique, {base_analysis['total_occurrences']} total
  Most common: {base_analysis['most_common'][:3]}
  Dominant (>1 occurrence): {base_analysis['dominant_apps']}

PRIV VARIANT APPS: {var_analysis['total_unique']} unique, {var_analysis['total_occurrences']} total
  Most common: {var_analysis['most_common'][:3]}
  Dominant (>1 occurrence): {var_analysis['dominant_apps']}

=== INSTRUCTIONS ===
Provide analysis in this format:

**PRIV BASE APPS INSIGHT** (2-3 sentences about patterns, dominance, anomalies)

**PRIV VARIANT APPS INSIGHT** (2-3 sentences)

**COMBINED ANALYSIS** (3-4 sentences about relationships and overall patterns)

**KEY FINDINGS**
• Finding 1
• Finding 2
• Finding 3

Keep it concise and data-driven."""

    return prompt


def generate_intelligent_fallback(device: str, build_type: str, data: Dict) -> str:
    """Generate intelligent insights when LLM is unavailable."""
    combinations = data.get('combinations', [])
    
    all_base_apps = []
    all_var_apps = []
    combo_sizes = []
    
    for combo in combinations:
        base = combo.get('priv_base_apps', [])
        var = combo.get('priv_var_apps', [])
        all_base_apps.extend(base)
        all_var_apps.extend(var)
        combo_sizes.append((len(base), len(var)))
    
    base_counter = Counter(all_base_apps)
    var_counter = Counter(all_var_apps)
    
    base_unique = len(base_counter)
    var_unique = len(var_counter)
    base_total = len(all_base_apps)
    var_total = len(all_var_apps)
    
    base_dominant = [app for app, count in base_counter.items() if count > 1]
    var_dominant = [app for app, count in var_counter.items() if count > 1]
    base_single = [app for app, count in base_counter.items() if count == 1]
    
    avg_base = base_total / len(combinations) if combinations else 0
    avg_var = var_total / len(combinations) if combinations else 0
    
    insights = []
    
    # PRIV BASE APPS INSIGHT
    insights.append("**PRIV BASE APPS INSIGHT**")
    if base_dominant:
        dominant_str = ', '.join(base_dominant[:3])
        insights.append(f"Analysis identified {len(base_dominant)} dominant base app(s) appearing across multiple combinations: {dominant_str}. These represent core privileged applications consistently required for this device configuration.")
    elif base_unique == base_total and base_unique > 0:
        insights.append(f"All {base_unique} base apps are unique across combinations, indicating highly specialized privileged application requirements for each fingerprint pairing. No single app dominates the configuration.")
    else:
        insights.append(f"The {base_unique} unique base apps show a distributed pattern across {len(combinations)} combinations, averaging {avg_base:.1f} apps per combination.")
    
    if len(base_single) > base_unique * 0.7 and base_unique > 0:
        insights.append(f"Notable: {len(base_single)} apps ({len(base_single)*100//max(base_unique,1)}%) appear only once, suggesting fingerprint-specific requirements.")
    
    # PRIV VARIANT APPS INSIGHT
    insights.append("\n**PRIV VARIANT APPS INSIGHT**")
    if var_dominant:
        dominant_str = ', '.join(var_dominant[:3])
        insights.append(f"Variant apps show {len(var_dominant)} recurring application(s): {dominant_str}. These likely represent shared system components across device variants.")
    elif var_unique == var_total and var_unique > 0:
        insights.append(f"Each of the {var_unique} variant apps is unique to its combination, indicating distinct customization per fingerprint configuration.")
    else:
        insights.append(f"The variant app distribution shows {var_unique} unique apps averaging {avg_var:.1f} per combination.")
    
    # COMBINED ANALYSIS
    insights.append("\n**COMBINED ANALYSIS**")
    if base_unique > var_unique:
        insights.append(f"Base apps ({base_unique}) show more diversity than variant apps ({var_unique}), suggesting privileged base configurations are more variable than system variants for this build type.")
    elif var_unique > base_unique:
        insights.append(f"Variant apps ({var_unique}) exceed base app diversity ({base_unique}), indicating system customization varies more than core privileged app requirements.")
    else:
        insights.append(f"Base and variant apps show similar diversity ({base_unique} each), suggesting balanced configuration complexity.")
    
    if combo_sizes:
        max_combo = max(combo_sizes, key=lambda x: x[0] + x[1])
        min_combo = min(combo_sizes, key=lambda x: x[0] + x[1])
        if max_combo != min_combo:
            insights.append(f"Combination complexity ranges from {sum(min_combo)} to {sum(max_combo)} total apps, indicating variable configuration requirements.")
    
    # KEY FINDINGS
    insights.append("\n**KEY FINDINGS**")
    findings = []
    
    if base_dominant:
        findings.append(f"• Core dependencies: {', '.join(base_dominant[:2])} appear in multiple configurations")
    if len(base_single) > 2:
        findings.append(f"• {len(base_single)} specialized base apps suggest fingerprint-specific requirements")
    if avg_base > 2:
        findings.append(f"• High base app density ({avg_base:.1f} avg) indicates complex privileged requirements")
    if var_unique < base_unique and var_unique > 0:
        findings.append(f"• Variant apps are more standardized than base apps across this configuration")
    
    if not findings:
        findings.append(f"• Configuration shows standard distribution with {len(combinations)} combinations")
        findings.append(f"• No single app dominates - distributed privileged requirements")
    
    insights.extend(findings)
    
    return "\n".join(insights)


def generate_device_insight(device: str, build_type: str, data: Dict, model: str = "llama3.2") -> Dict:
    """Generate comprehensive insights for a device/build type combination."""
    combinations = data.get('combinations', [])
    
    all_base_apps = []
    all_var_apps = []
    
    for combo in combinations:
        all_base_apps.extend(combo.get('priv_base_apps', []))
        all_var_apps.extend(combo.get('priv_var_apps', []))
    
    base_stats = analyze_app_distribution(all_base_apps)
    var_stats = analyze_app_distribution(all_var_apps)
    
    # Try LLM first
    prompt = build_ollama_prompt(device, build_type, data)
    llm_response = call_ollama(prompt, model)
    
    # Fallback to intelligent analysis
    if llm_response is None:
        llm_response = generate_intelligent_fallback(device, build_type, data)
    
    return {
        'device': device,
        'build_type': build_type,
        'statistics': {
            'total_combinations': len(combinations),
            'unique_fingerprints': data.get('unique_fingerprints_parsed', []),
            'unique_same_device_fps': data.get('unique_same_device_fps_parsed', []),
            'priv_base_apps': {
                'total_unique': base_stats['total_unique'],
                'total_occurrences': base_stats['total_occurrences'],
                'most_common': base_stats['most_common'],
                'dominant_apps': base_stats['dominant_apps']
            },
            'priv_var_apps': {
                'total_unique': var_stats['total_unique'],
                'total_occurrences': var_stats['total_occurrences'],
                'most_common': var_stats['most_common'],
                'dominant_apps': var_stats['dominant_apps']
            }
        },
        'llm_insight': llm_response,
        'combinations_data': combinations
    }


def generate_all_insights(results: Dict, model: str = "llama3.2") -> Dict:
    """Generate insights for all device/build combinations."""
    all_insights = {
        'generated_at': str(datetime.now()),
        'model_used': model,
        'device_insights': []
    }
    
    for device, build_types in results.items():
        for build_type, data in build_types.items():
            print(f"  Generating insight for {device} / {build_type}...")
            insight = generate_device_insight(device, build_type, data, model)
            all_insights['device_insights'].append(insight)
    
    return all_insights


# ============================================================================
# SAMPLE DATA
# ============================================================================

def create_sample_data() -> pd.DataFrame:
    """Create sample data with realistic fingerprints for testing."""
    data = {
        'fingerprint': [
            "google/sunfish/sunfish:11/RQ3A.210805",
            "google/sunfish/sunfish:11/RQ3A.210805",
            "google/sunfish/sunfish:11/RQ3A.210805",
            "google/redfin/redfin:11/RQ3A.210705",
            "google/redfin/redfin:11/RQ3A.210705",
            "google/bramble/bramble:11/RQ3A.210605",
            "google/bramble/bramble:11/RQ3A.210605",
            "google/sunfish/sunfish:11/RQ3A.210805",
            "google/sunfish/sunfish:11/RQ3A.210805",
            "google/redfin/redfin:11/RQ3A.210705",
            "google/redfin/redfin:11/RQ3A.210705",
        ],
        'same_device_fingerprints': [
            "google/oriole/oriole:12/SQ1D.220105",
            "google/raven/raven:12/SQ1D.220205",
            "google/panther/panther:13/TQ1A.230305",
            "google/oriole/oriole:12/SQ1D.220105",
            "google/raven/raven:12/SQ1D.220205",
            "google/raven/raven:12/SQ1D.220205",
            "google/panther/panther:13/TQ1A.230305",
            "google/oriole/oriole:12/SQ1D.220105",
            "google/raven/raven:12/SQ1D.220205",
            "google/oriole/oriole:12/SQ1D.220105",
            "google/panther/panther:13/TQ1A.230305",
        ],
        'priv_base_apps': [
            "['com.google.camera', 'com.google.dialer']", 
            "['com.google.messages']", 
            "['com.google.photos', 'com.google.drive']",
            "['com.google.calendar']", 
            "['com.google.maps', 'com.google.chrome']", 
            "['com.google.assistant']", 
            "['com.google.translate', 'com.google.docs', 'com.google.sheets']",
            "['com.google.gmail']", 
            "['com.google.meet']", 
            "['com.google.keep']", 
            "['com.google.files']"
        ],
        'priv_var_apps': [
            "['com.android.systemui']", 
            "['com.android.settings', 'com.android.launcher']", 
            "['com.android.bluetooth']",
            "['com.android.wifi', 'com.android.nfc']", 
            "['com.android.phone']", 
            "['com.android.contacts']", 
            "['com.android.calculator', 'com.android.clock']",
            "['com.android.gallery']", 
            "['com.android.music']", 
            "['com.android.browser']", 
            "['com.android.email']"
        ],
        'build_type': ['userdebug', 'userdebug', 'userdebug', 'userdebug', 'userdebug', 
                       'userdebug', 'userdebug', 'user', 'user', 'user', 'user'],
        'device': ['Pixel_4a'] * 11
    }
    
    data2 = {
        'fingerprint': [
            "samsung/beyond/beyond1:10/QP1A.190711",
            "samsung/canvas/canvas:10/QP1A.190811",
            "samsung/beyond/beyond1:10/QP1A.190711",
            "samsung/canvas/canvas:10/QP1A.190811"
        ],
        'same_device_fingerprints': [
            "samsung/gts7/gts7xl:11/RP1A.200720",
            "samsung/gts7/gts7xl:11/RP1A.200720",
            "samsung/a52/a52q:11/RP1A.200820",
            "samsung/a52/a52q:11/RP1A.200820"
        ],
        'priv_base_apps': [
            "['com.samsung.camera']", 
            "['com.samsung.gallery', 'com.samsung.notes']", 
            "['com.samsung.browser']", 
            "['com.samsung.health']"
        ],
        'priv_var_apps': [
            "['com.sec.android.app.launcher']", 
            "['com.sec.android.app.clock']", 
            "['com.sec.android.app.calculator']", 
            "['com.sec.android.app.music']"
        ],
        'build_type': ['eng', 'eng', 'eng', 'eng'],
        'device': ['Galaxy_S10'] * 4
    }
    
    return pd.concat([pd.DataFrame(data), pd.DataFrame(data2)], ignore_index=True)


# ============================================================================
# JAVASCRIPT DOCUMENT GENERATOR (with LLM Insights)
# ============================================================================

def get_js_generator_code() -> str:
    """Returns the JavaScript code for generating the Word document with insights."""
    return '''
const {
    Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
    Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType, 
    ShadingType, VerticalAlign, PageNumber, PageBreak
} = require('docx');
const fs = require('fs');

const insightsData = JSON.parse(fs.readFileSync('insights_data.json', 'utf8'));

const colors = {
    primary: "1F4E79",
    secondary: "2E75B6",
    accent: "5B9BD5",
    headerBg: "D6E3F0",
    tableBorder: "B4C6E7",
    altRow: "F2F7FB",
    insightBg: "FFF8E7",
    insightBorder: "F5C518",
    text: "333333",
    lightText: "666666",
    success: "2E7D32",
    warning: "F57C00"
};

const tableBorder = { style: BorderStyle.SINGLE, size: 8, color: colors.tableBorder };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };
const headerBorders = { 
    top: { style: BorderStyle.SINGLE, size: 12, color: colors.primary }, 
    bottom: { style: BorderStyle.SINGLE, size: 12, color: colors.primary }, 
    left: tableBorder, right: tableBorder 
};

function createHeaderCell(text, width) {
    return new TableCell({
        borders: headerBorders,
        width: { size: width, type: WidthType.DXA },
        shading: { fill: colors.headerBg, type: ShadingType.CLEAR },
        verticalAlign: VerticalAlign.CENTER,
        children: [
            new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 100, after: 100 },
                children: [new TextRun({ text, bold: true, size: 24, color: colors.primary, font: "Arial" })]
            })
        ]
    });
}

function createDataCell(content, width, isAltRow = false) {
    const children = [];
    if (Array.isArray(content) && content.length > 0) {
        content.forEach(item => {
            children.push(new Paragraph({
                spacing: { before: 60, after: 60 },
                indent: { left: 200 },
                children: [new TextRun({ text: "• " + item, size: 22, color: colors.text, font: "Arial" })]
            }));
        });
    } else {
        children.push(new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 80, after: 80 },
            children: [new TextRun({ text: "—", size: 22, color: "999999", font: "Arial" })]
        }));
    }
    return new TableCell({
        borders: cellBorders,
        width: { size: width, type: WidthType.DXA },
        shading: isAltRow ? { fill: colors.altRow, type: ShadingType.CLEAR } : undefined,
        verticalAlign: VerticalAlign.CENTER,
        children
    });
}

function createAppsTable(privBaseApps, privVarApps) {
    const colWidths = [4680, 4680];
    return new Table({
        columnWidths: colWidths,
        rows: [
            new TableRow({ tableHeader: true, children: [
                createHeaderCell("Priv Base Apps", colWidths[0]),
                createHeaderCell("Priv Variant Apps", colWidths[1])
            ]}),
            new TableRow({ children: [
                createDataCell(privBaseApps, colWidths[0]),
                createDataCell(privVarApps, colWidths[1])
            ]})
        ]
    });
}

function createInsightBox(insightText, statistics) {
    const paragraphs = [];
    
    paragraphs.push(new Paragraph({
        spacing: { before: 300, after: 150 },
        children: [new TextRun({ text: "📊 AI-Generated Insight Summary", bold: true, size: 26, color: colors.warning, font: "Arial" })]
    }));
    
    const baseStats = statistics.priv_base_apps || {};
    const varStats = statistics.priv_var_apps || {};
    
    paragraphs.push(new Paragraph({
        spacing: { before: 100, after: 100 },
        shading: { fill: colors.altRow, type: ShadingType.CLEAR },
        children: [
            new TextRun({ text: "Quick Stats: ", bold: true, size: 20, color: colors.text, font: "Arial" }),
            new TextRun({ text: (statistics.total_combinations || 0) + " combinations | " + (baseStats.total_unique || 0) + " unique base apps | " + (varStats.total_unique || 0) + " unique variant apps", size: 20, color: colors.text, font: "Arial" })
        ]
    }));
    
    if (baseStats.dominant_apps && baseStats.dominant_apps.length > 0) {
        paragraphs.push(new Paragraph({
            spacing: { before: 80, after: 80 },
            children: [
                new TextRun({ text: "Dominant Base Apps: ", bold: true, size: 20, color: colors.success, font: "Arial" }),
                new TextRun({ text: baseStats.dominant_apps.join(", "), size: 20, color: colors.text, font: "Arial" })
            ]
        }));
    }
    
    if (varStats.dominant_apps && varStats.dominant_apps.length > 0) {
        paragraphs.push(new Paragraph({
            spacing: { before: 80, after: 80 },
            children: [
                new TextRun({ text: "Dominant Variant Apps: ", bold: true, size: 20, color: colors.success, font: "Arial" }),
                new TextRun({ text: varStats.dominant_apps.join(", "), size: 20, color: colors.text, font: "Arial" })
            ]
        }));
    }
    
    if (insightText && insightText.length > 0) {
        const sections = insightText.split(/\\n\\n+|\\n(?=\\*\\*)/);
        sections.forEach((section, idx) => {
            if (section.trim()) {
                const isHeader = section.trim().match(/^\\*\\*.*\\*\\*$/);
                paragraphs.push(new Paragraph({
                    spacing: { before: idx === 0 ? 150 : 80, after: 80 },
                    children: [new TextRun({
                        text: section.trim().replace(/\\*\\*/g, ''),
                        bold: isHeader ? true : false,
                        size: isHeader ? 22 : 20,
                        color: isHeader ? colors.secondary : colors.text,
                        font: "Arial"
                    })]
                }));
            }
        });
    }
    
    paragraphs.push(new Paragraph({
        spacing: { before: 200, after: 200 },
        border: { bottom: { style: BorderStyle.DASHED, size: 4, color: colors.insightBorder } },
        children: [new TextRun({ text: "" })]
    }));
    
    return paragraphs;
}

function createDivider() {
    return new Paragraph({
        spacing: { before: 300, after: 300 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: colors.tableBorder } },
        children: [new TextRun({ text: "" })]
    });
}

function generateContent() {
    const deviceInsights = insightsData.device_insights || [];
    let allContent = [];
    
    allContent.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 800, after: 300 },
        children: [new TextRun({ text: "Fingerprint Grouping Report", bold: true, size: 52, color: colors.primary, font: "Arial" })]
    }));
    
    allContent.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 50, after: 100 },
        children: [new TextRun({ text: "with AI-Powered Insights", size: 28, italics: true, color: colors.secondary, font: "Arial" })]
    }));
    
    allContent.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 100, after: 600 },
        children: [new TextRun({ text: "Generated: " + new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }), size: 22, color: colors.lightText, font: "Arial" })]
    }));
    
    allContent.push(new Paragraph({
        spacing: { before: 100, after: 400 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 18, color: colors.primary } },
        children: [new TextRun({ text: "" })]
    }));
    
    deviceInsights.forEach((deviceInsight, deviceIdx) => {
        const device = deviceInsight.device;
        const buildType = deviceInsight.build_type;
        const statistics = deviceInsight.statistics || {};
        const llmInsight = deviceInsight.llm_insight || '';
        const combinations = deviceInsight.combinations_data || [];
        
        if (deviceIdx > 0) {
            allContent.push(new Paragraph({ children: [new PageBreak()] }));
        }
        
        allContent.push(new Paragraph({
            heading: HeadingLevel.HEADING_1,
            spacing: { before: 300, after: 150 },
            children: [new TextRun({ text: "Device: " + device, bold: true, size: 36, color: colors.primary, font: "Arial" })]
        }));
        
        allContent.push(new Paragraph({
            heading: HeadingLevel.HEADING_2,
            spacing: { before: 100, after: 200 },
            children: [new TextRun({ text: "Build Type: " + buildType, bold: true, size: 30, color: colors.secondary, font: "Arial" })]
        }));
        
        const insightParagraphs = createInsightBox(llmInsight, statistics);
        allContent = allContent.concat(insightParagraphs);
        
        combinations.forEach((combo, comboIdx) => {
            const fp = combo.fingerprint_parsed || combo.fingerprint || 'Unknown';
            const sdp = combo.same_device_fingerprint_parsed || combo.same_device_fingerprint || 'Unknown';
            
            if (comboIdx > 0) allContent.push(createDivider());
            
            allContent.push(new Paragraph({
                spacing: { before: 200, after: 150 },
                children: [
                    new TextRun({ text: "Fingerprint: ", bold: true, size: 24, color: colors.secondary, font: "Arial" }),
                    new TextRun({ text: String(fp), size: 24, color: colors.text, font: "Arial" }),
                    new TextRun({ text: "   |   ", size: 24, color: colors.tableBorder, font: "Arial" }),
                    new TextRun({ text: "Same Device Fingerprint: ", bold: true, size: 24, color: colors.secondary, font: "Arial" }),
                    new TextRun({ text: String(sdp), size: 24, color: colors.text, font: "Arial" })
                ]
            }));
            
            allContent.push(new Paragraph({ spacing: { before: 100, after: 100 }, children: [new TextRun({ text: "" })] }));
            allContent.push(createAppsTable(combo.priv_base_apps || [], combo.priv_var_apps || []));
        });
    });
    
    return allContent;
}

function generateDocument() {
    const content = generateContent();
    
    const doc = new Document({
        styles: { default: { document: { run: { font: "Arial", size: 22 } } } },
        sections: [{
            properties: { page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }, pageNumbers: { start: 1 } } },
            headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Fingerprint Report | AI Insights", size: 18, color: "999999", font: "Arial" })] })] }) },
            footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
                new TextRun({ text: "Page ", size: 18, color: "999999", font: "Arial" }),
                new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "999999", font: "Arial" }),
                new TextRun({ text: " of ", size: 18, color: "999999", font: "Arial" }),
                new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: "999999", font: "Arial" })
            ] })] }) },
            children: content
        }]
    });
    
    const outputFile = process.argv[2] || 'report_with_insights.docx';
    Packer.toBuffer(doc).then(buffer => {
        fs.writeFileSync(outputFile, buffer);
        console.log('Document generated: ' + outputFile);
    });
}

generateDocument();
'''


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Fingerprint Grouping Report with LLM Insights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python llm_supported_report.py --input data.csv --output report.docx
    python llm_supported_report.py --sample --output test_report.docx
    python llm_supported_report.py --input data.csv --model llama3.2
        """
    )
    parser.add_argument('--input', '-i', help='Input CSV file path')
    parser.add_argument('--output', '-o', default='report_with_insights.docx', help='Output Word document path')
    parser.add_argument('--sample', '-s', action='store_true', help='Use sample data for testing')
    parser.add_argument('--model', '-m', default='llama3.2', help='Ollama model to use (default: llama3.2)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FINGERPRINT REPORT WITH LLM INSIGHTS")
    print("=" * 60)
    
    # Load data
    if args.sample:
        print("\nUsing sample data for testing...")
        df = create_sample_data()
    elif args.input:
        print(f"\nLoading data from {args.input}...")
        if not os.path.exists(args.input):
            print(f"ERROR: File not found: {args.input}")
            return 1
        df = pd.read_csv(args.input)
    else:
        print("\nNo input specified. Using sample data...")
        df = create_sample_data()
    
    # Validate columns
    required_columns = ['fingerprint', 'same_device_fingerprints', 'priv_base_apps', 
                        'priv_var_apps', 'build_type', 'device']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        return 1
    
    print(f"Loaded {len(df)} rows")
    
    # Process data
    print("\n" + "-" * 60)
    print("PROCESSING DATA...")
    print("-" * 60)
    
    results = process_fingerprint_data(df)
    
    # Generate insights
    print("\n" + "-" * 60)
    print(f"GENERATING LLM INSIGHTS (model: {args.model})...")
    print("-" * 60)
    
    insights = generate_all_insights(results, args.model)
    
    # Write data for JS
    with open('insights_data.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    # Write JS file
    js_path = 'generate_report_llm.js'
    with open(js_path, 'w') as f:
        f.write(get_js_generator_code())
    
    # Generate Word document
    print("\n" + "-" * 60)
    print("GENERATING WORD DOCUMENT...")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            ['node', js_path, args.output],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"\n✓ Report generated successfully: {args.output}")
        else:
            print(f"ERROR: {result.stderr}")
            return 1
            
    except FileNotFoundError:
        print("ERROR: Node.js not found. Please install Node.js.")
        return 1
    except subprocess.TimeoutExpired:
        print("ERROR: Document generation timed out")
        return 1
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total devices: {len(results)}")
    print(f"Total device/build sections: {len(insights['device_insights'])}")
    total_combos = sum(len(i['combinations_data']) for i in insights['device_insights'])
    print(f"Total combinations: {total_combos}")
    print(f"Output file: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
