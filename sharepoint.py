"""
Fingerprint Grouping Report Generator (Non-LLM Version)

Features:
- Frequentist statistics and graphs (overall + per device)
- Top 10 apps and top 10 repeating combinations
- Clean, professional Word document output

Usage:
    python report.py --input your_data.csv --output report.docx
    python report.py --sample --output report.docx
"""

import pandas as pd
import json
import subprocess
import argparse
import os
import base64
from collections import Counter
from typing import Dict, List, Any, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# FINGERPRINT PARSING
# ============================================================================

def parse_fingerprint(fingerprint_value: str) -> str:
    """Parse fingerprint: extract word between 1st and 2nd "/", take last 3 chars, capitalize."""
    if not isinstance(fingerprint_value, str):
        return str(fingerprint_value)
    
    parts = fingerprint_value.split('/')
    if len(parts) >= 2:
        word = parts[1]
        return word[-3:].upper() if len(word) >= 3 else word.upper()
    return str(fingerprint_value)


# ============================================================================
# DATA PROCESSING
# ============================================================================

def get_unique_values(df: pd.DataFrame, column: str) -> List[Any]:
    return df[column].dropna().unique().tolist()


def parse_app_string(app_str: Any) -> List[str]:
    """Parse app string like "['app1', 'app2']" into list of strings."""
    if pd.isna(app_str) or app_str is None:
        return []
    if isinstance(app_str, list):
        return [str(x) for x in app_str]
    if isinstance(app_str, str):
        try:
            parsed = json.loads(app_str.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            return [str(parsed)]
        except:
            return [app_str] if app_str.strip() else []
    return [str(app_str)]


def flatten_apps(apps_list: List) -> List[str]:
    """Flatten list of apps, handling nested structures."""
    flattened = []
    for item in apps_list:
        flattened.extend(parse_app_string(item))
    seen = set()
    return [x for x in flattened if not (x in seen or seen.add(x))]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess: parse fingerprints, remove self-comparisons and duplicates."""
    df = df.copy()
    
    print("  Parsing fingerprints...")
    df['fingerprint_parsed'] = df['fingerprint'].apply(parse_fingerprint)
    df['same_device_fingerprint_parsed'] = df['same_device_fingerprints'].apply(parse_fingerprint)
    
    # Parse apps into actual lists for statistics
    df['priv_base_apps_list'] = df['priv_base_apps'].apply(parse_app_string)
    df['priv_var_apps_list'] = df['priv_var_apps'].apply(parse_app_string)
    
    original_count = len(df)
    
    # Remove self-comparisons
    self_mask = df['fingerprint_parsed'] == df['same_device_fingerprint_parsed']
    removed_self = self_mask.sum()
    df = df[~self_mask]
    if removed_self > 0:
        print(f"  Removed {removed_self} self-comparison rows (FP == SDP)")
    
    # Merge duplicates
    dup_cols = ['device', 'build_type', 'fingerprint_parsed', 'same_device_fingerprint_parsed']
    before_dedup = len(df)
    
    aggregated = []
    for keys, group in df.groupby(dup_cols, dropna=False):
        device, build_type, fp, sdp = keys
        first = group.iloc[0]
        
        all_base = []
        all_var = []
        for _, row in group.iterrows():
            all_base.extend(row['priv_base_apps_list'])
            all_var.extend(row['priv_var_apps_list'])
        
        aggregated.append({
            'fingerprint': first['fingerprint'],
            'same_device_fingerprints': first['same_device_fingerprints'],
            'fingerprint_parsed': fp,
            'same_device_fingerprint_parsed': sdp,
            'priv_base_apps': first['priv_base_apps'],
            'priv_var_apps': first['priv_var_apps'],
            'priv_base_apps_list': list(set(all_base)),
            'priv_var_apps_list': list(set(all_var)),
            'build_type': build_type,
            'device': device
        })
    
    df = pd.DataFrame(aggregated)
    merged = before_dedup - len(df)
    if merged > 0:
        print(f"  Merged {merged} duplicate rows")
    
    print(f"  Final: {len(df)} rows (from {original_count})")
    return df


# ============================================================================
# STATISTICS GENERATION
# ============================================================================

def compute_statistics(df: pd.DataFrame, label: str = "Overall") -> Dict:
    """Compute frequentist statistics for apps."""
    all_base = []
    all_var = []
    combinations = []
    
    for _, row in df.iterrows():
        base_apps = row['priv_base_apps_list'] if isinstance(row['priv_base_apps_list'], list) else []
        var_apps = row['priv_var_apps_list'] if isinstance(row['priv_var_apps_list'], list) else []
        
        all_base.extend(base_apps)
        all_var.extend(var_apps)
        
        fp = row['fingerprint_parsed']
        sdp = row['same_device_fingerprint_parsed']
        combo_key = f"{fp} vs {sdp}"
        combinations.append(combo_key)
    
    base_counter = Counter(all_base)
    var_counter = Counter(all_var)
    combo_counter = Counter(combinations)
    
    return {
        'label': label,
        'total_rows': len(df),
        'total_base_apps': len(all_base),
        'unique_base_apps': len(base_counter),
        'total_var_apps': len(all_var),
        'unique_var_apps': len(var_counter),
        'top_10_base': base_counter.most_common(10),
        'top_10_var': var_counter.most_common(10),
        'top_10_combos': combo_counter.most_common(10),
        'base_counter': base_counter,
        'var_counter': var_counter,
        'avg_base_per_row': len(all_base) / len(df) if len(df) > 0 else 0,
        'avg_var_per_row': len(all_var) / len(df) if len(df) > 0 else 0,
    }


def create_stats_chart(stats: Dict, output_path: str) -> str:
    """Create a statistics chart and return base64 encoded image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Statistics: {stats['label']}", fontsize=14, fontweight='bold')
    
    # Chart 1: Top 10 Base Apps
    if stats['top_10_base']:
        apps, counts = zip(*stats['top_10_base'][:10])
        apps = [a.split('.')[-1][:15] for a in apps]  # Shorten names
        y_pos = np.arange(len(apps))
        axes[0].barh(y_pos, counts, color='#2E75B6')
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(apps, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Frequency')
        axes[0].set_title('Top 10 Base Apps', fontsize=10, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'No Data', ha='center', va='center')
        axes[0].set_title('Top 10 Base Apps')
    
    # Chart 2: Top 10 Variant Apps
    if stats['top_10_var']:
        apps, counts = zip(*stats['top_10_var'][:10])
        apps = [a.split('.')[-1][:15] for a in apps]
        y_pos = np.arange(len(apps))
        axes[1].barh(y_pos, counts, color='#5B9BD5')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(apps, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Frequency')
        axes[1].set_title('Top 10 Variant Apps', fontsize=10, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No Data', ha='center', va='center')
        axes[1].set_title('Top 10 Variant Apps')
    
    # Chart 3: Top 10 Combinations
    if stats['top_10_combos']:
        combos, counts = zip(*stats['top_10_combos'][:10])
        y_pos = np.arange(len(combos))
        axes[2].barh(y_pos, counts, color='#70AD47')
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(combos, fontsize=8)
        axes[2].invert_yaxis()
        axes[2].set_xlabel('Frequency')
        axes[2].set_title('Top 10 FP vs SDP Combos', fontsize=10, fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, 'No Data', ha='center', va='center')
        axes[2].set_title('Top 10 Combinations')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', format='png')
    plt.close()
    
    with open(output_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_all_statistics(df: pd.DataFrame) -> Tuple[Dict, Dict[str, Dict], List[str]]:
    """Generate overall and per-device statistics with charts."""
    charts = []
    
    # Overall statistics
    overall_stats = compute_statistics(df, "Overall")
    overall_chart = create_stats_chart(overall_stats, '/tmp/overall_chart.png')
    charts.append(('Overall', overall_chart))
    
    # Per-device statistics
    device_stats = {}
    for device in df['device'].unique():
        device_df = df[df['device'] == device]
        stats = compute_statistics(device_df, f"Device: {device}")
        device_stats[device] = stats
        chart = create_stats_chart(stats, f'/tmp/{device}_chart.png')
        charts.append((device, chart))
    
    return overall_stats, device_stats, charts


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_fingerprint_data(df: pd.DataFrame) -> Dict:
    """Process data into hierarchical structure."""
    print("\nPREPROCESSING...")
    df = preprocess_dataframe(df)
    
    # Generate statistics
    print("\nGENERATING STATISTICS...")
    overall_stats, device_stats, charts = generate_all_statistics(df)
    
    results = {
        '_statistics': {
            'overall': overall_stats,
            'per_device': device_stats,
            'charts': charts
        }
    }
    
    print(f"\nPROCESSING COMBINATIONS...")
    for device in df['device'].unique():
        results[device] = {}
        device_df = df[df['device'] == device]
        
        for build_type in device_df['build_type'].unique():
            build_df = device_df[device_df['build_type'] == build_type]
            
            fps = list(build_df['fingerprint_parsed'].unique())
            sdps = list(build_df['same_device_fingerprint_parsed'].unique())
            
            combinations = []
            seen = set()
            
            for _, row in build_df.iterrows():
                fp = row['fingerprint_parsed']
                sdp = row['same_device_fingerprint_parsed']
                key = (fp, sdp)
                
                if key in seen or fp == sdp:
                    continue
                seen.add(key)
                
                combinations.append({
                    'fingerprint_parsed': fp,
                    'same_device_fingerprint_parsed': sdp,
                    'priv_base_apps': row['priv_base_apps_list'],
                    'priv_var_apps': row['priv_var_apps_list']
                })
            
            results[device][build_type] = {
                'unique_fps': fps,
                'unique_sdps': sdps,
                'combinations': combinations
            }
            
            print(f"  {device}/{build_type}: {len(combinations)} combinations")
    
    return results


# ============================================================================
# SAMPLE DATA
# ============================================================================

def create_sample_data() -> pd.DataFrame:
    """Create sample data with test cases for duplicates and self-comparisons."""
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
            "google/sunfish/sunfish:11/RQ3A.210805",  # Self-comparison test
            "google/catfish/catfish:11/RQ3A.999999",  # Duplicate test
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
            "google/sunfish/sunfish:12/SQ1D.220105",  # ISH vs ISH
            "google/oriole/oriole:12/SQ1D.220105",    # ISH vs OLE duplicate
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
            "['com.google.files']",
            "['com.google.selftest']",
            "['com.google.duplicate.app']",
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
            "['com.android.email']",
            "['com.android.selftest']",
            "['com.android.duplicate.var']",
        ],
        'build_type': ['userdebug'] * 7 + ['user'] * 4 + ['userdebug'] * 2,
        'device': ['Pixel_4a'] * 13
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
        'build_type': ['eng'] * 4,
        'device': ['Galaxy_S10'] * 4
    }
    
    return pd.concat([pd.DataFrame(data), pd.DataFrame(data2)], ignore_index=True)


# ============================================================================
# JAVASCRIPT GENERATOR
# ============================================================================

def get_js_code() -> str:
    return '''
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, Header, Footer,
        AlignmentType, BorderStyle, WidthType, ShadingType, VerticalAlign, PageNumber,
        PageBreak, ImageRun } = require('docx');
const fs = require('fs');

const data = JSON.parse(fs.readFileSync('report_data.json', 'utf8'));
const colors = { primary: "1F4E79", secondary: "2E75B6", accent: "5B9BD5", 
                 headerBg: "D6E3F0", border: "B4C6E7", alt: "F2F7FB", text: "333333" };

const border = { style: BorderStyle.SINGLE, size: 8, color: colors.border };
const cellBorders = { top: border, bottom: border, left: border, right: border };

function headerCell(text, width) {
    return new TableCell({
        borders: cellBorders, width: { size: width, type: WidthType.DXA },
        shading: { fill: colors.headerBg, type: ShadingType.CLEAR },
        children: [new Paragraph({ alignment: AlignmentType.CENTER, 
            children: [new TextRun({ text, bold: true, size: 22, color: colors.primary, font: "Arial" })] })]
    });
}

function dataCell(content, width, alt = false) {
    const children = Array.isArray(content) && content.length > 0
        ? content.map(item => new Paragraph({ spacing: { before: 40, after: 40 },
            children: [new TextRun({ text: "• " + item.split('.').pop(), size: 20, color: colors.text, font: "Arial" })] }))
        : [new Paragraph({ alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: "—", size: 20, color: "999999", font: "Arial" })] })];
    return new TableCell({ borders: cellBorders, width: { size: width, type: WidthType.DXA },
        shading: alt ? { fill: colors.alt, type: ShadingType.CLEAR } : undefined, children });
}

function createStatsTable(stats) {
    const rows = [
        ["Total Combinations", stats.total_rows],
        ["Unique Base Apps", stats.unique_base_apps],
        ["Unique Variant Apps", stats.unique_var_apps],
        ["Avg Base Apps/Row", stats.avg_base_per_row.toFixed(2)],
        ["Avg Var Apps/Row", stats.avg_var_per_row.toFixed(2)]
    ];
    return new Table({ columnWidths: [4680, 4680], rows: [
        new TableRow({ children: [headerCell("Metric", 4680), headerCell("Value", 4680)] }),
        ...rows.map((r, i) => new TableRow({ children: [
            dataCell([r[0]], 4680, i % 2 === 1), dataCell([String(r[1])], 4680, i % 2 === 1)
        ]}))
    ]});
}

function createTop10Table(title, items) {
    if (!items || items.length === 0) return new Paragraph({ children: [new TextRun({ text: "No data", italics: true })] });
    return new Table({ columnWidths: [1000, 5680, 2680], rows: [
        new TableRow({ children: [headerCell("#", 1000), headerCell(title, 5680), headerCell("Count", 2680)] }),
        ...items.slice(0, 10).map((item, i) => new TableRow({ children: [
            dataCell([String(i + 1)], 1000, i % 2 === 1),
            dataCell([item[0].split('.').pop()], 5680, i % 2 === 1),
            dataCell([String(item[1])], 2680, i % 2 === 1)
        ]}))
    ]});
}

function createAppsTable(base, variant) {
    return new Table({ columnWidths: [4680, 4680], rows: [
        new TableRow({ children: [headerCell("Priv Base Apps", 4680), headerCell("Priv Variant Apps", 4680)] }),
        new TableRow({ children: [dataCell(base, 4680), dataCell(variant, 4680)] })
    ]});
}

function divider() {
    return new Paragraph({ spacing: { before: 200, after: 200 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: colors.border } },
        children: [new TextRun({ text: "" })] });
}

function sectionTitle(text, level = 1) {
    const sizes = { 1: 36, 2: 28, 3: 24 };
    return new Paragraph({ spacing: { before: 300, after: 150 },
        children: [new TextRun({ text, bold: true, size: sizes[level], color: colors.primary, font: "Arial" })] });
}

function generateContent() {
    const content = [];
    const stats = data._statistics;
    
    // Title
    content.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400, after: 200 },
        children: [new TextRun({ text: "Fingerprint Grouping Report", bold: true, size: 48, color: colors.primary, font: "Arial" })] }));
    content.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
        children: [new TextRun({ text: new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }), size: 22, color: "666666", font: "Arial" })] }));
    content.push(divider());
    
    // Overall Statistics Section
    content.push(sectionTitle("📊 Overall Statistics", 1));
    content.push(createStatsTable(stats.overall));
    content.push(new Paragraph({ spacing: { before: 100, after: 100 }, children: [] }));
    
    // Overall Chart
    const overallChart = stats.charts.find(c => c[0] === 'Overall');
    if (overallChart) {
        content.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [
            new ImageRun({ type: "png", data: Buffer.from(overallChart[1], 'base64'), 
                transformation: { width: 600, height: 200 } })
        ]}));
    }
    
    // Top 10 Tables
    content.push(sectionTitle("Top 10 Base Apps", 2));
    content.push(createTop10Table("App Name", stats.overall.top_10_base));
    content.push(sectionTitle("Top 10 Variant Apps", 2));
    content.push(createTop10Table("App Name", stats.overall.top_10_var));
    content.push(sectionTitle("Top 10 FP vs SDP Combinations", 2));
    content.push(createTop10Table("Combination", stats.overall.top_10_combos));
    
    // Per-Device Statistics
    content.push(new Paragraph({ children: [new PageBreak()] }));
    content.push(sectionTitle("📱 Per-Device Statistics", 1));
    
    for (const [device, deviceStats] of Object.entries(stats.per_device)) {
        content.push(sectionTitle("Device: " + device, 2));
        content.push(createStatsTable(deviceStats));
        
        const deviceChart = stats.charts.find(c => c[0] === device);
        if (deviceChart) {
            content.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100 }, children: [
                new ImageRun({ type: "png", data: Buffer.from(deviceChart[1], 'base64'),
                    transformation: { width: 550, height: 180 } })
            ]}));
        }
        content.push(divider());
    }
    
    // Device Details
    content.push(new Paragraph({ children: [new PageBreak()] }));
    content.push(sectionTitle("📋 Detailed Combinations", 1));
    
    for (const [device, buildTypes] of Object.entries(data)) {
        if (device === '_statistics') continue;
        
        content.push(sectionTitle("Device: " + device, 2));
        
        for (const [buildType, btData] of Object.entries(buildTypes)) {
            content.push(new Paragraph({ spacing: { before: 200, after: 100 },
                children: [
                    new TextRun({ text: "Build Type: ", bold: true, size: 24, color: colors.secondary, font: "Arial" }),
                    new TextRun({ text: buildType, size: 24, color: colors.text, font: "Arial" }),
                    new TextRun({ text: "  (" + btData.combinations.length + " combinations)", size: 20, color: "888888", font: "Arial" })
                ]}));
            
            for (const combo of btData.combinations) {
                content.push(new Paragraph({ spacing: { before: 150, after: 80 },
                    children: [
                        new TextRun({ text: "FP: ", bold: true, size: 22, color: colors.secondary, font: "Arial" }),
                        new TextRun({ text: combo.fingerprint_parsed, size: 22, color: colors.text, font: "Arial" }),
                        new TextRun({ text: "  |  SDP: ", bold: true, size: 22, color: colors.secondary, font: "Arial" }),
                        new TextRun({ text: combo.same_device_fingerprint_parsed, size: 22, color: colors.text, font: "Arial" })
                    ]}));
                content.push(createAppsTable(combo.priv_base_apps || [], combo.priv_var_apps || []));
            }
            content.push(divider());
        }
    }
    
    return content;
}

const doc = new Document({
    styles: { default: { document: { run: { font: "Arial", size: 22 } } } },
    sections: [{
        properties: { page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
        headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
            children: [new TextRun({ text: "Fingerprint Grouping Report", size: 18, color: "999999", font: "Arial" })] })] }) },
        footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
            children: [
                new TextRun({ text: "Page ", size: 18, color: "999999", font: "Arial" }),
                new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "999999", font: "Arial" }),
                new TextRun({ text: " of ", size: 18, color: "999999", font: "Arial" }),
                new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: "999999", font: "Arial" })
            ]})] }) },
        children: generateContent()
    }]
});

Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync(process.argv[2] || 'report.docx', buffer);
    console.log('Generated: ' + (process.argv[2] || 'report.docx'));
});
'''


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Fingerprint Grouping Report')
    parser.add_argument('--input', '-i', help='Input CSV file')
    parser.add_argument('--output', '-o', default='report.docx', help='Output Word document')
    parser.add_argument('--sample', '-s', action='store_true', help='Use sample data')
    args = parser.parse_args()
    
    print("=" * 60)
    print("FINGERPRINT GROUPING REPORT GENERATOR")
    print("=" * 60)
    
    # Load data
    if args.sample:
        print("\nUsing sample data...")
        df = create_sample_data()
    elif args.input:
        print(f"\nLoading {args.input}...")
        df = pd.read_csv(args.input)
    else:
        print("\nNo input specified, using sample data...")
        df = create_sample_data()
    
    required = ['fingerprint', 'same_device_fingerprints', 'priv_base_apps', 'priv_var_apps', 'build_type', 'device']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return 1
    
    print(f"Loaded {len(df)} rows")
    
    # Process
    results = process_fingerprint_data(df)
    
    # Export
    print("\nEXPORTING...")
    
    # Convert stats for JSON (remove Counter objects)
    export_data = {}
    for key, value in results.items():
        if key == '_statistics':
            export_stats = {
                'overall': {k: v for k, v in value['overall'].items() if k not in ['base_counter', 'var_counter']},
                'per_device': {d: {k: v for k, v in s.items() if k not in ['base_counter', 'var_counter']} 
                              for d, s in value['per_device'].items()},
                'charts': value['charts']
            }
            export_data['_statistics'] = export_stats
        else:
            export_data[key] = value
    
    with open('report_data.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    
    with open('generate_report.js', 'w') as f:
        f.write(get_js_code())
    
    # Generate document
    print("\nGENERATING DOCUMENT...")
    try:
        result = subprocess.run(['node', 'generate_report.js', args.output], 
                               capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"\n✓ Report generated: {args.output}")
        else:
            print(f"ERROR: {result.stderr}")
            return 1
    except FileNotFoundError:
        print("ERROR: Node.js not found")
        return 1
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    stats = results['_statistics']['overall']
    print(f"Total combinations: {stats['total_rows']}")
    print(f"Unique base apps: {stats['unique_base_apps']}")
    print(f"Unique variant apps: {stats['unique_var_apps']}")
    print(f"Output: {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())
