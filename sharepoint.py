"""
Fingerprint Grouping Report Generator (Non-LLM Version)

This script processes a CSV file containing device fingerprint data,
groups and filters by fingerprint combinations, and generates a 
professional Word document report.

Usage:
    python report.py --input your_data.csv --output report.docx
    
    Or with sample data for testing:
    python report.py --sample --output report.docx

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
import tempfile
from collections import Counter
from typing import Dict, List, Any


# ============================================================================
# FINGERPRINT PARSING
# ============================================================================

def parse_fingerprint(fingerprint_value: str) -> str:
    """
    Parse fingerprint to extract the word between first and second "/",
    take last 3 characters, and capitalize them.
    
    Example: "google/sunfish/sunfish:11/RQ3A.210805.001.A1" -> "ISH"
             (sunfish -> ish -> ISH)
    
    If fingerprint doesn't match expected format, return original value.
    """
    if not isinstance(fingerprint_value, str):
        return str(fingerprint_value)
    
    parts = fingerprint_value.split('/')
    
    if len(parts) >= 2:
        # Get the word between first and second "/"
        word = parts[1]
        # Take last 3 characters and capitalize
        if len(word) >= 3:
            return word[-3:].upper()
        else:
            return word.upper()
    
    # If format doesn't match, return original
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
            # Try to parse as JSON list
            try:
                parsed = json.loads(item.replace("'", '"'))
                if isinstance(parsed, list):
                    flattened.extend(parsed)
                else:
                    flattened.append(str(parsed))
            except (json.JSONDecodeError, ValueError):
                # Just use the string as-is
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
    
    Algorithm:
    1. For each unique device
    2.   For each unique build_type in that device
    3.     Get unique fingerprints and same_device_fingerprints
    4.       For each fingerprint
    5.         For each same_device_fingerprint
    6.           Filter and collect priv_base_apps and priv_var_apps
    
    Returns:
        Dict with hierarchical structure containing all processed data
    """
    results = {}
    
    # Step 1: Get all unique devices
    unique_devices = get_unique_values(df, 'device')
    print(f"Found {len(unique_devices)} unique devices: {unique_devices}")
    
    for device in unique_devices:
        results[device] = {}
        
        # Step 2: Filter by device
        device_data = df[df['device'] == device]
        
        # Step 2.1: Get unique build types for this device
        unique_build_types = get_unique_values(device_data, 'build_type')
        print(f"  Device {device}: Found {len(unique_build_types)} build types: {unique_build_types}")
        
        for build_type in unique_build_types:
            # Step 3: Filter by build type
            build_data = device_data[device_data['build_type'] == build_type]
            
            # Step 3.1: Get unique fingerprints and same_device_fingerprints
            unique_fingerprints = get_unique_values(build_data, 'fingerprint')
            unique_same_device_fps = get_unique_values(build_data, 'same_device_fingerprints')
            
            # Parse fingerprints for display
            unique_fingerprints_parsed = [parse_fingerprint(fp) for fp in unique_fingerprints]
            unique_same_device_fps_parsed = [parse_fingerprint(sdp) for sdp in unique_same_device_fps]
            
            print(f"    Build Type {build_type}:")
            print(f"      Unique fingerprints: {unique_fingerprints_parsed}")
            print(f"      Unique same_device_fps: {unique_same_device_fps_parsed}")
            
            results[device][build_type] = {
                'unique_fingerprints': unique_fingerprints,
                'unique_fingerprints_parsed': unique_fingerprints_parsed,
                'unique_same_device_fps': unique_same_device_fps,
                'unique_same_device_fps_parsed': unique_same_device_fps_parsed,
                'combinations': []
            }
            
            # Step 4: For each fingerprint
            for fp_idx, fp in enumerate(unique_fingerprints):
                fp_data = build_data[build_data['fingerprint'] == fp]
                fp_parsed = unique_fingerprints_parsed[fp_idx]
                
                # Step 5: For each same_device_fingerprint
                for sdp_idx, sdp in enumerate(unique_same_device_fps):
                    # Step 6: Filter and collect
                    filtered_rows = fp_data[fp_data['same_device_fingerprints'] == sdp]
                    sdp_parsed = unique_same_device_fps_parsed[sdp_idx]
                    
                    if not filtered_rows.empty:
                        # Collect priv_base_apps and priv_var_apps
                        priv_base_apps = filtered_rows['priv_base_apps'].dropna().tolist()
                        priv_var_apps = filtered_rows['priv_var_apps'].dropna().tolist()
                        
                        # Flatten if they contain lists stored as strings
                        priv_base_apps = flatten_apps(priv_base_apps)
                        priv_var_apps = flatten_apps(priv_var_apps)
                        
                        combination = {
                            'fingerprint': fp,
                            'fingerprint_parsed': fp_parsed,
                            'same_device_fingerprint': sdp,
                            'same_device_fingerprint_parsed': sdp_parsed,
                            'priv_base_apps': priv_base_apps,
                            'priv_var_apps': priv_var_apps
                        }
                        
                        results[device][build_type]['combinations'].append(combination)
                        print(f"        FP={fp_parsed}, SDP={sdp_parsed}: {len(priv_base_apps)} base, {len(priv_var_apps)} var apps")
    
    return results


def generate_report_sections(results: Dict) -> List[Dict]:
    """
    Transform results into a flat structure suitable for report generation.
    """
    report_sections = []
    
    for device, build_types in results.items():
        for build_type, data in build_types.items():
            section = {
                'device': device,
                'build_type': build_type,
                'unique_fingerprints': data['unique_fingerprints'],
                'unique_fingerprints_parsed': data['unique_fingerprints_parsed'],
                'unique_same_device_fps': data['unique_same_device_fps'],
                'unique_same_device_fps_parsed': data['unique_same_device_fps_parsed'],
                'combinations': data['combinations']
            }
            report_sections.append(section)
    
    return report_sections


# ============================================================================
# SAMPLE DATA (for testing)
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
    
    # Add another device
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
        'device': ['Galaxy_S10', 'Galaxy_S10', 'Galaxy_S10', 'Galaxy_S10']
    }
    
    df1 = pd.DataFrame(data)
    df2 = pd.DataFrame(data2)
    
    return pd.concat([df1, df2], ignore_index=True)


# ============================================================================
# JAVASCRIPT DOCUMENT GENERATOR
# ============================================================================

def get_js_generator_code() -> str:
    """
    Returns the JavaScript code for generating the Word document.
    This uses docx-js library for professional formatting.
    """
    return '''
const {
    Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
    Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType, 
    ShadingType, VerticalAlign, PageNumber, PageBreak
} = require('docx');
const fs = require('fs');

// Load data from JSON
const reportData = JSON.parse(fs.readFileSync('report_data.json', 'utf8'));

// Color scheme
const colors = {
    primary: "1F4E79",
    secondary: "2E75B6",
    accent: "5B9BD5",
    headerBg: "D6E3F0",
    tableBorder: "B4C6E7",
    altRow: "F2F7FB",
    text: "333333",
    lightText: "666666"
};

// Table styling
const tableBorder = { style: BorderStyle.SINGLE, size: 8, color: colors.tableBorder };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };
const headerBorders = { 
    top: { style: BorderStyle.SINGLE, size: 12, color: colors.primary }, 
    bottom: { style: BorderStyle.SINGLE, size: 12, color: colors.primary }, 
    left: tableBorder, 
    right: tableBorder 
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
                children: [
                    new TextRun({
                        text: text,
                        bold: true,
                        size: 24,
                        color: colors.primary,
                        font: "Arial"
                    })
                ]
            })
        ]
    });
}

function createDataCell(content, width, isAltRow = false) {
    const children = [];
    
    if (Array.isArray(content) && content.length > 0) {
        content.forEach((item) => {
            children.push(
                new Paragraph({
                    spacing: { before: 60, after: 60 },
                    indent: { left: 200 },
                    children: [
                        new TextRun({
                            text: "• " + item,
                            size: 22,
                            color: colors.text,
                            font: "Arial"
                        })
                    ]
                })
            );
        });
    } else {
        children.push(
            new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 80, after: 80 },
                children: [
                    new TextRun({
                        text: "—",
                        size: 22,
                        color: "999999",
                        font: "Arial"
                    })
                ]
            })
        );
    }
    
    return new TableCell({
        borders: cellBorders,
        width: { size: width, type: WidthType.DXA },
        shading: isAltRow ? { fill: colors.altRow, type: ShadingType.CLEAR } : undefined,
        verticalAlign: VerticalAlign.CENTER,
        children: children
    });
}

function createAppsTable(privBaseApps, privVarApps) {
    const colWidths = [4680, 4680];
    
    return new Table({
        columnWidths: colWidths,
        rows: [
            new TableRow({
                tableHeader: true,
                children: [
                    createHeaderCell("Priv Base Apps", colWidths[0]),
                    createHeaderCell("Priv Variant Apps", colWidths[1])
                ]
            }),
            new TableRow({
                children: [
                    createDataCell(privBaseApps, colWidths[0]),
                    createDataCell(privVarApps, colWidths[1])
                ]
            })
        ]
    });
}

function createDivider() {
    return new Paragraph({
        spacing: { before: 300, after: 300 },
        border: {
            bottom: { style: BorderStyle.SINGLE, size: 6, color: colors.tableBorder }
        },
        children: [new TextRun({ text: "" })]
    });
}

function generateContent() {
    const sections = reportData.report_sections || [];
    let allContent = [];
    
    // Title
    allContent.push(
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 800, after: 300 },
            children: [
                new TextRun({
                    text: "Fingerprint Grouping Report",
                    bold: true,
                    size: 52,
                    color: colors.primary,
                    font: "Arial"
                })
            ]
        })
    );
    
    allContent.push(
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 100, after: 600 },
            children: [
                new TextRun({
                    text: "Generated: " + new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }),
                    size: 22,
                    color: colors.lightText,
                    font: "Arial"
                })
            ]
        })
    );
    
    // Thick divider
    allContent.push(
        new Paragraph({
            spacing: { before: 100, after: 400 },
            border: {
                bottom: { style: BorderStyle.SINGLE, size: 18, color: colors.primary }
            },
            children: [new TextRun({ text: "" })]
        })
    );
    
    let isFirstCombination = true;
    let lastDeviceBuild = null;
    
    sections.forEach((section) => {
        const device = section.device;
        const buildType = section.build_type;
        const combinations = section.combinations || [];
        const currentDeviceBuild = device + "|" + buildType;
        
        combinations.forEach((combo) => {
            const fp = combo.fingerprint_parsed || combo.fingerprint || 'Unknown';
            const sdp = combo.same_device_fingerprint_parsed || combo.same_device_fingerprint || 'Unknown';
            const privBaseApps = combo.priv_base_apps || [];
            const privVarApps = combo.priv_var_apps || [];
            
            if (!isFirstCombination) {
                allContent.push(createDivider());
            }
            isFirstCombination = false;
            
            // Device
            allContent.push(
                new Paragraph({
                    spacing: { before: 200, after: 100 },
                    children: [
                        new TextRun({
                            text: "Device: ",
                            bold: true,
                            size: 28,
                            color: colors.primary,
                            font: "Arial"
                        }),
                        new TextRun({
                            text: String(device),
                            size: 28,
                            color: colors.text,
                            font: "Arial"
                        })
                    ]
                })
            );
            
            // Build Type
            allContent.push(
                new Paragraph({
                    spacing: { before: 80, after: 150 },
                    children: [
                        new TextRun({
                            text: "Build Type: ",
                            bold: true,
                            size: 28,
                            color: colors.primary,
                            font: "Arial"
                        }),
                        new TextRun({
                            text: String(buildType),
                            size: 28,
                            color: colors.text,
                            font: "Arial"
                        })
                    ]
                })
            );
            
            // Fingerprint and Same Device Fingerprint
            allContent.push(
                new Paragraph({
                    spacing: { before: 150, after: 200 },
                    children: [
                        new TextRun({
                            text: "Fingerprint: ",
                            bold: true,
                            size: 24,
                            color: colors.secondary,
                            font: "Arial"
                        }),
                        new TextRun({
                            text: String(fp),
                            size: 24,
                            color: colors.text,
                            font: "Arial"
                        }),
                        new TextRun({
                            text: "   |   ",
                            size: 24,
                            color: colors.tableBorder,
                            font: "Arial"
                        }),
                        new TextRun({
                            text: "Same Device Fingerprint: ",
                            bold: true,
                            size: 24,
                            color: colors.secondary,
                            font: "Arial"
                        }),
                        new TextRun({
                            text: String(sdp),
                            size: 24,
                            color: colors.text,
                            font: "Arial"
                        })
                    ]
                })
            );
            
            // Spacing
            allContent.push(
                new Paragraph({
                    spacing: { before: 100, after: 100 },
                    children: [new TextRun({ text: "" })]
                })
            );
            
            // Apps Table
            allContent.push(createAppsTable(privBaseApps, privVarApps));
        });
    });
    
    return allContent;
}

function generateDocument() {
    const content = generateContent();
    
    const doc = new Document({
        styles: {
            default: {
                document: {
                    run: { font: "Arial", size: 22 }
                }
            }
        },
        sections: [{
            properties: {
                page: {
                    margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
                    pageNumbers: { start: 1 }
                }
            },
            headers: {
                default: new Header({
                    children: [
                        new Paragraph({
                            alignment: AlignmentType.RIGHT,
                            children: [
                                new TextRun({
                                    text: "Fingerprint Grouping Report",
                                    size: 18,
                                    color: "999999",
                                    font: "Arial"
                                })
                            ]
                        })
                    ]
                })
            },
            footers: {
                default: new Footer({
                    children: [
                        new Paragraph({
                            alignment: AlignmentType.CENTER,
                            children: [
                                new TextRun({ text: "Page ", size: 18, color: "999999", font: "Arial" }),
                                new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "999999", font: "Arial" }),
                                new TextRun({ text: " of ", size: 18, color: "999999", font: "Arial" }),
                                new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: "999999", font: "Arial" })
                            ]
                        })
                    ]
                })
            },
            children: content
        }]
    });
    
    const outputFile = process.argv[2] || 'report.docx';
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
        description='Generate Fingerprint Grouping Report from CSV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python report.py --input data.csv --output report.docx
    python report.py --sample --output test_report.docx
    
Required CSV columns:
    fingerprint, same_device_fingerprints, priv_base_apps, 
    priv_var_apps, build_type, device
        """
    )
    parser.add_argument('--input', '-i', help='Input CSV file path')
    parser.add_argument('--output', '-o', default='report.docx', help='Output Word document path')
    parser.add_argument('--sample', '-s', action='store_true', help='Use sample data for testing')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FINGERPRINT GROUPING REPORT GENERATOR")
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
        print("\nNo input specified. Use --sample for demo or --input for your CSV file.")
        print("Using sample data for demonstration...")
        df = create_sample_data()
    
    # Validate columns
    required_columns = ['fingerprint', 'same_device_fingerprints', 'priv_base_apps', 
                        'priv_var_apps', 'build_type', 'device']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return 1
    
    print(f"\nLoaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Process data
    print("\n" + "-" * 60)
    print("PROCESSING DATA...")
    print("-" * 60)
    
    results = process_fingerprint_data(df)
    report_sections = generate_report_sections(results)
    
    # Prepare data for JS
    report_data = {
        'results': results,
        'report_sections': report_sections
    }
    
    # Write JSON for JS consumption
    json_path = 'report_data.json'
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"\nData exported to {json_path}")
    
    # Write JS file
    js_path = 'generate_report.js'
    with open(js_path, 'w') as f:
        f.write(get_js_generator_code())
    print(f"JS generator written to {js_path}")
    
    # Run JS to generate Word document
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
            print(f"ERROR running Node.js: {result.stderr}")
            return 1
            
    except FileNotFoundError:
        print("ERROR: Node.js not found. Please install Node.js to generate Word documents.")
        print("The JSON data has been saved to report_data.json")
        return 1
    except subprocess.TimeoutExpired:
        print("ERROR: Document generation timed out")
        return 1
    
    # Cleanup temp files (optional - comment out to keep them)
    # os.remove(json_path)
    # os.remove(js_path)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total devices: {len(results)}")
    print(f"Total sections: {len(report_sections)}")
    total_combos = sum(len(s['combinations']) for s in report_sections)
    print(f"Total combinations: {total_combos}")
    print(f"Output file: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
