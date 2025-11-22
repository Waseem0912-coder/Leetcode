import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from docx import Document
from docx.shared import Inches
from collections import Counter
import json
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def filter_out_columns(df):
    """Filter out rows based on release_type and vr_qualified conditions"""
    col_to_filter = ['release_type', 'vr_qualified']
    
    # Remove rows where release_type == "SMR" or vr_qualified == "TRUE"
    initial_len = len(df)
    df = df[~((df['release_type'] == 'SMR') | (df['vr_qualified'] == 'TRUE'))]
    print(f"Filtered out {initial_len - len(df)} rows based on release_type='SMR' or vr_qualified='TRUE'")
    
    # Handle priv_app columns - replace "[]" with empty/null
    cols_to_clean = ["priv_app_in_base_only", "priv_app_in_variant_only"]
    for col in cols_to_clean:
        if col in df.columns:
            # Replace "[]" string with empty string
            df[col] = df[col].replace('[]', '')
            df[col] = df[col].replace(['[]', '[ ]'], '', regex=False)
            # Also handle NaN values
            df[col] = df[col].fillna('')
    
    return df

def fingerprint_truncation(df):
    """Extract items between first and second "/" in fingerprint columns"""
    cols = ["fingerprint", "same_device_fingerprint"]
    
    for col in cols:
        if col in df.columns:
            def extract_between_slashes(text):
                if pd.isna(text) or text == '':
                    return text
                # Convert to string if not already
                text = str(text)
                # Split by "/" and get the second element (between first and second slash)
                parts = text.split('/')
                if len(parts) >= 2:
                    return parts[1]
                return text
            
            df[f'{col}_extracted'] = df[col].apply(extract_between_slashes)
            print(f"Extracted fingerprint codes for column: {col}")
    
    return df

def parse_app_list(text):
    """Parse the app list string into a Python list"""
    if pd.isna(text) or text == '' or text == '[]':
        return []
    
    try:
        # Try to parse as JSON first
        if isinstance(text, str):
            # Clean up the string
            text = text.strip()
            if text.startswith('[') and text.endswith(']'):
                # Replace single quotes with double quotes for JSON parsing
                text_json = text.replace("'", '"')
                try:
                    return json.loads(text_json)
                except:
                    # If JSON parsing fails, try eval (be careful with this in production)
                    return eval(text)
            else:
                # If not in array format, split by comma
                return [item.strip() for item in text.split(',') if item.strip()]
        return []
    except:
        return []

def get_top_items_and_combinations(app_lists, top_n=5):
    """Get top individual items and combinations from a list of app lists"""
    # Count individual items
    individual_counter = Counter()
    # Count combinations (pairs)
    combination_counter = Counter()
    
    for apps in app_lists:
        if apps:  # Only process non-empty lists
            # Count individual items
            for app in apps:
                individual_counter[app] += 1
            
            # Count combinations (pairs) if there are at least 2 items
            if len(apps) >= 2:
                # Get all 2-item combinations
                for combo in combinations(sorted(apps), 2):
                    combination_counter[combo] += 1
    
    # Get top items
    top_individual = individual_counter.most_common(top_n)
    top_combinations = combination_counter.most_common(top_n)
    
    return top_individual, top_combinations

def stats_creation(df):
    """Create comprehensive statistics for the data"""
    stats_results = {}
    plots = []
    
    # Ensure we have the extracted fingerprint columns
    if 'fingerprint_extracted' not in df.columns:
        df = fingerprint_truncation(df)
    
    # Parse the app columns
    for col in ['priv_app_in_base_only', 'priv_app_in_variant_only']:
        if col in df.columns:
            df[f'{col}_parsed'] = df[col].apply(parse_app_list)
    
    # Get unique devices and release types
    devices = df['device'].unique() if 'device' in df.columns else []
    release_types = df['release_type'].unique() if 'release_type' in df.columns else []
    
    print(f"Found {len(devices)} unique devices and {len(release_types)} release types")
    
    # Analyze for each device and release type combination
    for device in devices[:10]:  # Limit to first 10 devices for performance
        stats_results[device] = {}
        
        for release_type in release_types:
            # Filter data for this device and release type
            mask = (df['device'] == device) & (df['release_type'] == release_type)
            subset = df[mask]
            
            if len(subset) == 0:
                continue
            
            stats_results[device][release_type] = {
                'count': len(subset),
                'fingerprints': {},
                'same_device_fingerprints': {}
            }
            
            # Analyze by fingerprint groups
            for fp_col, fp_name in [('fingerprint_extracted', 'fingerprints'), 
                                   ('same_device_fingerprint_extracted', 'same_device_fingerprints')]:
                if fp_col in subset.columns:
                    fp_groups = subset.groupby(fp_col)
                    
                    for fp_value, fp_data in fp_groups:
                        if pd.notna(fp_value):
                            stats_results[device][release_type][fp_name][fp_value] = {}
                            
                            # Analyze app columns
                            for app_col in ['priv_app_in_base_only', 'priv_app_in_variant_only']:
                                parsed_col = f'{app_col}_parsed'
                                if parsed_col in fp_data.columns:
                                    app_lists = fp_data[parsed_col].tolist()
                                    top_individual, top_combinations = get_top_items_and_combinations(app_lists)
                                    
                                    stats_results[device][release_type][fp_name][fp_value][app_col] = {
                                        'top_individual': top_individual,
                                        'top_combinations': top_combinations,
                                        'total_entries': len(app_lists),
                                        'non_empty_entries': sum(1 for apps in app_lists if apps)
                                    }
    
    # Create visualizations
    # 1. Device distribution
    device_counts = df['device'].value_counts().head(20)
    fig1 = px.bar(x=device_counts.index, y=device_counts.values,
                  title='Top 20 Devices by Frequency',
                  labels={'x': 'Device', 'y': 'Count'})
    plots.append(('device_distribution.html', fig1))
    
    # 2. Release type distribution
    if 'release_type' in df.columns:
        release_counts = df['release_type'].value_counts()
        fig2 = px.pie(values=release_counts.values, names=release_counts.index,
                      title='Release Type Distribution')
        plots.append(('release_type_distribution.html', fig2))
    
    # 3. App frequency analysis for top apps overall
    all_base_apps = []
    all_variant_apps = []
    
    if 'priv_app_in_base_only_parsed' in df.columns:
        for apps in df['priv_app_in_base_only_parsed']:
            all_base_apps.extend(apps)
    
    if 'priv_app_in_variant_only_parsed' in df.columns:
        for apps in df['priv_app_in_variant_only_parsed']:
            all_variant_apps.extend(apps)
    
    if all_base_apps:
        base_counter = Counter(all_base_apps)
        top_base = base_counter.most_common(15)
        if top_base:
            fig3 = px.bar(x=[item[1] for item in top_base], 
                         y=[item[0] for item in top_base],
                         orientation='h',
                         title='Top 15 Apps in Base Only',
                         labels={'x': 'Frequency', 'y': 'App'})
            plots.append(('top_base_apps.html', fig3))
    
    if all_variant_apps:
        variant_counter = Counter(all_variant_apps)
        top_variant = variant_counter.most_common(15)
        if top_variant:
            fig4 = px.bar(x=[item[1] for item in top_variant], 
                         y=[item[0] for item in top_variant],
                         orientation='h',
                         title='Top 15 Apps in Variant Only',
                         labels={'x': 'Frequency', 'y': 'App'})
            plots.append(('top_variant_apps.html', fig4))
    
    # Save plots
    for filename, fig in plots:
        fig.write_html(f'/home/claude/{filename}')
        print(f"Saved plot: {filename}")
    
    return stats_results, plots

def create_report(df, stats_results, plots):
    """Create a DOCX report with statistics and insights"""
    doc = Document()
    
    # Title
    doc.add_heading('Android Device Configuration Analysis Report', 0)
    
    # Summary Section
    doc.add_heading('Executive Summary', 1)
    doc.add_paragraph(f'Total records analyzed: {len(df):,}')
    doc.add_paragraph(f'Unique devices: {df["device"].nunique() if "device" in df.columns else 0}')
    doc.add_paragraph(f'Release types: {df["release_type"].unique().tolist() if "release_type" in df.columns else []}')
    
    # Data Overview
    doc.add_heading('Data Overview', 1)
    doc.add_paragraph('This report analyzes the application configurations across different Android devices, '
                     'release types, and fingerprint variations.')
    
    # Key Findings Section
    doc.add_heading('Key Findings', 1)
    
    # Find the most analyzed device
    if stats_results:
        first_device = list(stats_results.keys())[0]
        doc.add_heading(f'Analysis for Device: {first_device}', 2)
        
        for release_type, rt_data in stats_results[first_device].items():
            doc.add_heading(f'Release Type: {release_type}', 3)
            doc.add_paragraph(f'Total entries: {rt_data["count"]}')
            
            # Report on fingerprints
            for fp_type in ['fingerprints', 'same_device_fingerprints']:
                if rt_data[fp_type]:
                    doc.add_heading(f'{fp_type.replace("_", " ").title()}', 4)
                    
                    for fp_value, fp_data in list(rt_data[fp_type].items())[:3]:  # Top 3 fingerprints
                        doc.add_paragraph(f'\nFingerprint: {fp_value}', style='List Bullet')
                        
                        for app_col in ['priv_app_in_base_only', 'priv_app_in_variant_only']:
                            if app_col in fp_data:
                                doc.add_paragraph(f'{app_col.replace("_", " ").title()}:', style='List Bullet 2')
                                
                                # Top individual apps
                                if fp_data[app_col]['top_individual']:
                                    doc.add_paragraph('Top Individual Apps:', style='List Bullet 3')
                                    for app, count in fp_data[app_col]['top_individual']:
                                        doc.add_paragraph(f'{app}: {count} occurrences', style='List Bullet 3')
                                
                                # Top combinations
                                if fp_data[app_col]['top_combinations']:
                                    doc.add_paragraph('Top App Combinations:', style='List Bullet 3')
                                    for combo, count in fp_data[app_col]['top_combinations']:
                                        doc.add_paragraph(f'{combo[0]} + {combo[1]}: {count} occurrences', 
                                                        style='List Bullet 3')
    
    # Statistical Summary
    doc.add_heading('Statistical Summary', 1)
    
    # Overall app statistics
    all_base_apps = []
    all_variant_apps = []
    
    if 'priv_app_in_base_only_parsed' in df.columns:
        for apps in df['priv_app_in_base_only_parsed']:
            all_base_apps.extend(apps)
    
    if 'priv_app_in_variant_only_parsed' in df.columns:
        for apps in df['priv_app_in_variant_only_parsed']:
            all_variant_apps.extend(apps)
    
    if all_base_apps:
        doc.add_heading('Base Apps Statistics', 2)
        doc.add_paragraph(f'Total unique apps in base: {len(set(all_base_apps))}')
        doc.add_paragraph(f'Total app occurrences: {len(all_base_apps)}')
        
        base_counter = Counter(all_base_apps)
        doc.add_paragraph('\nTop 10 Most Frequent Base Apps:')
        for app, count in base_counter.most_common(10):
            doc.add_paragraph(f'• {app}: {count} occurrences', style='List Bullet')
    
    if all_variant_apps:
        doc.add_heading('Variant Apps Statistics', 2)
        doc.add_paragraph(f'Total unique apps in variant: {len(set(all_variant_apps))}')
        doc.add_paragraph(f'Total app occurrences: {len(all_variant_apps)}')
        
        variant_counter = Counter(all_variant_apps)
        doc.add_paragraph('\nTop 10 Most Frequent Variant Apps:')
        for app, count in variant_counter.most_common(10):
            doc.add_paragraph(f'• {app}: {count} occurrences', style='List Bullet')
    
    # Conclusions
    doc.add_heading('Conclusions and Recommendations', 1)
    doc.add_paragraph('Based on the analysis of device configurations and application distributions:')
    doc.add_paragraph('• There is significant variation in application configurations across different devices and release types.')
    doc.add_paragraph('• Certain applications appear consistently across multiple configurations, indicating core functionality.')
    doc.add_paragraph('• The fingerprint-based grouping reveals patterns in device variants and their associated applications.')
    
    # Save the document
    doc.save('/home/claude/analysis_report.docx')
    print("Report saved as analysis_report.docx")
    
    return doc

def main():
    """Main execution function"""
    print("Starting Android Device Configuration Analysis...")
    
    # Check if output.csv exists
    csv_path = "output.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found in current directory")
        return
    
    # Read CSV with chunking for large files
    print(f"Reading {csv_path}...")
    
    try:
        # First, get a sample to understand the structure
        sample_df = pd.read_csv(csv_path, nrows=10000)
        print(f"Sample loaded. Columns found: {sample_df.columns.tolist()}")
        
        # Process the full file in chunks if it's large
        chunk_size = 50000
        chunks = []
        
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            # Apply filtering to each chunk
            filtered_chunk = filter_out_columns(chunk)
            chunks.append(filtered_chunk)
            
            if len(chunks) * chunk_size >= 200000:  # Limit to 200k rows for analysis
                print("Reached analysis limit of 200k rows")
                break
        
        # Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(df)} rows after filtering")
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        # Try with a smaller sample
        print("Attempting to read a smaller sample...")
        df = pd.read_csv(csv_path, nrows=50000)
        df = filter_out_columns(df)
    
    # Apply fingerprint truncation
    print("\nApplying fingerprint truncation...")
    df = fingerprint_truncation(df)
    
    # Create statistics
    print("\nGenerating statistics and visualizations...")
    stats_results, plots = stats_creation(df)
    
    # Create report
    print("\nCreating report...")
    create_report(df, stats_results, plots)
    
    # Save processed data sample
    df.head(1000).to_csv('/home/claude/processed_sample.csv', index=False)
    print("\nSaved processed sample to processed_sample.csv")
    
    print("\nAnalysis complete! Check the following files:")
    print("- analysis_report.docx : Comprehensive report")
    print("- device_distribution.html : Device frequency chart")
    print("- release_type_distribution.html : Release type distribution")
    print("- top_base_apps.html : Top apps in base configuration")
    print("- top_variant_apps.html : Top apps in variant configuration")
    print("- processed_sample.csv : Sample of processed data")

if __name__ == "__main__":
    main()
