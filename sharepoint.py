"""
LLM-Powered Insights Generator for Fingerprint Data

This module uses Ollama to generate intelligent insights about the fingerprint
grouping data, analyzing patterns in priv_base_apps and priv_var_apps across
devices and build types.

Usage:
    python generate_insights.py --input fingerprint_report.json --output insights_report.json
"""

import json
import subprocess
from typing import Dict, List, Any
from collections import Counter
import argparse


def call_ollama(prompt: str, model: str = "llama3.2") -> str:
    """
    Call Ollama API to generate text.
    
    Args:
        prompt: The prompt to send to the model
        model: The Ollama model to use (default: llama3.2)
    
    Returns:
        Generated text response
    """
    try:
        # Use subprocess to call ollama
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
            print(f"Ollama error: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("Ollama timed out")
        return None
    except FileNotFoundError:
        print("Ollama not found, using intelligent fallback insights")
        return None
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None


def generate_intelligent_insight(device: str, build_type: str, data: Dict) -> str:
    """
    Generate intelligent insights based on data analysis when LLM is unavailable.
    This provides meaningful analysis based on statistical patterns.
    """
    combinations = data.get('combinations', [])
    
    # Collect all apps
    all_base_apps = []
    all_var_apps = []
    combo_sizes = []
    
    for combo in combinations:
        base = combo.get('priv_base_apps', [])
        var = combo.get('priv_var_apps', [])
        all_base_apps.extend(base)
        all_var_apps.extend(var)
        combo_sizes.append((len(base), len(var)))
    
    # Analyze
    base_counter = Counter(all_base_apps)
    var_counter = Counter(all_var_apps)
    
    base_unique = len(base_counter)
    var_unique = len(var_counter)
    base_total = len(all_base_apps)
    var_total = len(all_var_apps)
    
    # Find dominant apps
    base_dominant = [app for app, count in base_counter.items() if count > 1]
    var_dominant = [app for app, count in var_counter.items() if count > 1]
    
    # Find single-occurrence apps
    base_single = [app for app, count in base_counter.items() if count == 1]
    var_single = [app for app, count in var_counter.items() if count == 1]
    
    # Calculate average apps per combination
    avg_base = base_total / len(combinations) if combinations else 0
    avg_var = var_total / len(combinations) if combinations else 0
    
    # Build insight text
    insights = []
    
    # PRIV BASE APPS INSIGHT
    insights.append("**PRIV BASE APPS INSIGHT**")
    if base_dominant:
        insights.append(f"Analysis identified {len(base_dominant)} dominant base app(s) appearing across multiple combinations: {', '.join(base_dominant[:3])}. These represent core privileged applications consistently required for this device configuration.")
    elif base_unique == base_total:
        insights.append(f"All {base_unique} base apps are unique across combinations, indicating highly specialized privileged application requirements for each fingerprint pairing. No single app dominates the configuration.")
    else:
        insights.append(f"The {base_unique} unique base apps show a distributed pattern across {len(combinations)} combinations, averaging {avg_base:.1f} apps per combination.")
    
    if len(base_single) > base_unique * 0.7:
        insights.append(f"Notable: {len(base_single)} apps ({len(base_single)*100//base_unique}%) appear only once, suggesting fingerprint-specific base app requirements.")
    
    # PRIV VARIANT APPS INSIGHT
    insights.append("\n**PRIV VARIANT APPS INSIGHT**")
    if var_dominant:
        insights.append(f"Variant apps show {len(var_dominant)} recurring application(s): {', '.join(var_dominant[:3])}. These likely represent shared system components across device variants.")
    elif var_unique == var_total:
        insights.append(f"Each of the {var_unique} variant apps is unique to its combination, indicating distinct customization per fingerprint configuration.")
    else:
        insights.append(f"The variant app distribution shows {var_unique} unique apps averaging {avg_var:.1f} per combination.")
    
    # COMBINED ANALYSIS
    insights.append("\n**COMBINED ANALYSIS**")
    
    # Compare distributions
    if base_unique > var_unique:
        insights.append(f"Base apps ({base_unique}) show more diversity than variant apps ({var_unique}), suggesting privileged base configurations are more variable than system variants for this build type.")
    elif var_unique > base_unique:
        insights.append(f"Variant apps ({var_unique}) exceed base app diversity ({base_unique}), indicating system customization varies more than core privileged app requirements.")
    else:
        insights.append(f"Base and variant apps show similar diversity ({base_unique} each), suggesting balanced configuration complexity.")
    
    # Analyze combination sizes
    max_combo = max(combo_sizes, key=lambda x: x[0] + x[1])
    min_combo = min(combo_sizes, key=lambda x: x[0] + x[1])
    
    if max_combo != min_combo:
        insights.append(f"Combination complexity ranges from {sum(min_combo)} to {sum(max_combo)} total apps, indicating variable configuration requirements across fingerprint pairings.")
    
    # KEY FINDINGS
    insights.append("\n**KEY FINDINGS**")
    findings = []
    
    if base_dominant:
        findings.append(f"• Core dependencies: {', '.join(base_dominant[:2])} appear in multiple configurations")
    if len(base_single) > 2:
        findings.append(f"• {len(base_single)} specialized base apps suggest fingerprint-specific requirements")
    if avg_base > 2:
        findings.append(f"• High base app density ({avg_base:.1f} avg) indicates complex privileged requirements")
    if var_unique < base_unique:
        findings.append(f"• Variant apps are more standardized than base apps across this configuration")
    
    if not findings:
        findings.append(f"• Configuration shows standard distribution with {len(combinations)} combinations")
        findings.append(f"• No single app dominates - distributed privileged requirements")
    
    insights.extend(findings)
    
    return "\n".join(insights)


def analyze_app_distribution(apps_list: List[str]) -> Dict:
    """
    Analyze the distribution and patterns in an apps list.
    
    Returns statistics about:
    - Total unique apps
    - Most common apps
    - App frequency distribution
    """
    all_apps = []
    for apps in apps_list:
        if isinstance(apps, list):
            all_apps.extend(apps)
        elif isinstance(apps, str):
            all_apps.append(apps)
    
    counter = Counter(all_apps)
    
    return {
        'total_unique': len(counter),
        'total_occurrences': len(all_apps),
        'most_common': counter.most_common(5),
        'frequency_distribution': dict(counter),
        'single_occurrence_apps': [app for app, count in counter.items() if count == 1],
        'dominant_apps': [app for app, count in counter.items() if count > 1]
    }


def build_analysis_prompt(device: str, build_type: str, data: Dict) -> str:
    """
    Build a comprehensive prompt for LLM analysis of a device/build combination.
    
    This prompt is designed to extract meaningful insights about:
    1. Individual column patterns (priv_base_apps, priv_var_apps)
    2. Combined patterns and correlations
    3. Anomalies and notable observations
    """
    combinations = data.get('combinations', [])
    
    # Collect all apps for analysis
    all_base_apps = []
    all_var_apps = []
    combination_details = []
    
    for combo in combinations:
        fp = combo.get('fingerprint_parsed', combo.get('fingerprint', 'Unknown'))
        sdp = combo.get('same_device_fingerprint_parsed', combo.get('same_device_fingerprint', 'Unknown'))
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
    
    # Analyze distributions
    base_analysis = analyze_app_distribution([all_base_apps])
    var_analysis = analyze_app_distribution([all_var_apps])
    
    # Build the prompt
    prompt = f"""You are a senior Android system analyst reviewing privileged application data for device configurations.

TASK: Analyze the following data for Device "{device}" with Build Type "{build_type}" and provide actionable insights.

=== RAW DATA ===
Total Fingerprint Combinations: {len(combinations)}
Unique Fingerprints: {data.get('unique_fingerprints_parsed', data.get('unique_fingerprints', []))}
Unique Same-Device Fingerprints: {data.get('unique_same_device_fps_parsed', data.get('unique_same_device_fps', []))}

=== COMBINATION BREAKDOWN ===
"""
    
    for i, detail in enumerate(combination_details, 1):
        prompt += f"""
Combination {i}:
  Fingerprint: {detail['fingerprint']}
  Same Device FP: {detail['same_device_fp']}
  Priv Base Apps ({len(detail['base_apps'])}): {', '.join(detail['base_apps']) if detail['base_apps'] else 'None'}
  Priv Variant Apps ({len(detail['var_apps'])}): {', '.join(detail['var_apps']) if detail['var_apps'] else 'None'}
"""

    prompt += f"""
=== STATISTICAL SUMMARY ===

PRIV BASE APPS ANALYSIS:
- Total unique apps: {base_analysis['total_unique']}
- Total occurrences: {base_analysis['total_occurrences']}
- Most common apps: {base_analysis['most_common']}
- Apps appearing only once: {len(base_analysis['single_occurrence_apps'])}
- Dominant apps (>1 occurrence): {base_analysis['dominant_apps']}

PRIV VARIANT APPS ANALYSIS:
- Total unique apps: {var_analysis['total_unique']}
- Total occurrences: {var_analysis['total_occurrences']}
- Most common apps: {var_analysis['most_common']}
- Apps appearing only once: {len(var_analysis['single_occurrence_apps'])}
- Dominant apps (>1 occurrence): {var_analysis['dominant_apps']}

=== ANALYSIS INSTRUCTIONS ===

Please provide a structured analysis with the following sections:

1. **PRIV BASE APPS INSIGHT** (2-3 sentences):
   - Identify patterns: Are certain apps consistently present?
   - Note if one or two apps dominate across combinations
   - Highlight any anomalies (e.g., an app only in one combination)

2. **PRIV VARIANT APPS INSIGHT** (2-3 sentences):
   - Same analysis as above for variant apps
   - Note any differences in distribution pattern vs base apps

3. **COMBINED ANALYSIS** (3-4 sentences):
   - How do base and variant apps relate to each other?
   - Are there fingerprint combinations with notably more/fewer apps?
   - What does the overall distribution suggest about this device/build configuration?

4. **KEY FINDINGS** (bullet points):
   - List 2-4 actionable observations
   - Include any recommendations if patterns suggest issues

Keep the response concise, professional, and data-driven. Focus on patterns that would matter to a system configuration engineer."""

    return prompt


def generate_device_build_insight(device: str, build_type: str, data: Dict) -> Dict:
    """
    Generate comprehensive insights for a specific device/build type combination.
    
    Returns a dictionary with:
    - raw_statistics: Numerical analysis
    - llm_insight: AI-generated narrative insight
    - key_findings: Extracted key points
    """
    combinations = data.get('combinations', [])
    
    # Collect statistics
    all_base_apps = []
    all_var_apps = []
    
    for combo in combinations:
        all_base_apps.extend(combo.get('priv_base_apps', []))
        all_var_apps.extend(combo.get('priv_var_apps', []))
    
    base_stats = analyze_app_distribution([all_base_apps])
    var_stats = analyze_app_distribution([all_var_apps])
    
    # Build prompt and try LLM first
    prompt = build_analysis_prompt(device, build_type, data)
    llm_response = call_ollama(prompt)
    
    # If Ollama failed, use intelligent fallback
    if llm_response is None:
        llm_response = generate_intelligent_insight(device, build_type, data)
    
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


def generate_all_insights(report_data: Dict) -> Dict:
    """
    Generate insights for all device/build type combinations in the report.
    
    Args:
        report_data: The loaded fingerprint_report.json data
    
    Returns:
        Dictionary with insights for each device/build combination
    """
    results = report_data.get('results', {})
    all_insights = {
        'generated_at': str(__import__('datetime').datetime.now()),
        'device_insights': []
    }
    
    for device, build_types in results.items():
        for build_type, data in build_types.items():
            print(f"\nGenerating insight for Device: {device}, Build Type: {build_type}...")
            
            insight = generate_device_build_insight(device, build_type, data)
            all_insights['device_insights'].append(insight)
            
            print(f"  ✓ Generated insight with {insight['statistics']['total_combinations']} combinations")
    
    return all_insights


def main():
    parser = argparse.ArgumentParser(description='Generate LLM-powered insights from fingerprint data')
    parser.add_argument('--input', '-i', default='fingerprint_report.json', help='Input JSON file')
    parser.add_argument('--output', '-o', default='insights_report.json', help='Output insights JSON file')
    parser.add_argument('--model', '-m', default='llama3.2', help='Ollama model to use')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LLM-POWERED INSIGHTS GENERATOR")
    print("="*60)
    
    # Load the processed data
    print(f"\nLoading data from {args.input}...")
    with open(args.input, 'r') as f:
        report_data = json.load(f)
    
    # Generate insights
    print("\nGenerating insights using Ollama...")
    insights = generate_all_insights(report_data)
    
    # Save insights
    print(f"\nSaving insights to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(insights, f, indent=2)
    
    print("\n" + "="*60)
    print("INSIGHTS GENERATION COMPLETE")
    print("="*60)
    
    # Print summary
    for insight in insights['device_insights']:
        print(f"\n--- {insight['device']} / {insight['build_type']} ---")
        print(f"Combinations: {insight['statistics']['total_combinations']}")
        print(f"Unique Base Apps: {insight['statistics']['priv_base_apps']['total_unique']}")
        print(f"Unique Var Apps: {insight['statistics']['priv_var_apps']['total_unique']}")
        print(f"\nLLM Insight Preview:")
        print(insight['llm_insight'][:500] + "..." if len(insight['llm_insight']) > 500 else insight['llm_insight'])


if __name__ == "__main__":
    main()
