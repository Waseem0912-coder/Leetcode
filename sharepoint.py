import sys
import pandas as pd
import time
from datetime import datetime
import re
import argparse
import json
import subprocess

# Gemini setup
if "google.colab" in sys.modules:
    from google.colab import auth
    auth.authenticate_user()

import os

PROJECT_ID = "gen-lang-test1"
LOCATION = "global"
MODEL_ID = "gemini-2.0-flash-exp"  # or your preferred Gemini model
API_HOST = "aiplatform.googleapis.com"

def get_access_token():
    """Get Google Cloud access token"""
    try:
        result = subprocess.run(
            ['gcloud', 'auth', 'print-access-token'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Error getting access token: {e}")
        return None

def gemini_generate_snapshot(row):
    """Generate a snapshot using Gemini API"""
    prompt = f"""
Original Issue Data:
ISSUE DETAILS:
Priority: {row['PRIORITY'] if pd.notna(row['PRIORITY']) else 'Missing'}
Type: {row['TYPE'] if pd.notna(row['TYPE']) else 'Missing'}
Title: {row['TITLE'] if pd.notna(row['TITLE']) else 'Missing'}
Assignee: {row['ASSIGNEE'] if pd.notna(row['ASSIGNEE']) else 'Missing'}
Status: {row['STATUS'] if pd.notna(row['STATUS']) else 'Missing'}
Issue ID: {row['ISSUE_ID'] if pd.notna(row['ISSUE_ID']) else 'Missing'}
Modified Time: {row['MODIFIED_TIME (UTC)'] if pd.notna(row['MODIFIED_TIME (UTC)']) else 'Missing'}
DESCRIPTION:
{row['description'] if pd.notna(row['description']) else 'No description provided'}
COMMENTS:
{row['comments'] if pd.notna(row['comments']) else 'No comments provided'}
Category Labels:
Framework, Dex, Apps, Hardware, Settings, Smart Switch, Keyboard, Connectivity, Camera, Others, Notification, UX/SystemUI, SUW, Home/Launcher, Work Profile
Sub-Category Labels (technology-focused, no vendor names):
Performance, Compatibility, Data Transfer, Input Methods, WiFi, Bluetooth, NFC, Mobile Data, Battery, Display, Audio, Sensors, Camera Functionality, Biometrics, User Preferences, Accessibility, Privacy, Account Management, Backup/Restore, Virtual Keyboard, Physical Keyboard, Text Prediction, Keyboard Layout, Photo Quality, Video Recording, Camera Modes, App Management, App Updates, App Crashes, App Permissions, System Settings, Navigation, Themes, Widgets, Setup Process, Device Registration, Home Screen, App Drawer, Icons, Gestures, App Switching, Corporate Profile, Security Policies, Work/Personal Separation, MDM Integration, Desktop Experience, Window Management, Dual Mode, Touch Response, Animations, Status Bar, Notification Panel, Notification Sounds, Do Not Disturb, Initial Configuration, Account Setup, Migration, Cross-platform Sync, Transfer Speed, Hotspot, AirDrop, Network Stability, Camera Hardware, Pre-installed Apps, Third-party Apps, App Store, System APIs, Memory Management, Boot Process, System Stability, User Interface, Miscellaneous, Unknown, Uncategorized, General Issues, Push Notifications
Chain of Thought Instructions:
1. Category Extraction:
- CRITICAL: Extract the exact text from the PRIORITY field in ISSUE DETAILS. This is a mandatory requirement.
- The PRIORITY field value is: {row['PRIORITY'] if pd.notna(row['PRIORITY']) else '[Missing]'}
- You MUST use this exact value in your output as "Priority [Exact PRIORITY Value]"
- If PRIORITY field is missing: Output "[Missing]"
- Assign a single most relevant Category Label from the provided list by analyzing the core issue domain in DESCRIPTION and COMMENTS
- Assign one or multiple relevant Sub-Category Labels that describe specific aspects of the issue (technology-focused, no vendor names)
- Select only the most relevant labels that directly relate to the issue content
- Output format: "Priority [Value from Priority field or "[Missing]"] Category: [Category Label] [Space-separated list of relevant Sub-Category Labels]"
- IMPORTANT: Do not include "Sub-Category Labels:" in the output, just list the labels directly after the Category Label
- CORRECT EXAMPLE: "Priority P2 Category: Connectivity Bluetooth Network Stability"
- INCORRECT EXAMPLE: "Priority P2 Category: Connectivity Sub-Category Labels: Bluetooth Network Stability"
- CORRECT FORMAT:
Priority P2 Category: Connectivity Bluetooth Network Stability
Highlights:
  • Issue: [Issue description] Status: [Status]
  • Problem: [Problem description]
  • Final Action: [Final action description]
2. Highlights Construction:
- Issue: Extract from TITLE or DESCRIPTION
- Status: Use the exact text from the STATUS field in ISSUE DETAILS (e.g., "Open", "Fixed", "In Progress", "Won't Fix", "Missing" if unclear)
- Problem Description: Root cause from DESCRIPTION/COMMENTS (executive view)
- Final Action: From COMMENTS/DESCRIPTION (concise explanation of resolution actions, clearly identifying whether Google or Samsung took the action and what was done)
3. Formatting Requirements:
- Format output exactly as specified below with no additional sections or content
- Ensure no duplication of content between sections
- Each bullet point in the Highlights section must be on its own line
- The Highlights section must contain exactly three bullet points in this order:
  1. Issue and Status on the same line
  2. Problem on its own line
  3. Final Action on its own line
- The "Highlights:" section header must be on its own line, not combined with the Category line
- Use the exact format specified to prevent structural issues
4. Structure Output:
- Priority [Value from Priority field or "[Missing]"] Category: [Category Label] [Space-separated list of relevant Sub-Category Labels]
- Highlights:
  • Issue: [value] Status: [value] (Use the exact text from the STATUS field in ISSUE DETAILS, e.g., "Open", "Fixed", "In Progress", "Won't Fix", "Missing" if unclear)
  • Problem: [value]
  • Final Action: [value] (Concise explanation of resolution actions, clearly identifying whether Google or Samsung took the action and what was done)
5. Important Rules:
- Strictly use only input field data (no inferences)
- Use "[Missing]" for undefined values
- Output exactly two sections in this precise order:
  1. Category section (exactly one line)
  2. Highlights section (exactly three lines)
- Do not duplicate content
- Follow the exact format specified to prevent structural issues
"""
    
    try:
        import requests
        
        access_token = get_access_token()
        if not access_token:
            return "Error: Could not get access token"
        
        api_endpoint = f"https://{API_HOST}/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": {
                "role": "USER",
                "parts": {"text": prompt}
            },
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.3,
                "maxOutputTokens": 2500
            }
        }
        
        response = requests.post(api_endpoint, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        full_response = result['candidates'][0]['content']['parts'][0]['text'].strip()
        
        # Clean up response (remove think tags if present)
        pattern = r"<think>.*?</think>"
        cleaned_response = re.sub(pattern, "", full_response, flags=re.DOTALL)
        
        return cleaned_response.strip()
        
    except Exception as e:
        print(f"Error processing issue {row['ISSUE_ID']}: {e}")
        return f"Error generating snapshot: {str(e)}"

def ollama_generate_snapshot(row):
    """Generate a snapshot for a single issue with the specified format"""
    prompt = f"""
Original Issue Data:
ISSUE DETAILS:
Priority: {row['PRIORITY'] if pd.notna(row['PRIORITY']) else 'Missing'}
Type: {row['TYPE'] if pd.notna(row['TYPE']) else 'Missing'}
Title: {row['TITLE'] if pd.notna(row['TITLE']) else 'Missing'}
Assignee: {row['ASSIGNEE'] if pd.notna(row['ASSIGNEE']) else 'Missing'}
Status: {row['STATUS'] if pd.notna(row['STATUS']) else 'Missing'}
Issue ID: {row['ISSUE_ID'] if pd.notna(row['ISSUE_ID']) else 'Missing'}
Modified Time: {row['MODIFIED_TIME (UTC)'] if pd.notna(row['MODIFIED_TIME (UTC)']) else 'Missing'}
DESCRIPTION:
{row['description'] if pd.notna(row['description']) else 'No description provided'}
COMMENTS:
{row['comments'] if pd.notna(row['comments']) else 'No comments provided'}
Category Labels:
Framework, Dex, Apps, Hardware, Settings, Smart Switch, Keyboard, Connectivity, Camera, Others, Notification, UX/SystemUI, SUW, Home/Launcher, Work Profile
Sub-Category Labels (technology-focused, no vendor names):
Performance, Compatibility, Data Transfer, Input Methods, WiFi, Bluetooth, NFC, Mobile Data, Battery, Display, Audio, Sensors, Camera Functionality, Biometrics, User Preferences, Accessibility, Privacy, Account Management, Backup/Restore, Virtual Keyboard, Physical Keyboard, Text Prediction, Keyboard Layout, Photo Quality, Video Recording, Camera Modes, App Management, App Updates, App Crashes, App Permissions, System Settings, Navigation, Themes, Widgets, Setup Process, Device Registration, Home Screen, App Drawer, Icons, Gestures, App Switching, Corporate Profile, Security Policies, Work/Personal Separation, MDM Integration, Desktop Experience, Window Management, Dual Mode, Touch Response, Animations, Status Bar, Notification Panel, Notification Sounds, Do Not Disturb, Initial Configuration, Account Setup, Migration, Cross-platform Sync, Transfer Speed, Hotspot, AirDrop, Network Stability, Camera Hardware, Pre-installed Apps, Third-party Apps, App Store, System APIs, Memory Management, Boot Process, System Stability, User Interface, Miscellaneous, Unknown, Uncategorized, General Issues, Push Notifications
Chain of Thought Instructions:
1. Category Extraction:
- CRITICAL: Extract the exact text from the PRIORITY field in ISSUE DETAILS. This is a mandatory requirement.
- The PRIORITY field value is: {row['PRIORITY'] if pd.notna(row['PRIORITY']) else '[Missing]'}
- You MUST use this exact value in your output as "Priority [Exact PRIORITY Value]"
- If PRIORITY field is missing: Output "[Missing]"
- Assign a single most relevant Category Label from the provided list by analyzing the core issue domain in DESCRIPTION and COMMENTS
- Assign one or multiple relevant Sub-Category Labels that describe specific aspects of the issue (technology-focused, no vendor names)
- Select only the most relevant labels that directly relate to the issue content
- Output format: "Priority [Value from Priority field or "[Missing]"] Category: [Category Label] [Space-separated list of relevant Sub-Category Labels]"
- IMPORTANT: Do not include "Sub-Category Labels:" in the output, just list the labels directly after the Category Label
- CORRECT EXAMPLE: "Priority P2 Category: Connectivity Bluetooth Network Stability"
- INCORRECT EXAMPLE: "Priority P2 Category: Connectivity Sub-Category Labels: Bluetooth Network Stability"
- CORRECT FORMAT:
Priority P2 Category: Connectivity Bluetooth Network Stability
Highlights:
  • Issue: [Issue description] Status: [Status]
  • Problem: [Problem description]
  • Final Action: [Final action description]
2. Highlights Construction:
- Issue: Extract from TITLE or DESCRIPTION
- Status: Use the exact text from the STATUS field in ISSUE DETAILS (e.g., "Open", "Fixed", "In Progress", "Won't Fix", "Missing" if unclear)
- Problem Description: Root cause from DESCRIPTION/COMMENTS (executive view)
- Final Action: From COMMENTS/DESCRIPTION (concise explanation of resolution actions, clearly identifying whether Google or Samsung took the action and what was done)
3. Formatting Requirements:
- Format output exactly as specified below with no additional sections or content
- Ensure no duplication of content between sections
- Each bullet point in the Highlights section must be on its own line
- The Highlights section must contain exactly three bullet points in this order:
  1. Issue and Status on the same line
  2. Problem on its own line
  3. Final Action on its own line
- The "Highlights:" section header must be on its own line, not combined with the Category line
- Use the exact format specified to prevent structural issues
4. Structure Output:
- Priority [Value from Priority field or "[Missing]"] Category: [Category Label] [Space-separated list of relevant Sub-Category Labels]
- Highlights:
  • Issue: [value] Status: [value] (Use the exact text from the STATUS field in ISSUE DETAILS, e.g., "Open", "Fixed", "In Progress", "Won't Fix", "Missing" if unclear)
  • Problem: [value]
  • Final Action: [value] (Concise explanation of resolution actions, clearly identifying whether Google or Samsung took the action and what was done)
5. Important Rules:
- Strictly use only input field data (no inferences)
- Use "[Missing]" for undefined values
- Output exactly two sections in this precise order:
  1. Category section (exactly one line)
  2. Highlights section (exactly three lines)
- Do not duplicate content
- Follow the exact format specified to prevent structural issues
"""
    try:
        import ollama
        # Send to Ollama model with parameters to reduce hallucinations
        response = ollama.generate(
            model='hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q5_K_M',
            prompt=prompt,
            options={
                'temperature': 0.2,
                'top_p': 0.3,
                'num_predict': 2500,
                'repeat_penalty': 1.1
            }
        )

        full_response = response['response'].strip()
        pattern = r"<think>.*?</think>"
        cleaned_response = re.sub(pattern, "", full_response, flags=re.DOTALL)
        return cleaned_response.strip()
    except Exception as e:
        print(f"Error processing issue {row['ISSUE_ID']}: {e}")
        return f"Error generating snapshot: {str(e)}"

def process_csv(input_file, output_file, use_gemini=False):
    """Process the entire CSV file and save snapshots"""
    
    # Read the input CSV
    df = pd.read_csv(input_file)
    
    # Create a new dataframe for snapshots
    snapshot_data = []
    
    model_name = "Gemini" if use_gemini else "Ollama"
    print(f"Processing {len(df)} issues using {model_name}...")
    
    # Process each row
    for index, row in df.iterrows():
        print(f"Processing issue {index + 1}/{len(df)}: {row['ISSUE_ID']}")
        
        # Generate snapshot using selected model
        if use_gemini:
            snapshot = gemini_generate_snapshot(row)
        else:
            snapshot = ollama_generate_snapshot(row)
        
        # Add to snapshot data
        snapshot_data.append({
            'ISSUE_ID': row['ISSUE_ID'],
            'SNAPSHOT': snapshot,
            'TIMESTAMP': datetime.now().isoformat()
        })
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    # Create DataFrame with snapshots
    snapshots_df = pd.DataFrame(snapshot_data)
    
    # Save to CSV
    snapshots_df.to_csv(output_file, index=False)
    print(f"Snapshots saved to {output_file}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV and generate snapshots')
    parser.add_argument('--input', default='test.csv', help='Input CSV file path')
    parser.add_argument('--output', default='test_output.csv', help='Output CSV file path')
    parser.add_argument('--use-gemini', action='store_true', help='Use Gemini instead of Ollama')
    
    args = parser.parse_args()
    
    process_csv(args.input, args.output, use_gemini=args.use_gemini)
