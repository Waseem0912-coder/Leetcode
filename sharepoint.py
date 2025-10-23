import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import time
import os
import random

def setup_browser(playwright):
    """Setup browser with anti-detection measures"""
    browser = playwright.chromium.launch(
        headless=False,  # Set to True if you want headless mode
        args=[
            '--no-sandbox',
            '--disable-blink-features=AutomationControlled',
            '--disable-dev-shm-usage',
        ]
    )
    
    # Create context with realistic browser fingerprint
    context = browser.new_context(
        viewport={'width': 1920, 'height': 1080},
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        locale='en-US',
        timezone_id='America/New_York',
    )
    
    # Add extra headers to look more human
    context.set_extra_http_headers({
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
    })
    
    page = context.new_page()
    
    # Inject script to remove webdriver property
    page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
    """)
    
    return browser, context, page

def random_delay(min_seconds=2, max_seconds=5):
    """Add random delay to mimic human behavior"""
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)

def wait_for_login(page, base_url):
    """Wait for user to login"""
    print("Please log in to the system...")
    page.goto(base_url)
    
    try:
        # Wait for a specific element that indicates login is complete
        page.wait_for_selector("main, .builds, build-list", timeout=60000)
        print("Login detected. Proceeding with automation.")
        return True
    except:
        print("Timeout waiting for login. Please check the system.")
        return False

def wait_for_user_input():
    """Wait for user to press Enter before continuing"""
    input("Press Enter after logging in to start processing...")

def search_builds(page, base_url, fingerprint):
    """Search for a build by fingerprint using URL parameter"""
    search_url = f"{base_url}{fingerprint}"
    
    try:
        page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        random_delay(1, 2)
        
        # Wait for results to load
        page.wait_for_selector("mat-table", timeout=10000)
        return True
    except Exception as e:
        print(f"Error searching for build: {e}")
        return False

def extract_build_links(page):
    """Extract build links from the search results table"""
    build_links = []
    
    try:
        # Wait for table to be present
        page.wait_for_selector("mat-table", timeout=10000)
        random_delay(1, 2)
        
        # Get all rows
        rows = page.query_selector_all("mat-row")
        print(f"Found {len(rows)} rows in the table")
        
        for idx, row in enumerate(rows):
            link = None
            
            # Strategy 1: Find link in fingerprint column
            try:
                fingerprint_cell = row.query_selector("mat-cell[cdk-column-fingerprint]")
                if fingerprint_cell:
                    anchor = fingerprint_cell.query_selector("a")
                    if anchor:
                        link = anchor.get_attribute("href")
                        if link:
                            print(f"Row {idx+1}: Found link via fingerprint column")
            except:
                pass
            
            # Strategy 2: Find any anchor with linkableButton class
            if not link:
                try:
                    anchor = row.query_selector("a.linkableButton")
                    if anchor:
                        link = anchor.get_attribute("href")
                        if link:
                            print(f"Row {idx+1}: Found link via linkableButton class")
                except:
                    pass
            
            # Strategy 3: Find any clickable anchor in the row
            if not link:
                try:
                    anchors = row.query_selector_all("a")
                    for anchor in anchors:
                        href = anchor.get_attribute("href")
                        if href and "builds/" in href:
                            link = href
                            print(f"Row {idx+1}: Found link via generic anchor search")
                            break
                except:
                    pass
            
            if link:
                # Make sure it's a full URL
                if not link.startswith("http"):
                    base = page.url.split("/approvals")[0]
                    link = base + link if link.startswith("/") else base + "/" + link
                build_links.append(link)
            else:
                print(f"Row {idx+1}: No link found")
        
    except Exception as e:
        print(f"Error extracting build links: {e}")
    
    print(f"Total links extracted: {len(build_links)}")
    return build_links

def navigate_with_retry(page, url, max_retries=3, retry_delay=10):
    """Navigate to URL with retry logic and anti-blocking measures"""
    for attempt in range(max_retries):
        try:
            print(f"  Navigating to: {url}")
            
            # Add random delay before navigation to avoid detection
            if attempt > 0:
                random_delay(retry_delay, retry_delay + 3)
            
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            # Wait a bit for page to fully load
            random_delay(2, 3)
            
            # Check for error pages
            content = page.content().lower()
            current_url = page.url.lower()
            
            if any(error in content for error in ["err_name_not_resolved", "err_connection", "err_internet_disconnected", "err_network_changed"]):
                raise Exception("Network error page detected")
            
            if "chrome-error://" in current_url or "data:text/html,chromewebdata" in current_url:
                raise Exception("Chrome error page detected")
            
            # Check if we can find the expected content
            try:
                page.wait_for_selector("build-metadata, body", timeout=5000)
                print(f"  ✅ Navigation successful")
                return True
            except:
                raise Exception("Expected content not found")
                
        except Exception as e:
            print(f"  ❌ Navigation attempt {attempt + 1}/{max_retries} failed: {str(e)[:100]}")
            if attempt < max_retries - 1:
                wait_time = retry_delay + (attempt * 5)  # Increase wait time with each retry
                print(f"  ⏳ Waiting {wait_time} seconds before retry (possible rate limiting)...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Failed to navigate after {max_retries} attempts")
                return False
    
    return False

def get_approved_by(page):
    """Extract 'Approved By' information from the build detail page"""
    try:
        # Wait for metadata section to load
        page.wait_for_selector("build-metadata", timeout=10000)
        
        # Find all labels
        labels = page.query_selector_all("build-metadata label")
        
        for label in labels:
            if "Approved By" in label.text_content():
                # Find the parent row and then the output element
                parent_row = label.evaluate("element => element.closest('ape-labeled-row')")
                if parent_row:
                    output = page.query_selector("ape-labeled-row output")
                    if output:
                        approved_by = output.text_content().strip()
                        return approved_by
        
        return "Not Found"
    except Exception as e:
        print(f"Error getting approved by: {e}")
        return "Error"

def classify_approval_type(approved_by):
    """Classify the approval type based on who approved"""
    if approved_by == "APFE":
        return "Auto"
    elif approved_by == "android-partner-prodops@system.gserviceaccount.com":
        return "Bot"
    elif approved_by in ["Not Found", "Error", "No Builds Found", "Navigation Error"]:
        return approved_by
    else:
        return "Human"

def process_csv_data(page, base_url, csv_file):
    """Process all fingerprints from CSV file"""
    df = pd.read_csv(csv_file)
    
    # Limit to first 5 rows for testing
    df = df[:5]
    
    all_results = []
    multi_search_results = []
    navigation_failures = 0
    
    print(f"Processing {len(df)} fingerprints from {csv_file}")
    
    for index, row in df.iterrows():
        fingerprint = row['Fingerprint']
        print(f"\n{'='*60}")
        print(f"Processing fingerprint {index + 1}/{len(df)}: {fingerprint}")
        print(f"{'='*60}")
        
        # Add random delay between searches to avoid rate limiting
        if index > 0:
            random_delay(3, 6)
        
        # Search for the build
        if not search_builds(page, base_url, fingerprint):
            continue
        
        # Extract links from results page
        build_links = extract_build_links(page)
        
        if len(build_links) == 0:
            print(f"No builds found for fingerprint: {fingerprint}")
            approved_by = "No Builds Found"
            approval_type = classify_approval_type(approved_by)
            all_results.append({
                'Fingerprint': fingerprint,
                'Build Link': '',
                'Approved By': approved_by,
                'Approval Type': approval_type
            })
            continue
        
        # Handle multiple build links
        if len(build_links) > 1:
            print(f"Found {len(build_links)} builds for fingerprint: {fingerprint}")
            for i, link in enumerate(build_links):
                print(f"  Build {i+1}: {link}")
                
                # Add delay between navigations
                random_delay(2, 4)
                
                if not navigate_with_retry(page, link, max_retries=3, retry_delay=10):
                    print(f"  ⚠️  Skipping build {i+1} due to navigation failure")
                    navigation_failures += 1
                    multi_search_results.append({
                        'Fingerprint': fingerprint,
                        'Build Link': link,
                        'Approved By': 'Navigation Error',
                        'Approval Type': 'Navigation Error'
                    })
                    continue
                
                random_delay(2, 3)
                approved_by = get_approved_by(page)
                approval_type = classify_approval_type(approved_by)
                
                multi_search_results.append({
                    'Fingerprint': fingerprint,
                    'Build Link': link,
                    'Approved By': approved_by,
                    'Approval Type': approval_type
                })
        else:
            link = build_links[0]
            print(f"Single build found: {link}")
            
            random_delay(2, 4)
            
            if not navigate_with_retry(page, link, max_retries=3, retry_delay=10):
                print(f"  ⚠️  Skipping due to navigation failure")
                navigation_failures += 1
                all_results.append({
                    'Fingerprint': fingerprint,
                    'Build Link': link,
                    'Approved By': 'Navigation Error',
                    'Approval Type': 'Navigation Error'
                })
                continue
            
            random_delay(2, 3)
            approved_by = get_approved_by(page)
            approval_type = classify_approval_type(approved_by)
            
            all_results.append({
                'Fingerprint': fingerprint,
                'Build Link': link,
                'Approved By': approved_by,
                'Approval Type': approval_type
            })
    
    if navigation_failures > 0:
        print(f"\n⚠️  Warning: {navigation_failures} navigation failures occurred")
    
    return all_results, multi_search_results

def main():
    base_url = "https://partner.android.com/approvals/builds?a=8010&q="
    csv_file = "SearchData.csv"
    
    with sync_playwright() as playwright:
        browser, context, page = setup_browser(playwright)
        
        try:
            # Wait for user to log in
            if not wait_for_login(page, base_url):
                print("Failed to detect login. Exiting.")
                return
            
            # Wait for user input before starting processing
            wait_for_user_input()
            
            # Process CSV data
            if not os.path.exists(csv_file):
                print(f"{csv_file} file not found. Please create a CSV with Fingerprint column.")
                return
            
            all_results, multi_search_results = process_csv_data(page, base_url, csv_file)
            
            # Save results to CSV files
            if all_results:
                df_all = pd.DataFrame(all_results)
                df_all.to_csv("results.csv", index=False)
                print(f"\nSaved {len(all_results)} results to results.csv")
                
                # Print summary of approval types
                print("\n" + "="*60)
                print("APPROVAL TYPE SUMMARY")
                print("="*60)
                approval_counts = df_all['Approval Type'].value_counts()
                for approval_type, count in approval_counts.items():
                    print(f"{approval_type}: {count}")
                print("="*60)
            
            if multi_search_results:
                df_multi = pd.DataFrame(multi_search_results)
                df_multi.to_csv("multi_search_results.csv", index=False)
                print(f"\nSaved {len(multi_search_results)} multi-search results to multi_search_results.csv")
                
                # Print summary for multi-search results too
                print("\n" + "="*60)
                print("MULTI-SEARCH APPROVAL TYPE SUMMARY")
                print("="*60)
                approval_counts_multi = df_multi['Approval Type'].value_counts()
                for approval_type, count in approval_counts_multi.items():
                    print(f"{approval_type}: {count}")
                print("="*60)
        
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            browser.close()

if __name__ == "__main__":
    main()
