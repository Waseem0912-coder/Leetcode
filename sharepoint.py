import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import os

def setup_driver():
    """Setup Chrome driver with options"""
    chrome_options = Options()
    # Uncomment the next line if you want to run in headless mode
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Initialize the driver
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def wait_for_login(driver, base_url):
    """Wait for user to login by checking for a specific element that indicates login is complete"""
    print("Please log in to the system...")
    driver.get(base_url)

    # Wait for a specific element that indicates login is complete
    try:
        # Try different common elements that appear after login
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.TAG_NAME, "main")) or
            EC.presence_of_element_located((By.CLASS_NAME, "builds")) or
            EC.presence_of_element_located((By.CSS_SELECTOR, "build-list"))
        )
        print("Login detected. Proceeding with automation.")
        return True
    except:
        print("Timeout waiting for login. Please check the system.")
        return False

def wait_for_user_input():
    """Wait for user to press Enter before continuing"""
    input("Press Enter after logging in to start processing...")

def search_builds(driver, base_url, fingerprint):
    """Search for a build by fingerprint using URL parameter"""
    # Construct URL with fingerprint parameter
    search_url = f"{base_url}{fingerprint}"

    # Navigate to the search URL
    driver.get(search_url)

    # Wait for results to load
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "mat-table"))
        )
        return True
    except Exception as e:
        print(f"Error waiting for search results: {e}")
        return False

def extract_build_links(driver):
    """Extract build links from the search results table"""
    build_links = []
    try:
        # Wait for the table to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "mat-table"))
        )

        # Give extra time for dynamic content to render
        time.sleep(2)

        # Find all rows in the table
        rows = driver.find_elements(By.CSS_SELECTOR, "mat-row")
        print(f"Found {len(rows)} rows in the table")

        for idx, row in enumerate(rows):
            try:
                # Try multiple strategies to find the link
                link = None
                
                # Strategy 1: Find link in fingerprint column
                try:
                    fingerprint_cell = row.find_element(By.CSS_SELECTOR, "mat-cell[cdk-column-fingerprint]")
                    anchor = fingerprint_cell.find_element(By.TAG_NAME, "a")
                    link = anchor.get_attribute("href")
                    print(f"Row {idx+1}: Found link via fingerprint column: {link}")
                except:
                    pass
                
                # Strategy 2: Find any anchor with linkableButton class in the row
                if not link:
                    try:
                        anchor = row.find_element(By.CSS_SELECTOR, "a.linkableButton")
                        link = anchor.get_attribute("href")
                        print(f"Row {idx+1}: Found link via linkableButton class: {link}")
                    except:
                        pass
                
                # Strategy 3: Find any clickable anchor in the row
                if not link:
                    try:
                        anchors = row.find_elements(By.TAG_NAME, "a")
                        for anchor in anchors:
                            href = anchor.get_attribute("href")
                            if href and "builds/" in href:
                                link = href
                                print(f"Row {idx+1}: Found link via generic anchor search: {link}")
                                break
                    except:
                        pass
                
                # Strategy 4: Click the row and capture URL change
                if not link:
                    try:
                        current_url = driver.current_url
                        row.click()
                        time.sleep(1)
                        new_url = driver.current_url
                        if new_url != current_url and "builds/" in new_url:
                            link = new_url
                            print(f"Row {idx+1}: Found link via row click: {link}")
                            # Navigate back to search results
                            driver.back()
                            time.sleep(2)
                    except:
                        pass
                
                if link:
                    build_links.append(link)
                else:
                    print(f"Row {idx+1}: No link found")
                    # Debug: Print row HTML
                    print(f"Row HTML snippet: {row.get_attribute('innerHTML')[:200]}...")

            except Exception as e:
                print(f"Error processing row {idx+1}: {e}")
                continue

    except Exception as e:
        print(f"Error extracting build links: {e}")
        import traceback
        traceback.print_exc()

    print(f"Total links extracted: {len(build_links)}")
    return build_links


def navigate_with_retry(driver, url, max_retries=3, retry_delay=3):
    """Navigate to URL with retry logic for network errors"""
    for attempt in range(max_retries):
        try:
            driver.get(url)
            # Wait a moment to ensure page starts loading
            time.sleep(1)
            
            # Check if we got a network error page
            page_source = driver.page_source.lower()
            if "err_name_not_resolved" in page_source or "err_connection" in page_source:
                raise Exception("Network error detected in page")
            
            return True
        except Exception as e:
            print(f"  Navigation attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"  Failed to navigate after {max_retries} attempts")
                return False
    return False

def get_approved_by(driver):
    """Extract 'Approved By' information from the build detail page"""
    try:
        # Wait for metadata section to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "build-metadata"))
        )

        # Find the build metadata section
        metadata_section = driver.find_element(By.TAG_NAME, "build-metadata")

        # Look for 'Approved By' label and extract its value
        labels = metadata_section.find_elements(By.TAG_NAME, "label")
        for label in labels:
            if "Approved By" in label.text:
                # Find the corresponding output element
                parent_row = label.find_element(By.XPATH, "./parent::ape-labeled-row")
                output_element = parent_row.find_element(By.TAG_NAME, "output")
                approved_by = output_element.text.strip()
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

def process_csv_data(driver, base_url, csv_file):
    """Process all fingerprints from CSV file"""
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Limit to first 5 rows for testing
    df = df[:5]

    # Create lists to store results
    all_results = []
    multi_search_results = []

    print(f"Processing {len(df)} fingerprints from {csv_file}")

    for index, row in df.iterrows():
        fingerprint = row['Fingerprint']
        print(f"\n{'='*60}")
        print(f"Processing fingerprint {index + 1}/{len(df)}: {fingerprint}")
        print(f"{'='*60}")

        # Search for the build using URL parameter
        if not search_builds(driver, base_url, fingerprint):
            continue

        # Extract links from results page
        build_links = extract_build_links(driver)

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

                # Navigate to detail page with retry
                if not navigate_with_retry(driver, link):
                    print(f"  Skipping build {i+1} due to navigation failure")
                    multi_search_results.append({
                        'Fingerprint': fingerprint,
                        'Build Link': link,
                        'Approved By': 'Navigation Error',
                        'Approval Type': 'Navigation Error'
                    })
                    continue

                # Wait for page to load
                time.sleep(2)

                # Get approved by info
                approved_by = get_approved_by(driver)
                approval_type = classify_approval_type(approved_by)

                multi_search_results.append({
                    'Fingerprint': fingerprint,
                    'Build Link': link,
                    'Approved By': approved_by,
                    'Approval Type': approval_type
                })
        else:
            # Single build case - navigate to detail page
            link = build_links[0]
            print(f"Single build found: {link}")

            # Navigate to detail page with retry
            if not navigate_with_retry(driver, link):
                print(f"  Skipping due to navigation failure")
                all_results.append({
                    'Fingerprint': fingerprint,
                    'Build Link': link,
                    'Approved By': 'Navigation Error',
                    'Approval Type': 'Navigation Error'
                })
                continue

            # Wait for page to load
            time.sleep(2)

            # Get approved by info
            approved_by = get_approved_by(driver)
            approval_type = classify_approval_type(approved_by)

            all_results.append({
                'Fingerprint': fingerprint,
                'Build Link': link,
                'Approved By': approved_by,
                'Approval Type': approval_type
            })

    return all_results, multi_search_results

def main():
    # Base URL from requirements
    base_url = "https://partner.android.com/approvals/builds?a=8010&q="

    # Hardcoded CSV file path
    csv_file = "SearchData.csv"

    # Setup driver
    driver = setup_driver()

    try:
        # Wait for user to log in
        if not wait_for_login(driver, base_url):
            print("Failed to detect login. Exiting.")
            return

        # Wait for user input before starting processing
        wait_for_user_input()

        # Process CSV data
        if not os.path.exists(csv_file):
            print(f"{csv_file} file not found. Please create a CSV with Fingerprint column.")
            return

        all_results, multi_search_results = process_csv_data(driver, base_url, csv_file)

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
        driver.quit()

if __name__ == "__main__":
    main()import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import os

def setup_driver():
    """Setup Chrome driver with options"""
    chrome_options = Options()
    # Uncomment the next line if you want to run in headless mode
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Initialize the driver
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def wait_for_login(driver, base_url):
    """Wait for user to login by checking for a specific element that indicates login is complete"""
    print("Please log in to the system...")
    driver.get(base_url)

    # Wait for a specific element that indicates login is complete
    try:
        # Try different common elements that appear after login
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.TAG_NAME, "main")) or
            EC.presence_of_element_located((By.CLASS_NAME, "builds")) or
            EC.presence_of_element_located((By.CSS_SELECTOR, "build-list"))
        )
        print("Login detected. Proceeding with automation.")
        return True
    except:
        print("Timeout waiting for login. Please check the system.")
        return False

def wait_for_user_input():
    """Wait for user to press Enter before continuing"""
    input("Press Enter after logging in to start processing...")

def search_builds(driver, base_url, fingerprint):
    """Search for a build by fingerprint using URL parameter"""
    # Construct URL with fingerprint parameter
    search_url = f"{base_url}{fingerprint}"

    # Navigate to the search URL
    driver.get(search_url)

    # Wait for results to load
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "mat-table"))
        )
        return True
    except Exception as e:
        print(f"Error waiting for search results: {e}")
        return False

def extract_build_links(driver):
    """Extract build links from the search results table"""
    build_links = []
    try:
        # Wait for the table to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "mat-table"))
        )

        # Give extra time for dynamic content to render
        time.sleep(2)

        # Find all rows in the table
        rows = driver.find_elements(By.CSS_SELECTOR, "mat-row")
        print(f"Found {len(rows)} rows in the table")

        for idx, row in enumerate(rows):
            try:
                # Try multiple strategies to find the link
                link = None
                
                # Strategy 1: Find link in fingerprint column
                try:
                    fingerprint_cell = row.find_element(By.CSS_SELECTOR, "mat-cell[cdk-column-fingerprint]")
                    anchor = fingerprint_cell.find_element(By.TAG_NAME, "a")
                    link = anchor.get_attribute("href")
                    print(f"Row {idx+1}: Found link via fingerprint column: {link}")
                except:
                    pass
                
                # Strategy 2: Find any anchor with linkableButton class in the row
                if not link:
                    try:
                        anchor = row.find_element(By.CSS_SELECTOR, "a.linkableButton")
                        link = anchor.get_attribute("href")
                        print(f"Row {idx+1}: Found link via linkableButton class: {link}")
                    except:
                        pass
                
                # Strategy 3: Find any clickable anchor in the row
                if not link:
                    try:
                        anchors = row.find_elements(By.TAG_NAME, "a")
                        for anchor in anchors:
                            href = anchor.get_attribute("href")
                            if href and "builds/" in href:
                                link = href
                                print(f"Row {idx+1}: Found link via generic anchor search: {link}")
                                break
                    except:
                        pass
                
                # Strategy 4: Click the row and capture URL change
                if not link:
                    try:
                        current_url = driver.current_url
                        row.click()
                        time.sleep(1)
                        new_url = driver.current_url
                        if new_url != current_url and "builds/" in new_url:
                            link = new_url
                            print(f"Row {idx+1}: Found link via row click: {link}")
                            # Navigate back to search results
                            driver.back()
                            time.sleep(2)
                    except:
                        pass
                
                if link:
                    build_links.append(link)
                else:
                    print(f"Row {idx+1}: No link found")
                    # Debug: Print row HTML
                    print(f"Row HTML snippet: {row.get_attribute('innerHTML')[:200]}...")

            except Exception as e:
                print(f"Error processing row {idx+1}: {e}")
                continue

    except Exception as e:
        print(f"Error extracting build links: {e}")
        import traceback
        traceback.print_exc()

    print(f"Total links extracted: {len(build_links)}")
    return build_links


def navigate_with_retry(driver, url, max_retries=3, retry_delay=3):
    """Navigate to URL with retry logic for network errors"""
    for attempt in range(max_retries):
        try:
            driver.get(url)
            # Wait a moment to ensure page starts loading
            time.sleep(1)
            
            # Check if we got a network error page
            page_source = driver.page_source.lower()
            if "err_name_not_resolved" in page_source or "err_connection" in page_source:
                raise Exception("Network error detected in page")
            
            return True
        except Exception as e:
            print(f"  Navigation attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"  Failed to navigate after {max_retries} attempts")
                return False
    return False

def get_approved_by(driver):
    """Extract 'Approved By' information from the build detail page"""
    try:
        # Wait for metadata section to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "build-metadata"))
        )

        # Find the build metadata section
        metadata_section = driver.find_element(By.TAG_NAME, "build-metadata")

        # Look for 'Approved By' label and extract its value
        labels = metadata_section.find_elements(By.TAG_NAME, "label")
        for label in labels:
            if "Approved By" in label.text:
                # Find the corresponding output element
                parent_row = label.find_element(By.XPATH, "./parent::ape-labeled-row")
                output_element = parent_row.find_element(By.TAG_NAME, "output")
                approved_by = output_element.text.strip()
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

def process_csv_data(driver, base_url, csv_file):
    """Process all fingerprints from CSV file"""
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Limit to first 5 rows for testing
    df = df[:5]

    # Create lists to store results
    all_results = []
    multi_search_results = []

    print(f"Processing {len(df)} fingerprints from {csv_file}")

    for index, row in df.iterrows():
        fingerprint = row['Fingerprint']
        print(f"\n{'='*60}")
        print(f"Processing fingerprint {index + 1}/{len(df)}: {fingerprint}")
        print(f"{'='*60}")

        # Search for the build using URL parameter
        if not search_builds(driver, base_url, fingerprint):
            continue

        # Extract links from results page
        build_links = extract_build_links(driver)

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

                # Navigate to detail page with retry
                if not navigate_with_retry(driver, link):
                    print(f"  Skipping build {i+1} due to navigation failure")
                    multi_search_results.append({
                        'Fingerprint': fingerprint,
                        'Build Link': link,
                        'Approved By': 'Navigation Error',
                        'Approval Type': 'Navigation Error'
                    })
                    continue

                # Wait for page to load
                time.sleep(2)

                # Get approved by info
                approved_by = get_approved_by(driver)
                approval_type = classify_approval_type(approved_by)

                multi_search_results.append({
                    'Fingerprint': fingerprint,
                    'Build Link': link,
                    'Approved By': approved_by,
                    'Approval Type': approval_type
                })
        else:
            # Single build case - navigate to detail page
            link = build_links[0]
            print(f"Single build found: {link}")

            # Navigate to detail page with retry
            if not navigate_with_retry(driver, link):
                print(f"  Skipping due to navigation failure")
                all_results.append({
                    'Fingerprint': fingerprint,
                    'Build Link': link,
                    'Approved By': 'Navigation Error',
                    'Approval Type': 'Navigation Error'
                })
                continue

            # Wait for page to load
            time.sleep(2)

            # Get approved by info
            approved_by = get_approved_by(driver)
            approval_type = classify_approval_type(approved_by)

            all_results.append({
                'Fingerprint': fingerprint,
                'Build Link': link,
                'Approved By': approved_by,
                'Approval Type': approval_type
            })

    return all_results, multi_search_results

def main():
    # Base URL from requirements
    base_url = "https://partner.android.com/approvals/builds?a=8010&q="

    # Hardcoded CSV file path
    csv_file = "SearchData.csv"

    # Setup driver
    driver = setup_driver()

    try:
        # Wait for user to log in
        if not wait_for_login(driver, base_url):
            print("Failed to detect login. Exiting.")
            return

        # Wait for user input before starting processing
        wait_for_user_input()

        # Process CSV data
        if not os.path.exists(csv_file):
            print(f"{csv_file} file not found. Please create a CSV with Fingerprint column.")
            return

        all_results, multi_search_results = process_csv_data(driver, base_url, csv_file)

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
        driver.quit()

if __name__ == "__main__":
    main()
