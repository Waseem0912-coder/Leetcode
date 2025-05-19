

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import traceback
import csv # For CSV output
import os # For checking if CSV file exists

BASE_URL = "https://issuetracker.google.com/issues/"
CSV_FILENAME = "scraped_google_issues_aggregated.csv" # New filename for clarity

# scrape_issue_tracker_page function remains the same as the previous good version
# I'll include it here for completeness but the core logic inside is unchanged for scraping
def scrape_issue_tracker_page(url, driver):
    print(f"\n--- Scraping URL: {url} ---")
    issue_title = "Not found"
    issue_description_text = "Not found"
    all_comments_list = []

    try:
        driver.get(url)
        time.sleep(3)

        page_title_full = driver.title
        if " [" in page_title_full and "] - Issue Tracker" in page_title_full:
            issue_title = page_title_full.split(" [")[0]
        elif page_title_full and " - Issue Tracker" in page_title_full:
             issue_title = page_title_full.split(" - Issue Tracker")[0]
        elif page_title_full:
            issue_title = page_title_full
        else:
            issue_title = "Title not parseable from page_title"
        print(f"  Extracted Issue Title: {issue_title}")

        description_author = "Unknown"
        description_timestamp = "Unknown"
        try:
            description_container_locator = (By.ID, "comment1")
            description_container_element = WebDriverWait(driver, 15).until(
                EC.visibility_of_element_located(description_container_locator)
            )
            
            desc_text_element = description_container_element.find_element(By.CSS_SELECTOR, ".type-m.markdown-display")
            issue_description_text = desc_text_element.text.strip()

            try:
                desc_author_element = description_container_element.find_element(By.CSS_SELECTOR, "b-user-display-name")
                description_author = desc_author_element.text.strip()
            except NoSuchElementException:
                print("  Warning: Could not find author for description (comment1).")
            
            try:
                desc_time_element = description_container_element.find_element(By.CSS_SELECTOR, "b-formatted-date-time time")
                description_timestamp = desc_time_element.get_attribute("title").strip()
            except NoSuchElementException:
                print("  Warning: Could not find timestamp for description (comment1).")
            
            # Add description to all_comments_list so it's part of the concatenated comments
            # but also keep it separate in issue_description_text
            all_comments_list.append({
                "comment_id": "comment1 (Description)",
                "author": description_author,
                "timestamp_details": description_timestamp,
                "text": issue_description_text,
            })
            print(f"  Extracted Description (Comment #1) details.")

        except TimeoutException:
            print("  Timeout: Description container (comment1) not found.")
            issue_description_text = "Description block (comment1) not found"
        except NoSuchElementException as e:
            print(f"  NoSuchElement: Could not find a sub-element within description (Comment #1): {e}")
            if 'description_container_element' in locals() and issue_description_text == "Not found":
                try:
                    desc_text_element_fallback = description_container_element.find_element(By.CSS_SELECTOR, ".type-m.markdown-display")
                    issue_description_text = desc_text_element_fallback.text.strip()
                    all_comments_list.append({
                        "comment_id": "comment1 (Description - text only)",
                        "author": "Unknown",
                        "timestamp_details": "Unknown",
                        "text": issue_description_text,
                    })
                    print("  Fallback: Extracted only description text for comment1.")
                except:
                    print("  Fallback: Failed to get description text for comment1 either.")
                    issue_description_text = "Description text not extractable"
        except Exception as e:
            print(f"  Error extracting description (Comment #1) details: {e}")
            issue_description_text = f"Error extracting description: {e}"

        try:
            comments_list_container_locator = (By.TAG_NAME, "issue-event-list")
            comments_list_element = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(comments_list_container_locator)
            )
            time.sleep(1) 

            history_event_elements_xpath = ".//div[contains(@class, 'bv2-event')]//b-history-event"
            all_history_events = comments_list_element.find_elements(By.XPATH, history_event_elements_xpath)
            print(f"  Found {len(all_history_events)} total b-history-event elements for comments.")

            if not all_history_events and len(all_comments_list) == 0: # If no description and no other events
                print("  No b-history-event elements found and no initial description found.")
            elif not all_history_events and len(all_comments_list) > 0: # Only description was found
                 print("  No further b-history-event elements found (only initial description if present).")


            for event_element in all_history_events:
                comment_id_attr = event_element.get_attribute("id")
                
                if comment_id_attr == "comment1" and any(c['comment_id'].startswith("comment1 (Description") for c in all_comments_list):
                    continue

                try:
                    comment_text_element = event_element.find_element(By.CSS_SELECTOR, ".type-m.markdown-display")
                    comment_text = comment_text_element.text.strip()

                    author = "Unknown"
                    timestamp = "Unknown"
                    try:
                        author_element = event_element.find_element(By.CSS_SELECTOR, "b-user-display-name")
                        author = author_element.text.strip()
                    except NoSuchElementException:
                        pass # Silently pass if author not found for subsequent comments
                    
                    try:
                        time_element = event_element.find_element(By.CSS_SELECTOR, "b-formatted-date-time time")
                        timestamp = time_element.get_attribute("title").strip()
                    except NoSuchElementException:
                        pass # Silently pass if time not found
                    
                    current_comment_id_str = comment_id_attr if comment_id_attr else "Event (text found)"
                    
                    all_comments_list.append({
                        "comment_id": current_comment_id_str,
                        "author": author,
                        "timestamp_details": timestamp,
                        "text": comment_text,
                    })
                    print(f"    Extracted text comment/event: {current_comment_id_str}")

                except NoSuchElementException:
                    pass 
                except Exception as e_inner:
                    print(f"    Error processing event '{comment_id_attr or 'Unknown ID'}': {e_inner}")

        except TimeoutException:
            print("  Timeout: Main comments container (issue-event-list) not found. No subsequent comments.")
        except Exception as e:
            print(f"  Error extracting subsequent comments: {e}")

    except Exception as e_outer:
        print(f"  Major error during Selenium operations for {url}: {e_outer}")

    return {
        "url": url,
        "title": issue_title,
        "description_text": issue_description_text, # The text of the first comment
        "all_comments_list": all_comments_list # List of dicts for all text entries
    }

def write_to_aggregated_csv(data_list, filename):
    """
    Writes the scraped data to a CSV file with one row per issue.
    Comments are aggregated into a single cell.
    """
    # Define CSV headers
    fieldnames = ['issue_url', 'issue_title', 'description_text', 'all_comments_concatenated']
    
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists: # Write header only if file is new
            writer.writeheader()
            print(f"CSV header written to {filename}")

        for issue_data in data_list:
            # Concatenate all comments into a single string
            # Exclude the initial description from "subsequent comments" if it's already in description_text
            # For this aggregated view, we'll format them nicely.
            
            comments_str_parts = []
            # The first item in all_comments_list is the description (if found)
            # Subsequent items are other comments.
            
            # We already have `issue_data['description_text']` which is the primary description.
            # Now, let's get the *other* comments.
            subsequent_comments = []
            if issue_data.get('all_comments_list'):
                for c_idx, comment_dict in enumerate(issue_data['all_comments_list']):
                    # Only include if it's not the primary description text we already have
                    # or if it's a distinct comment entry
                    if c_idx == 0 and comment_dict['text'] == issue_data['description_text']:
                        # This is the description, already handled by 'description_text' column
                        # However, we might still want its author/time in the concatenated string.
                        # For simplicity now, we'll just take subsequent comments for the 'all_comments_concatenated'
                        pass
                    else: # Subsequent comments or description if it wasn't the first item
                         subsequent_comments.append(
                            f"--- Comment ID: {comment_dict.get('comment_id', 'N/A')} ---\n"
                            f"Author: {comment_dict.get('author', 'N/A')}\n"
                            f"Timestamp: {comment_dict.get('timestamp_details', 'N/A')}\n"
                            f"Text:\n{comment_dict.get('text', '')}\n\n"
                        )
            
            all_comments_concatenated = "".join(subsequent_comments).strip()
            if not all_comments_concatenated and not issue_data.get('description_text', '').startswith("Description block"): # If no subsequent comments, and description was found
                all_comments_concatenated = "No subsequent comments found."
            elif not all_comments_concatenated and issue_data.get('description_text', '').startswith("Description block"): # No description, no comments
                all_comments_concatenated = "No comments or description found."


            writer.writerow({
                'issue_url': issue_data.get('url', 'N/A'),
                'issue_title': issue_data.get('title', 'N/A'),
                'description_text': issue_data.get('description_text', 'Not found'), # This is comment1's text
                'all_comments_concatenated': all_comments_concatenated
            })
    print(f"Aggregated data appended to {filename}")


if __name__ == "__main__":
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        print("Chrome driver initialized in headless mode.")

        issue_ids = [
            "399131921",
            "405946938", # Example of a potentially non-existent issue
            "392668960",    # From your initial JS data - AOSP ID
            "418025711"  # Another test
        ]
        urls_to_scrape = [f"{BASE_URL}{issue_id}" for issue_id in issue_ids]
        
        results_for_csv = []

        for i, url in enumerate(urls_to_scrape):
            print(f"\nProcessing URL {i+1}/{len(urls_to_scrape)}: {url}")
            data = scrape_issue_tracker_page(url, driver)
            results_for_csv.append(data)
            
            if i < len(urls_to_scrape) - 1:
                 time.sleep(2)

        if results_for_csv:
            write_to_aggregated_csv(results_for_csv, CSV_FILENAME) # Use the new CSV writing function
        else:
            print("No data was scraped to write to CSV.")

    except Exception as e:
        print(f"A major error occurred in the main script: {e}")
        traceback.print_exc()
    finally:
        print("\nScript finished. Quitting driver.")
        if driver:
            driver.quit()
