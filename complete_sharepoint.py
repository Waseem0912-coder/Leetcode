import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

import pandas as pd


BASE_ISSUE_URL = "https://issuetracker.google.com/issues/"
LOGIN_BASE_URL = "https://issuetracker.google.com/issues"


def build_driver(headless: bool) -> webdriver.Chrome:
    opts = ChromeOptions()
    if headless:
        # Modern headless mode
        opts.add_argument("--headless=new")
    # Sensible defaults for CI/headless environments
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,1000")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])  # reduce automation banner
    opts.add_experimental_option("useAutomationExtension", False)
    try:
        driver = webdriver.Chrome(options=opts)
    except WebDriverException as e:
        print("Failed to start Chrome WebDriver. Ensure Chrome is installed.", file=sys.stderr)
        raise e
    return driver


def wait_for_page_ready(driver: webdriver.Chrome, wait_time: int) -> None:
    WebDriverWait(driver, wait_time).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    # Give Angular/JS a brief moment to render dynamic content
    time.sleep(1.0)


def try_click_more_like_buttons(driver: webdriver.Chrome) -> None:
    # Click any obvious expanders to reveal comments/content
    labels = [
        "Show more",
        "View more",
        "View all",
        "Expand",
        "See more",
        "More",
        "View all updates",
    ]
    for label in labels:
        try:
            elems = driver.find_elements(By.XPATH, f"//button[normalize-space()='{label}'] | //span[normalize-space()='{label}']/ancestor::button")
            for el in elems:
                if el.is_displayed() and el.is_enabled():
                    el.click()
                    time.sleep(0.25)
        except Exception:
            continue


def auto_scroll(driver: webdriver.Chrome, max_rounds: int = 20) -> None:
    last_height = driver.execute_script("return document.body.scrollHeight || document.documentElement.scrollHeight")
    for _ in range(max_rounds):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(0.6)
        new_height = driver.execute_script("return document.body.scrollHeight || document.documentElement.scrollHeight")
        if not new_height or new_height == last_height:
            break
        last_height = new_height


def is_empty_comment(comment: Dict[str, Any]) -> bool:
    content = (comment.get("content") or "").strip()
    if not content:
        return True
    lowered = content.lower()
    deletion_markers = [
        "comment deleted",
        "this comment has been deleted",
        "deleted comment",
        "no content",
        "empty update",
    ]
    if any(marker in lowered for marker in deletion_markers):
        return True
    if not (comment.get("user") or "").strip() and not (comment.get("datetime") or "").strip():
        return True
    if re.fullmatch(r"#[0-9]+", content):
        return True
    if content.lower() in {"comment", "comments"}:
        return True
    return False


def extract_comments_dom(driver: webdriver.Chrome) -> List[Dict[str, Any]]:
    # Run a single JS snippet to gather best-effort comment data from various page structures.
    script = r"""
const results = [];

// Helper: get visible innerText trimmed
const text = (el) => (el && el.innerText ? el.innerText.trim() : "");

// Helper: find closest container that looks like a comment block
const closestCommentContainer = (start) => {
  if (!start) return null;
  let el = start;
  const seen = new Set();
  const isCommentLike = (node) => {
    if (!node || !node.tagName) return false;
    const tag = node.tagName.toLowerCase();
    if (["issue-comment", "activity-comment", "comment"].includes(tag)) return true;
    const cls = (node.getAttribute("class") || "").toLowerCase();
    // Only consider nodes with explicit comment markers; avoid generic activity items
    return /\bcomment\b/.test(cls) || /\bbv2-comment\b/.test(cls);
  };
  while (el && !isCommentLike(el) && !seen.has(el)) {
    seen.add(el);
    el = el.parentElement;
  }
  return el || start;
};

// Strategy A: anchors/elements with id like commentN
const anchors = Array.from(document.querySelectorAll('[id^="comment"], a[href^="#comment"], [data-comment-id]'))
  .filter(n => {
    const id = (n.id || "") + "";
    if (/^comment\d+$/.test(id)) return true;
    if (n.getAttribute && n.getAttribute('data-comment-id')) return true;
    const href = (n.getAttribute && n.getAttribute('href')) ? n.getAttribute('href') : "";
    return /^#comment\d+$/.test(href);
  });

// Strategy B: generic comment-like containers
const commentLike = Array.from(document.querySelectorAll('issue-comment, .bv2-comment, .comment, [class*="comment" i]'));

const candidates = new Set([...anchors.map(a => closestCommentContainer(a)), ...commentLike]);

let idx = 0;
for (const node of candidates) {
  if (!node) continue;
  // Try to determine comment id/number
  let cid = node.getAttribute('id') || '';
  let cnum = '';
  if (/^comment\d+$/.test(cid)) {
    cnum = '#' + cid.replace('comment','');
  } else {
    const a = node.querySelector('[id^="comment"], a[href^="#comment"]');
    if (a) {
      const aid = a.id || '';
      if (/^comment\d+$/.test(aid)) {
        cid = aid;
        cnum = '#' + aid.replace('comment','');
      } else {
        const href = a.getAttribute('href') || '';
        const m = href.match(/#comment(\d+)/);
        if (m) { cid = 'comment' + m[1]; cnum = '#' + m[1]; }
      }
    }
  }
  if (!cid) cid = 'comment' + (++idx);
  if (!cnum) cnum = '#' + (idx);

  // User/author
  let user = '';
  let userEmail = '';
  const userCand = node.querySelector('[class*="author" i], [class*="user" i], .bv2-comment-author, a[href*="profiles.google.com"], a[href^="mailto:"]');
  if (userCand) {
    user = text(userCand);
    if (userCand.getAttribute) {
      const href = userCand.getAttribute('href') || '';
      if (href.startsWith('mailto:')) {
        userEmail = href.replace(/^mailto:/, '').trim();
      }
    }
  }
  if (!user) {
    // Try to find something that looks like an email near the top of the comment
    const emailMatch = (text(node) || '').match(/[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}/);
    if (emailMatch) {
      user = emailMatch[0];
      userEmail = emailMatch[0];
    }
  }

  // Datetime
  let dt = '';
  let dtFull = '';
  const timeEl = node.querySelector('time, [title*=":" i], [class*="time" i], [class*="date" i]');
  if (timeEl) {
    dt = text(timeEl) || (timeEl.getAttribute('aria-label') || '');
    dtFull = timeEl.getAttribute('title') || dt;
  }

  // Content
  let content = '';
  const contentEl = node.querySelector('.comment-content, .bv2-comment-content, [class*="content" i]');
  if (contentEl) {
    content = text(contentEl);
  } else {
    // Fallback: text minus header-like bits
    content = text(node);
  }

  // Trim long whitespace
  const squash = s => (s || '').replace(/\s+$/g, '').replace(/^\s+/g, '').replace(/\s{2,}/g, ' ').trim();

  results.push({
    comment_id: cid,
    comment_number: cnum,
    user: squash(user),
    user_email: squash(userEmail),
    datetime: squash(dt),
    datetime_full: squash(dtFull),
    content: (content || '').trim()
  });
}

// Deduplicate by comment_id while preserving order
const seen = new Set();
const deduped = [];
for (const r of results) {
  if (r.comment_id && !seen.has(r.comment_id)) { seen.add(r.comment_id); deduped.push(r); }
}
return deduped;
"""

    try:
        items = driver.execute_script(script)
        if not isinstance(items, list):
            return []
        # Basic normalization
        normalized = []
        for it in items:
            if not isinstance(it, dict):
                continue
            c = {
                "comment_id": str(it.get("comment_id", "")).strip(),
                "comment_number": str(it.get("comment_number", "")).strip(),
                "user": str(it.get("user", "")).strip(),
                "user_email": str(it.get("user_email", "")).strip(),
                "datetime": str(it.get("datetime", "")).strip(),
                "datetime_full": str(it.get("datetime_full", "")).strip(),
                "content": str(it.get("content", "")).strip(),
            }
            normalized.append(c)
        return normalized
    except Exception:
        return []


def infer_issue_id(url: str) -> str:
    m = re.search(r"/issues/(\d+)", url)
    return m.group(1) if m else "issue"


def build_issue_url(value: str) -> str:
    value = value.strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"{BASE_ISSUE_URL}{value}"


def prompt_for_login(driver: webdriver.Chrome, headless: bool) -> None:
    if headless:
        print("Headless mode enabled; skipping manual login prompt.")
        return
    print("Navigating to login page. Complete any required authentication in the browser window.")
    driver.get(LOGIN_BASE_URL)
    time.sleep(2)
    input("Press Enter after completing login (if required) to continue...")


def format_comment_entry(comment: Dict[str, Any], description_comment: Optional[Dict[str, Any]]) -> Optional[str]:
    number = (comment.get("comment_number") or "").strip().lstrip("#")
    if description_comment:
        desc_number = (description_comment.get("comment_number") or "").strip().lstrip("#")
        if number == desc_number:
            return None

    comment_text = comment.get("content", "")
    comment_number = comment.get("comment_number", "")
    label = comment_number or (f"#{number}" if number else "")
    comment_body = comment_text.strip()
    entry_parts: List[str] = []
    entry_parts.append(f"Comment{label}" if label else "Comment")
    if comment_body:
        entry_parts.append(comment_body)
    entry = "\n".join(entry_parts)
    return f"\n\n{entry}" if entry else None


def scrape_issue(
    driver: webdriver.Chrome,
    url: str,
    wait_time: int,
    include_empty: bool,
    verbose: bool = True,
) -> Dict[str, Any]:
    issue_id = infer_issue_id(url)
    if verbose:
        print(f"\nStarting extraction from {url} (issue {issue_id})...")

    driver.get(url)
    wait_for_page_ready(driver, wait_time)

    try_click_more_like_buttons(driver)
    auto_scroll(driver)

    for tab_label in ("All updates", "Comments", "Updates"):
        try:
            tab = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable((By.XPATH, f"//*[normalize-space()='{tab_label}']"))
            )
            tab.click()
            time.sleep(0.4)
        except Exception:
            continue

    comments_all = extract_comments_dom(driver)
    comments_filtered = comments_all if include_empty else [c for c in comments_all if not is_empty_comment(c)]

    if verbose:
        extra = "" if len(comments_filtered) == len(comments_all) else f" ({len(comments_all)} including empty placeholders)"
        print(f"Found {len(comments_filtered)} comments{extra}")

    description_comment = next(
        (c for c in comments_all if (c.get("comment_number") or "").strip().lstrip("#") == "1"),
        None,
    )
    if not description_comment and comments_filtered:
        description_comment = comments_filtered[0]

    description_text = ""
    if description_comment:
        description_text = description_comment.get("content", "")

    comment_entries: List[str] = []
    for comment in comments_filtered:
        entry = format_comment_entry(comment, description_comment)
        if entry:
            comment_entries.append(entry)

    comments_combined = "".join(comment_entries).lstrip()
    if not comments_combined:
        comments_combined = "No comments"

    if verbose and comments_filtered:
        print("\n================================================================================")
        print("EXTRACTED COMMENTS")
        print("================================================================================\n")
        for i, comment in enumerate(comments_filtered, 1):
            print(f"--- Comment {i} ---")
            print(f"Comment Number: {comment.get('comment_number','')}")
            print(f"Comment ID: {comment.get('comment_id','')}")
            print(f"User: {comment.get('user','')}")
            if comment.get("user_email"):
                print(f"User Email: {comment.get('user_email')}")
            print(f"Date/Time: {comment.get('datetime','')}")
            if comment.get("datetime_full") and comment.get("datetime_full") != comment.get("datetime"):
                print(f"Full Date/Time: {comment.get('datetime_full')}")
            content_preview = (comment.get("content") or "").splitlines()
            joined = " ".join(line.strip() for line in content_preview)
            print(f"Content Preview: {joined[:200]}")
            print("-" * 80 + "\n")

    return {
        "issue_id": issue_id,
        "url": url,
        "description": description_text,
        "comments": comments_combined,
        "description_comment": description_comment,
        "comments_filtered": comments_filtered,
        "comments_all": comments_all,
    }


def process_single_issue(args: argparse.Namespace, url: str) -> None:
    driver = build_driver(headless=args.headless)
    try:
        if args.prompt_login:
            prompt_for_login(driver, args.headless)

        result = scrape_issue(
            driver=driver,
            url=url,
            wait_time=args.wait_time,
            include_empty=args.include_empty,
            verbose=not args.quiet,
        )

        out_path = Path(args.output) if args.output else None
        if out_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path(f"comments_{result['issue_id']}_{ts}.csv")
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".csv")

        row = {
            "ISSUE_ID": str(result["issue_id"]),
            "description": result["description"],
            "comments": result["comments"],
        }
        df = pd.DataFrame([row], columns=["ISSUE_ID", "description", "comments"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        if not args.quiet:
            print(f"Saved dataframe with {len(df)} row to {out_path}")
            if result["description_comment"]:
                print("Description captured from comment #1.")
            else:
                print("No comment #1 found; description column left blank.")

    except TimeoutException:
        print(
            "Timed out waiting for page to load. Try increasing --wait-time or disable --headless to log in.",
            file=sys.stderr,
        )
        sys.exit(1)
    except WebDriverException as e:
        print(f"WebDriver error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def process_csv(args: argparse.Namespace) -> None:
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: File '{args.csv}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty.")
        sys.exit(1)

    print(f"CSV loaded successfully. Shape: {df.shape}")
    print("Available columns:", df.columns.tolist())

    if "ISSUE_ID" not in df.columns:
        print("Error: 'ISSUE_ID' column not found in CSV file.")
        sys.exit(1)

    df_copy = df.copy()
    if "description" not in df_copy.columns:
        df_copy["description"] = ""
    if "comments" not in df_copy.columns:
        df_copy["comments"] = ""

    issue_ids = df_copy["ISSUE_ID"].astype(str).tolist()
    print(f"Found {len(issue_ids)} issue IDs")
    print("First 5 issue IDs:", issue_ids[:5])

    driver = build_driver(headless=args.headless)

    results: List[Dict[str, Any]] = []
    try:
        if args.prompt_login or not args.headless:
            prompt_for_login(driver, args.headless)

        for index, issue_id in enumerate(issue_ids, 1):
            url = build_issue_url(issue_id)
            print(f"\nProcessing issue {index}/{len(issue_ids)}: {issue_id}")
            result = scrape_issue(
                driver=driver,
                url=url,
                wait_time=args.wait_time,
                include_empty=args.include_empty,
                verbose=not args.quiet,
            )
            results.append(result)

    except TimeoutException:
        print(
            "Timed out waiting for page to load. Try increasing --wait-time or disable --headless to log in.",
            file=sys.stderr,
        )
        sys.exit(1)
    except WebDriverException as e:
        print(f"WebDriver error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    print("\nUpdating dataframe with results...")
    for result in results:
        issue_id = result["issue_id"]
        idx = df_copy[df_copy["ISSUE_ID"].astype(str) == str(issue_id)].index
        if len(idx) > 0:
            row_index = idx[0]
            df_copy.loc[row_index, "description"] = result["description"]
            df_copy.loc[row_index, "comments"] = result["comments"]
            if not args.quiet:
                has_comments = result["comments"] != "No comments"
                print(f"Updated issue {issue_id} with {'comments' if has_comments else 'no comments'}")
        else:
            print(f"Warning: Issue ID {issue_id} not found in original dataframe")

    if args.output:
        out_path = Path(args.output)
    else:
        source = Path(args.csv)
        out_path = source.with_name(f"{source.stem}_updated.csv")
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".csv")

    final_df = pd.DataFrame({
        "ISSUE_ID": df_copy["ISSUE_ID"].astype(str),
        "description": df_copy["description"],
        "comments": df_copy["comments"],
    })

    final_df.to_csv(out_path, index=False)
    print(f"\nUpdated CSV saved to: {out_path}")

    total_issues = len(results)
    issues_with_comments = sum(1 for r in results if r["comments"] != "No comments")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total issues processed: {total_issues}")
    print(f"Issues with comments: {issues_with_comments}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract comments from Google Issue Tracker using Selenium")
    parser.add_argument("--url", help="Full issue URL to scrape")
    parser.add_argument("--issue-id", help="Issue ID appended to the base URL to form the target URL")
    parser.add_argument("--csv", help="Path to CSV file containing an ISSUE_ID column for batch processing")
    parser.add_argument("--output", default=None, help="Output CSV filename (default: auto)")
    parser.add_argument("--headless", action="store_true", help="Run Chrome headless")
    parser.add_argument("--include-empty", action="store_true", help="Include empty/deleted placeholder comments")
    parser.add_argument("--wait-time", type=int, default=12, help="Max wait time for elements (default: %(default)s)")
    parser.add_argument("--prompt-login", action="store_true", help="Prompt for manual login before scraping")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output during scraping")

    args = parser.parse_args()

    if args.csv:
        process_csv(args)
        return

    target_url: Optional[str] = None
    if args.url:
        target_url = args.url
    elif args.issue_id:
        target_url = build_issue_url(str(args.issue_id))

    if not target_url:
        print("Error: provide either --url or --issue-id (or use --csv for batch mode).", file=sys.stderr)
        sys.exit(1)

    process_single_issue(args, target_url)


if __name__ == "__main__":
    main()
