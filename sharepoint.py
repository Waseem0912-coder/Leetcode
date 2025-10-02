import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

import pandas as pd


DEFAULT_URL = "https://issuetracker.google.com/issues/446543771"


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


def main():
    parser = argparse.ArgumentParser(description="Extract comments from Google Issue Tracker using Selenium")
    parser.add_argument("--url", default=DEFAULT_URL, help="Issue URL (default: %(default)s)")
    parser.add_argument("--output", default=None, help="Output CSV filename (default: auto)")
    parser.add_argument("--headless", action="store_true", help="Run Chrome headless")
    parser.add_argument("--include-empty", action="store_true", help="Include empty/deleted placeholder comments")
    parser.add_argument("--wait-time", type=int, default=12, help="Max wait time for elements (default: %(default)s)")

    args = parser.parse_args()
    url = args.url
    issue_id = infer_issue_id(url)

    print(f"Starting extraction from {url}...")
    driver = build_driver(headless=args.headless)
    try:
        print(f"Loading URL: {url}")
        driver.get(url)
        wait_for_page_ready(driver, args.wait_time)

        # Try to ensure comments are present on screen
        try_click_more_like_buttons(driver)
        auto_scroll(driver)

        # If there is a tab/filter that hides comments, try to click an updates/comments tab by label
        for tab_label in ("All updates", "Comments", "Updates"):
            try:
                tab = WebDriverWait(driver, 2).until(
                    EC.element_to_be_clickable((By.XPATH, f"//*[normalize-space()='{tab_label}']"))
                )
                tab.click()
                time.sleep(0.4)
            except Exception:
                pass

        # Extract
        comments_all = extract_comments_dom(driver)

        # Filter out empty placeholders unless explicitly requested
        def _is_empty(c: Dict[str, Any]) -> bool:
            content = (c.get("content") or "").strip()
            if not content:
                return True
            # Treat known deletion messages as empty
            lowered = content.lower()
            deletion_markers = [
                "comment deleted",
                "this comment has been deleted",
                "deleted comment",
                "no content",
                "empty update",
            ]
            if any(m in lowered for m in deletion_markers):
                return True

            # If there is no user and no datetime, likely a placeholder
            if not (c.get("user") or "").strip() and not (c.get("datetime") or "").strip():
                return True

            # Ignore comments that just repeat an anchor label (e.g., "#8") or generic word "Comment"
            if re.fullmatch(r"#[0-9]+", content):
                return True
            if content.lower() in {"comment", "comments"}:
                return True

            return False

        if args.include_empty:
            comments = comments_all
        else:
            comments = [c for c in comments_all if not _is_empty(c)]

        print(f"Found {len(comments)} comments", end="")
        if len(comments_all) != len(comments):
            print(f" ({len(comments_all)} including empty placeholders)")
        else:
            print()

        # Pretty print in console
        if comments:
            print("\n================================================================================")
            print("EXTRACTED COMMENTS")
            print("================================================================================\n")
            for i, c in enumerate(comments, 1):
                print(f"--- Comment {i} ---")
                print(f"Comment Number: {c.get('comment_number','')}")
                print(f"Comment ID: {c.get('comment_id','')}")
                print(f"User: {c.get('user','')}")
                if c.get("user_email"):
                    print(f"User Email: {c.get('user_email')}")
                print(f"Date/Time: {c.get('datetime','')}")
                if c.get("datetime_full") and c.get("datetime_full") != c.get("datetime"):
                    print(f"Full Date/Time: {c.get('datetime_full')}")
                content_preview = (c.get("content") or "").splitlines()
                joined = " ".join(line.strip() for line in content_preview)
                print(f"Content Preview: {joined[:200]}")
                print("-" * 80 + "\n")

        # Save to DataFrame (CSV)
        if args.output:
            out_path = Path(args.output)
            if out_path.suffix == "":
                out_path = out_path.with_suffix(".csv")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path(f"comments_{issue_id}_{ts}.csv")

        # Prepare description (comment #1) and arrange output rows
        description_comment = None
        for record in comments_all:
            number = (record.get("comment_number") or "").strip().lstrip("#")
            if number == "1":
                description_comment = record
                break
        if not description_comment and comments:
            description_comment = comments[0]
        description_text = ""
        if description_comment:
            description_text = description_comment.get("content", "")

        comment_entries: List[str] = []
        for c in comments:
            number = (c.get("comment_number") or "").strip().lstrip("#")
            if description_comment and number == "1":
                continue  # skip comment #1 in comments column
            comment_text = c.get("content", "")
            comment_number = c.get("comment_number", "")
            if comment_number:
                label = comment_number
            else:
                label = f"#{number}" if number else ""
            comment_body = comment_text.strip()
            entry_parts: List[str] = []
            if label:
                entry_parts.append(f"Comment{label}")
            else:
                entry_parts.append("Comment")
            if comment_body:
                entry_parts.append(comment_body)
            entry = "\n".join(entry_parts)
            comment_entries.append(f"\n\n{entry}" if entry else "")

        comments_combined = "".join(entry for entry in comment_entries if entry).lstrip()

        row = {
            "issue_id": issue_id,
            "description": description_text,
            "comments": comments_combined,
        }

        df = pd.DataFrame([row], columns=["issue_id", "description", "comments"])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved dataframe with {len(df)} rows to {out_path}")
        if description_comment:
            print("Description captured from comment #1.")
        else:
            print("No comment #1 found; description column left blank.")

    except TimeoutException:
        print("Timed out waiting for page to load. Try increasing --wait-time or disable --headless to log in.", file=sys.stderr)
        sys.exit(1)
    except WebDriverException as e:
        print(f"WebDriver error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
