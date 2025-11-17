import time
import random

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from sentiment_test import analyze_youtube_comments

VIDEO_URL = "https://www.youtube.com/watch?v=lcp0W6a_Omc"

# Speed-related parameters
SCROLL_PAUSE = 2.0        # Adjust for faster/slower scrolling
MAX_SCROLL_ROUNDS = 30
TARGET_COMMENTS = 200     # Stop scrolling after reaching this number (optional)


def setup_driver(headless: bool = False):
    """Set up the Chrome WebDriver."""
    options = webdriver.ChromeOptions()

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    # Load page faster (do not wait for all resources)
    options.page_load_strategy = "eager"

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    # Reduce implicit wait (faster element search)
    driver.implicitly_wait(2)
    return driver


def scroll_until_comments_appear(driver):
    """
    Scroll down gradually until comment threads become visible.
    """
    print("▶ Scrolling until comment section appears...")

    for i in range(8):
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(SCROLL_PAUSE)

        threads = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")
        print(f"  - Check {i+1}: Detected thread count = {len(threads)}")

        if len(threads) > 0:
            print("✅ Comment threads detected!")
            return True

    print("❌ Failed to detect comment section after multiple scroll attempts.")
    return False


def scroll_more_for_more_comments(driver):
    """
    Auto-scroll until no more comments are loading (infinite-scroll mode).
    """
    print("▶ Scrolling further to load more comments... (infinite mode)")

    last_height = 0

    while True:
        threads = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")
        current_count = len(threads)
        print(f"  - Current comment count: {current_count}")

        # Scroll to bottom
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(SCROLL_PAUSE)

        # Check if scroll height changed
        new_height = driver.execute_script("return document.documentElement.scrollHeight")

        if new_height == last_height:
            print("❌ Page height no longer increasing → Reached the bottom.")
            break

        last_height = new_height


def collect_comments(driver):
    """Collect comments from the current YouTube page and return a list of dicts."""
    comments_data = []

    has_comments = scroll_until_comments_appear(driver)
    if not has_comments:
        return comments_data

    scroll_more_for_more_comments(driver)

    threads = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")
    print(f"▶ Final collected comment threads: {len(threads)}")

    for t in threads:
        try:
            author_el = t.find_element(By.CSS_SELECTOR, "#author-text span")
            content_el = t.find_element(By.CSS_SELECTOR, "yt-attributed-string#content-text span")

            author = author_el.text.strip()
            text = content_el.text.strip()

            if not text:
                continue

            comments_data.append({
                "author": author,
                "text": text
            })
        except Exception:
            continue

    return comments_data


def pick_random_winner(comments_data, k: int = 1):
    """Pick k random winners from the comment list."""
    if not comments_data:
        print("❌ No comment data available. Please check crawling results.")
        return

    if k == 1:
        winner = random.choice(comments_data)
        print("\n===== 🎉 Winner (1 person) 🎉 =====")
        print("Author:", winner["author"])
        print("Comment:", winner["text"])
        print("===============================")
    else:
        if k > len(comments_data):
            k = len(comments_data)

        winners = random.sample(comments_data, k=k)

        print(f"\n===== 🎉 Winners ({k} people) 🎉 =====")
        for i, w in enumerate(winners, start=1):
            print(f"[{i}] Author: {w['author']}")
            print(f"     Comment: {w['text']}")
            print("--------------------------------")
        print("================================")


def main():
    driver = setup_driver(headless=False)

    try:
        print("▶ Opening YouTube page...")
        driver.get(VIDEO_URL)

        # Adjust initial loading delay depending on your network
        time.sleep(3)

        comments = collect_comments(driver)
        print(f"▶ Valid comments collected: {len(comments)}")

        # pick_random_winner(comments, k=1)
        analyze_youtube_comments(comments)

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
