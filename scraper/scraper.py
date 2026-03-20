#!/usr/bin/env python3
"""Web scraper for extracting title, text, and categories from websites."""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def extract_page(url: str, html: str) -> dict | None:
    """Extract title, text content, and categories from HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Title — from <h1>
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""
    if not title:
        return None

    # Body — from <div class="text-story">
    content_div = soup.find("div", class_="text")
    if content_div is None:
        return None

    # Extract text, clean whitespace
    text = content_div.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if len(text) < 50:
        return None

    # Categories — format: <p><strong>Kategori:</strong> <a href="...">Category</a></p>
    # or: <p><strong>Kategori:</strong><a>First</a> och <a>Second</a></p>
    categories = []
    for p in soup.find_all("p"):
        strong = p.find("strong")
        if strong and "kategori" in strong.get_text(strip=True).lower():
            for a in p.find_all("a"):
                cat = a.get_text(strip=True)
                if cat:
                    categories.append(cat)
            break

    return {
        "url": url,
        "title": title,
        "text": text,
        "categories": categories,
    }


DEFAULT_URL_PATTERN = re.compile(r"^/\w+/\d+/\S+/?$")


def extract_links(url: str, html: str, base_domain: str) -> list[str]:
    """Extract same-domain links from HTML."""
    soup = BeautifulSoup(html, "lxml")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Skip anchors, javascript, mailto
        if href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        # Resolve relative URLs
        full_url = urljoin(url, href)

        # Strip fragment
        full_url = full_url.split("#")[0]

        # Check same domain
        parsed = urlparse(full_url)
        if parsed.netloc == base_domain and full_url not in links:
            links.append(full_url)

    return links


class Crawler:
    def __init__(
        self,
        start_url: str,
        workers: int = 3,
        delay: float = 1.5,
        max_pages: int = 0,
        verify_ssl: bool = True,
        url_pattern: re.Pattern | None = None,
        fresh: bool = False,
    ):
        self.start_url = start_url
        parsed = urlparse(start_url)
        self.base_domain = parsed.netloc
        self.scheme = parsed.scheme
        self.workers = workers
        self.delay = delay
        self.max_pages = max_pages
        self.url_pattern = url_pattern or DEFAULT_URL_PATTERN

        self.visited: set[str] = set()
        self.visited_lock = threading.Lock()
        self.queue: list[str] = [start_url]
        self.queue_lock = threading.Lock()

        self.results: list[dict] = []
        self.results_lock = threading.Lock()

        self.output_dir = os.path.join("output", self.base_domain)
        self.pages_dir = os.path.join(self.output_dir, "pages")
        os.makedirs(self.pages_dir, exist_ok=True)

        self.state_file = os.path.join(self.output_dir, "state.json")
        self.state_lock = threading.Lock()
        self.save_interval = 100  # Save state every N pages

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"
            }
        )
        self.session.verify = verify_ssl

        if not fresh:
            self._load_state()

    def _load_state(self):
        """Load visited URLs and queue from state file."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.visited = set(state.get("visited", []))
            # Only keep queued URLs that haven't been visited
            self.queue = [u for u in state.get("queue", []) if u not in self.visited]
            # Add start URL if not visited and not in queue
            if self.start_url not in self.visited and self.start_url not in self.queue:
                self.queue.insert(0, self.start_url)
            print(
                f"Resuming: {len(self.visited)} URLs already visited, {len(self.queue)} in queue"
            )
        except (json.JSONDecodeError, KeyError):
            print("Warning: Could not load state file, starting fresh")

    def _save_state(self):
        """Save visited URLs and queue to state file."""
        with self.state_lock:
            with self.visited_lock:
                visited_list = list(self.visited)
            with self.queue_lock:
                queue_list = list(self.queue)
            state = {
                "visited": visited_list,
                "queue": queue_list,
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f)

    def _should_visit(self, url: str) -> bool:
        with self.visited_lock:
            if url in self.visited:
                return False
            if self.max_pages and len(self.visited) >= self.max_pages:
                return False
            self.visited.add(url)
            return True

    def _get_next_url(self) -> str | None:
        with self.queue_lock:
            if self.queue:
                return self.queue.pop(0)
        return None

    def _add_urls(self, urls: list[str]):
        with self.queue_lock:
            for url in urls:
                if url not in self.visited and url not in self.queue:
                    self.queue.append(url)

    def _save_page(self, page_data: dict):
        slug = re.sub(
            r"[^a-z0-9]+", "-", urlparse(page_data["url"]).path.strip("/").lower()
        )
        if not slug:
            slug = "index"
        filepath = os.path.join(self.pages_dir, f"{slug}.jsonl")

        with self.results_lock:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json.dumps(page_data, ensure_ascii=False) + "\n")
            self.results.append(page_data)

    def _crawl_page(self, url: str) -> list[str]:
        """Crawl a single page. Returns list of new links found."""
        time.sleep(self.delay)

        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [ERROR] {url}: {e}")
            return []

        if "text/html" not in resp.headers.get("Content-Type", ""):
            return []

        html = resp.text
        path = urlparse(url).path

        # Always extract links for crawling
        links = extract_links(url, html, self.base_domain)

        # Only save pages matching the URL pattern
        if self.url_pattern.match(path):
            page_data = extract_page(url, html)
            if page_data:
                self._save_page(page_data)
                print(f"  [OK] {url} — {page_data['title'][:60]}")
            else:
                print(f"  [SKIP] {url} — no extractable content")
        else:
            print(f"  [CRAWL] {url} — following links")

        return links

    def crawl(self):
        print(f"Crawling {self.start_url}")
        print(
            f"Domain: {self.base_domain} | Workers: {self.workers} | Delay: {self.delay}s"
        )
        if self.max_pages:
            print(f"Max pages: {self.max_pages}")
        print()

        active_workers = 0
        active_lock = threading.Lock()
        pages_since_save = 0
        pages_since_save_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            while True:
                # Check if we should stop
                url = self._get_next_url()
                if url is None:
                    # Wait for active workers to finish
                    with active_lock:
                        if active_workers == 0:
                            break
                    time.sleep(0.1)
                    continue

                if not self._should_visit(url):
                    continue

                with active_lock:
                    active_workers += 1

                future = executor.submit(self._crawl_page, url)

                def _done(fut, _url=url):
                    nonlocal active_workers, pages_since_save
                    try:
                        new_links = fut.result()
                        self._add_urls(new_links)
                        should_save = False
                        with pages_since_save_lock:
                            pages_since_save += 1
                            if pages_since_save >= self.save_interval:
                                should_save = True
                                pages_since_save = 0
                        if should_save:
                            self._save_state()
                    except Exception as e:
                        print(f"  [ERROR] {_url}: {e}")
                    with active_lock:
                        active_workers -= 1

                future.add_done_callback(_done)

        self._save_state()
        self._write_dataset()

    def _write_dataset(self):
        dataset_path = os.path.join(self.output_dir, "dataset.jsonl")
        with open(dataset_path, "w", encoding="utf-8") as f:
            for page in self.results:
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
        print(f"\nDone! {len(self.results)} pages saved to {dataset_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape a website for LLM training data"
    )
    parser.add_argument("url", help="Starting URL to crawl")
    parser.add_argument(
        "--workers", type=int, default=3, help="Max concurrent requests (default: 3)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds between requests per worker (default: 1.5)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=0, help="Max pages to scrape (0 = unlimited)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable SSL certificate verification",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Regex pattern for URL paths to save (default: ^/novell/\\d+/[^/]+/?$)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore saved state and start fresh",
    )
    args = parser.parse_args()

    url_pattern = re.compile(args.pattern) if args.pattern else None

    crawler = Crawler(
        start_url=args.url,
        workers=args.workers,
        delay=args.delay,
        max_pages=args.max_pages,
        verify_ssl=not args.no_verify,
        url_pattern=url_pattern,
        fresh=args.fresh,
    )

    try:
        crawler.crawl()
    except KeyboardInterrupt:
        print("\nInterrupted! Saving progress...")
        crawler._save_state()
        crawler._write_dataset()


if __name__ == "__main__":
    main()
