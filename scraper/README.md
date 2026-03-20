# Web Scraper

Concurrent web scraper for extracting title, text, and categories from websites. Outputs JSONL files suitable for LLM training data.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python scraper.py <starting-url> [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--workers` | 3 | Max concurrent requests |
| `--delay` | 1.5 | Seconds between requests per worker |
| `--max-pages` | 0 | Max pages to scrape (0 = unlimited) |
| `--no-verify` | false | Disable SSL certificate verification |
| `--pattern` | `^/novell/\d+/[^/]+/?$` | Regex for URL paths to save |
| `--fresh` | false | Ignore saved state, start fresh |

### Examples

```bash
# Basic scrape
python scraper.py https://example.com

# Faster with 5 workers
python scraper.py https://example.com --workers 5

# Limit to 100 pages
python scraper.py https://example.com --max-pages 100

# Faster scraping, shorter delay
python scraper.py https://example.com --workers 8 --delay 0.5
```

## Output

Results are saved to `output/<domain>/`:
- `pages/` — Individual JSONL file per page
- `dataset.jsonl` — Consolidated file with all pages
- `state.json` — Crawl state for resuming

### Output format

```json
{"url": "https://...", "title": "...", "text": "...", "categories": ["..."]}
```

## Resuming

The scraper saves progress to `state.json` every 100 pages. If interrupted (Ctrl+C or crash), simply re-run the same command to resume where it left off.

Use `--fresh` to discard saved state and start over:
```bash
python scraper.py https://example.com --fresh
```

## Ctrl+C

Press Ctrl+C to stop crawling. Progress will be saved and `dataset.jsonl` will be written with all pages scraped so far.
