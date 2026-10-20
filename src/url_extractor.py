"""
CAGE-Fake: URL Content Extraction Module
Uses Trafilatura for extracting clean title and main article body from web pages.
Ensures extraction failure aborts cleanly without making false predictions.
"""

from typing import Dict, Any, Optional
import re
from urllib.parse import urlparse
import requests
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    trafilatura = None
    HAS_TRAFILATURA = False

from src.utils import setup_logger

logger = setup_logger("url_extractor")

class ArticleExtractionError(Exception):
    """Raised when an article URL cannot be fetched or parsed."""
    pass

def validate_url(url: str) -> bool:
    """Validates basic URL syntax."""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme in ("http", "https") and parsed.netloc)
    except Exception:
        return False

def extract_article_from_url(url: str, timeout: int = 15) -> Dict[str, Any]:
    """
    Extracts article title and clean text content from a given URL.
    Returns:
        {
            "url": str,
            "title": str,
            "text": str,
            "domain": str,
            "author": Optional[str],
            "date": Optional[str],
            "word_count": int,
        }
    Raises ArticleExtractionError if extraction fails or content is insufficient.
    """
    url = str(url or "").strip()
    if not validate_url(url):
        raise ArticleExtractionError(f"Invalid URL provided: '{url}'. Please provide a valid http:// or https:// link.")

    if not HAS_TRAFILATURA or trafilatura is None:
        raise ArticleExtractionError(
            "The 'trafilatura' package is required for URL extraction. "
            "Please install it with: pip install trafilatura"
        )

    parsed = urlparse(url)
    domain = parsed.netloc

    logger.info(f"Fetching URL: {url} (domain: {domain})...")

    downloaded = None
    try:
        # Try trafilatura fetch
        downloaded = trafilatura.fetch_url(url)
    except Exception as e:
        logger.warning(f"Trafilatura fetch failed ({e}). Attempting fallback with custom headers...")

    # Fallback to requests if trafilatura fetch returned empty
    if not downloaded:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                downloaded = resp.text
            else:
                raise ArticleExtractionError(
                    f"HTTP Error {resp.status_code} while fetching '{url}'. The web server refused access or page is missing."
                )
        except requests.RequestException as e:
            raise ArticleExtractionError(f"Network error while connecting to '{url}': {e}")

    if not downloaded:
        raise ArticleExtractionError(f"Could not retrieve any content from '{url}'. The site may be unreachable, blocking bots, or paywalled.")

    # Extract text with Trafilatura
    try:
        extracted_text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            output_format="txt",
        )
        metadata = trafilatura.extract_metadata(downloaded)
    except Exception as e:
        raise ArticleExtractionError(f"Parsing error during article content extraction: {e}")

    if not extracted_text or len(extracted_text.strip()) < 50:
        raise ArticleExtractionError(
            f"Extraction failed for '{url}': Extracted content is too short or empty (< 50 characters). "
            f"The site may require JavaScript rendering, a subscription login, or anti-scraping CAPTCHA."
        )

    title = ""
    author = None
    date = None

    if metadata:
        title = metadata.title or ""
        author = metadata.author
        date = metadata.date

    # Fallback title from first line if title is empty
    if not title:
        first_line = extracted_text.strip().split("\n")[0]
        if len(first_line) < 120:
            title = first_line
        else:
            title = f"Article from {domain}"

    word_count = len(extracted_text.split())

    logger.info(f"Successfully extracted article '{title}' ({word_count} words) from {domain}")

    return {
        "url": url,
        "title": title.strip(),
        "text": extracted_text.strip(),
        "domain": domain,
        "author": author,
        "date": date,
        "word_count": word_count,
    }
