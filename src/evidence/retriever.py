"""
CAGE-Fake: Multi-Source Evidence Retriever with Disk Caching and Deterministic NER Queries
Primary retrieval: Tavily Search API (news-focused, returns raw article content).
Fallback sources: DuckDuckGo news/web search, Google Fact Check Tools API, Wikipedia.
All responses and scraped pages are cached on disk to avoid redundant API calls.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import hashlib
import time
import re
import requests
import trafilatura
from duckduckgo_search import DDGS

from src.utils import setup_logger
from src.evidence.domain_whitelist import extract_domain, is_trusted_domain

logger = setup_logger("retriever")

CACHE_FILE = Path("outputs/cache/evidence_retrieval_cache.json")

# Import Tavily config (key + params) from central config
try:
    from config import TAVILY_API_KEY, TAVILY_MAX_RESULTS, TAVILY_INCLUDE_RAW_CONTENT
except ImportError:
    TAVILY_API_KEY = ""
    TAVILY_MAX_RESULTS = 5
    TAVILY_INCLUDE_RAW_CONTENT = True


# High-signal domain keywords in crime, legal, court, and statutory reporting
LEGAL_CRIME_KEYWORDS = {
    "pocso", "court", "judge", "sentenced", "imprisonment", "life",
    "convicted", "verdict", "acquitted", "rape", "assault", "murder",
    "fir", "police", "arrested", "custody", "bail", "trial", "minor",
    "special", "sessions", "district", "allahabad", "delhi", "supreme",
    "agra", "kasganj", "up", "uttar", "pradesh",
}


class EvidenceRetriever:
    """
    Retrieves candidate corroborating or debunking web pages for a factual claim.

    Sources (in priority order):
    1. Tavily Search API  — news-focused, returns raw article content, highly reliable.
    2. DuckDuckGo News & Web Search — live web news fallback with retry.
    3. Google Fact Check Tools API — authoritative debunks and fact checks.
    4. Wikipedia Search API — encyclopedic background context.

    All responses and scraped pages are cached on disk to avoid redundant API calls.
    """

    def __init__(self, cache_file: Path = CACHE_FILE, max_results_per_source: int = 4):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.max_results = max_results_per_source
        self.cache: Dict[str, Any] = self._load_cache()

        # Initialise Tavily client once (lazy — only if the package is available)
        self._tavily_client = None
        if TAVILY_API_KEY:
            try:
                from tavily import TavilyClient  # type: ignore
                self._tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
                logger.info("Tavily Search API client initialised (primary retrieval source).")
            except ImportError:
                logger.warning(
                    "tavily-python package not installed. "
                    "Install it with: pip install tavily-python>=0.3.0. "
                    "Falling back to DuckDuckGo only."
                )
            except Exception as e:
                logger.warning(f"Tavily client initialisation failed: {e}")

    def _load_cache(self) -> Dict[str, Any]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load retrieval cache: {e}")
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save retrieval cache: {e}")

    def _get_cache_key(self, query: str, source: str) -> str:
        h = hashlib.sha256(f"{source}:{query.strip().lower()}".encode("utf-8")).hexdigest()
        return h

    @staticmethod
    def generate_targeted_query(claim: str) -> str:
        """
        Formulates a deterministic, high-precision search query from a claim by
        extracting core named entities (Persons, Courts, Cities, Statutes like POCSO/Agra),
        crime/legal keywords, and critical numbers.
        Example: 'Agra Pocso court life imprisonment Sonika Chaudhary'
        """
        clean = re.sub(r"[\"'\(\)\[\]]", " ", claim).strip()
        entities: List[str] = []

        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
            except Exception:
                nlp = spacy.blank("en")
            doc = nlp(clean)
            if hasattr(doc, "ents"):
                for ent in doc.ents:
                    if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "NORP", "DATE", "CARDINAL", "LAW"):
                        txt = ent.text.strip()
                        if len(txt) > 2 and txt.lower() not in [e.lower() for e in entities]:
                            entities.append(txt)
        except Exception:
            pass

        # Extract capitalized words (names, places) if spaCy missed any
        words = clean.split()
        cap_words = [
            w.strip(".,;:?!")
            for w in words
            if len(w) > 2 and w[0].isupper() and not w.isupper()
        ]
        for cw in cap_words:
            if cw.lower() not in [e.lower() for e in entities] and cw.lower() not in {
                "the", "and", "for", "with", "from", "that", "this"
            }:
                entities.append(cw)

        # Extract domain terms (POCSO, court, rape, sentenced, etc.)
        domain_terms = [
            w.strip(".,;:?!")
            for w in words
            if w.lower().strip(".,;:?!") in LEGAL_CRIME_KEYWORDS
        ]

        # Combine entities and domain terms
        combined = []
        seen = set()
        for item in entities + domain_terms:
            item_clean = item.strip()
            item_lower = item_clean.lower()
            if item_lower not in seen and len(item_clean) > 1:
                seen.add(item_lower)
                combined.append(item_clean)

        if len(combined) >= 2:
            query = " ".join(combined[:8])
        else:
            # Fallback to first 12 non-stopword tokens
            query = " ".join([
                w for w in words[:12]
                if w.lower() not in {"in", "an", "the", "a", "of", "to", "was", "were", "is", "at", "by", "on"}
            ])

        return query.strip()

    def retrieve_for_claim(self, claim: str) -> List[Dict[str, Any]]:
        """
        Retrieves relevant pages/snippets from Tavily, DuckDuckGo, FactCheck API, and Wikipedia.
        Uses deterministic NER-targeted query generation as primary with fallback to verbatim claim.
        Returns a list of dicts:
        [{
            'title': ...,
            'url': ...,
            'domain': ...,
            'source_type': 'Tavily' | 'DuckDuckGo' | 'GoogleFactCheck' | 'Wikipedia',
            'is_trusted': bool,
            'content': ... (full article text or snippet)
        }]
        """
        clean_query = claim.strip()
        targeted_query = self.generate_targeted_query(clean_query)

        # Shortened fallback query if verbatim sentence is long
        words = clean_query.split()
        verbatim_query = " ".join(words[:16]) if len(words) > 16 else clean_query

        results = []

        # 1. Tavily Search (primary — news-focused, raw article content)
        # First query with high-precision targeted NER query
        tavily_results = self._search_tavily(targeted_query)
        if not tavily_results and targeted_query != verbatim_query:
            # Retry with verbatim query
            tavily_results = self._search_tavily(verbatim_query)
        results.extend(tavily_results)

        # 2. DuckDuckGo Web Search & News (fallback / supplement)
        ddg_results = self._search_duckduckgo(targeted_query)
        if not ddg_results and targeted_query != verbatim_query:
            ddg_results = self._search_duckduckgo(verbatim_query)
        results.extend(ddg_results)

        # 3. Google Fact Check Tools API
        fc_results = self._search_google_fact_check(targeted_query)
        results.extend(fc_results)

        # 4. Wikipedia Search
        wiki_results = self._search_wikipedia(targeted_query)
        results.extend(wiki_results)

        # Scrape and enrich with Trafilatura for items that have only short snippets.
        # Tavily already returns raw_content, so skip those to avoid wasting time.
        enriched_results = []
        for item in results:
            url = item.get("url", "")
            if url and len(item.get("content", "")) < 300 and item.get("source_type") != "Tavily":
                scraped = self._scrape_url(url)
                if scraped:
                    item["content"] = scraped
            enriched_results.append(item)

        return enriched_results

    def _search_tavily(self, query: str) -> List[Dict[str, Any]]:
        """
        Searches via Tavily Search API with advanced depth and news topic focus.
        Returns raw article content for each result when available.
        """
        if self._tavily_client is None:
            return []

        key = self._get_cache_key(query, "tavily")
        if key in self.cache:
            return self.cache[key]

        items = []
        for attempt in range(2):
            try:
                response = self._tavily_client.search(
                    query=query,
                    search_depth="advanced",          # deep search for richer evidence
                    topic="news",                      # news-focused results
                    max_results=TAVILY_MAX_RESULTS,
                    include_raw_content=TAVILY_INCLUDE_RAW_CONTENT,
                    include_answer=False,
                )
                for r in response.get("results", []):
                    url = r.get("url", "")
                    domain = extract_domain(url)
                    content = r.get("raw_content") or r.get("content", "")
                    if content:
                        content = str(content)[:4000]
                    items.append({
                        "title": r.get("title", ""),
                        "url": url,
                        "domain": domain,
                        "source_type": "Tavily",
                        "is_trusted": is_trusted_domain(domain),
                        "content": content,
                        "tavily_score": r.get("score", 0.0),
                    })
                logger.debug(f"Tavily returned {len(items)} results for: {query[:60]}")
                break
            except Exception as e:
                logger.debug(f"Tavily search attempt {attempt+1} error for query '{query}': {e}")
                time.sleep(0.5)

        self.cache[key] = items
        self._save_cache()
        return items

    def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        key = self._get_cache_key(query, "ddg")
        if key in self.cache:
            return self.cache[key]

        items = []
        # Attempt 1: DuckDuckGo news search (news-focused, high editorial corroboration)
        try:
            with DDGS(timeout=8) as ddgs:
                try:
                    news_gen = ddgs.news(query, max_results=self.max_results)
                    for r in news_gen:
                        url = r.get("url", "")
                        domain = extract_domain(url)
                        items.append({
                            "title": r.get("title", ""),
                            "url": url,
                            "domain": domain,
                            "source_type": "DuckDuckGo News",
                            "is_trusted": is_trusted_domain(domain),
                            "content": r.get("body", ""),
                        })
                except Exception as ne:
                    logger.debug(f"DuckDuckGo news search failed: {ne}")

                # If news search didn't yield enough, supplement with general web text search
                if len(items) < 2:
                    ddg_gen = ddgs.text(query, max_results=self.max_results)
                    for r in ddg_gen:
                        url = r.get("href", "")
                        domain = extract_domain(url)
                        items.append({
                            "title": r.get("title", ""),
                            "url": url,
                            "domain": domain,
                            "source_type": "DuckDuckGo Web Search",
                            "is_trusted": is_trusted_domain(domain),
                            "content": r.get("body", ""),
                        })
        except Exception as e:
            logger.debug(f"DuckDuckGo search error for query '{query}': {e}")

        self.cache[key] = items
        self._save_cache()
        return items

    def _search_google_fact_check(self, query: str) -> List[Dict[str, Any]]:
        key = self._get_cache_key(query, "google_factcheck")
        if key in self.cache:
            return self.cache[key]

        items = []
        try:
            api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            resp = requests.get(api_url, params={"query": query, "languageCode": "en"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for c in data.get("claims", [])[:self.max_results]:
                    for review in c.get("claimReview", []):
                        url = review.get("url", "")
                        domain = extract_domain(url)
                        pub = review.get("publisher", {}).get("name", "Fact Checker")
                        rating = review.get("textualRating", "Unrated")
                        title = c.get("text", "")
                        summary = f"Claim: {title}. Rating: {rating}. Review by {pub}."
                        items.append({
                            "title": title,
                            "url": url,
                            "domain": domain,
                            "source_type": "Google Fact Check Tools",
                            "is_trusted": True,
                            "content": summary,
                            "factcheck_rating": rating,
                        })
        except Exception as e:
            logger.debug(f"Google FactCheck API search error: {e}")

        self.cache[key] = items
        self._save_cache()
        return items

    def _search_wikipedia(self, query: str) -> List[Dict[str, Any]]:
        key = self._get_cache_key(query, "wikipedia")
        if key in self.cache:
            return self.cache[key]

        items = []
        try:
            endpoint = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 2,
            }
            resp = requests.get(endpoint, params=params, headers={"User-Agent": "CAGE-Fake/2.0"}, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                for s in data.get("query", {}).get("search", []):
                    title = s.get("title", "")
                    page_id = s.get("pageid")
                    clean_snippet = s.get("snippet", "").replace('<span class="searchmatch">', "").replace('</span>', "")
                    url = f"https://en.wikipedia.org/?curid={page_id}"
                    items.append({
                        "title": f"Wikipedia: {title}",
                        "url": url,
                        "domain": "en.wikipedia.org",
                        "source_type": "Wikipedia Reference",
                        "is_trusted": True,
                        "content": clean_snippet,
                    })
        except Exception as e:
            logger.debug(f"Wikipedia search error: {e}")

        self.cache[key] = items
        self._save_cache()
        return items

    def _scrape_url(self, url: str) -> str:
        key = self._get_cache_key(url, "scrape")
        if key in self.cache:
            return self.cache[key]

        try:
            resp = requests.get(url, timeout=4, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            if resp.status_code == 200:
                text = trafilatura.extract(resp.text, include_comments=False)
                if text and len(text.strip()) > 100:
                    clean = text[:4000]
                    self.cache[key] = clean
                    self._save_cache()
                    return clean
        except Exception as e:
            logger.debug(f"Failed to scrape {url}: {e}")

        self.cache[key] = ""
        self._save_cache()
        return ""
