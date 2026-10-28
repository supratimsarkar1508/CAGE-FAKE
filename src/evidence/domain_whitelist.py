"""
CAGE-Fake: Trusted Domain Whitelist & Reputation Scorer
Maintains authoritative newsroom and official government source whitelists.
"""

from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

# Authoritative high-credibility newsrooms, fact-checkers, and government domains
TRUSTED_DOMAINS = {
    # International Wire Services & Authoritative Press
    "reuters.com",
    "apnews.com",
    "afp.com",
    "bbc.com",
    "bbc.co.uk",
    "bloomberg.com",
    "aljazeera.com",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "ft.com",
    "nature.com",
    "science.org",
    "who.int",
    "un.org",
    "nih.gov",
    "cdc.gov",
    "nasa.gov",

    # Authoritative Indian National Press & Official Repositories
    "pib.gov.in",          # Press Information Bureau (Official Indian Gov)
    "thehindu.com",
    "indianexpress.com",
    "timesofindia.indiatimes.com",
    "hindustantimes.com",
    "ndtv.com",
    "thewire.in",
    "scroll.in",
    "livemint.com",
    "business-standard.com",
    "prsindia.org",
    "sci.gov.in",          # Supreme Court of India
    "main.sci.gov.in",
    "eci.gov.in",          # Election Commission of India

    # Fact-Checking Organizations (IFCN Signatories)
    "altnews.in",
    "boomlive.in",
    "thequint.com",
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
}


def extract_domain(url: str) -> str:
    """Extracts lowercase normalized root domain from a URL."""
    if not url:
        return ""
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def is_trusted_domain(url_or_domain: str) -> bool:
    """Checks whether URL or domain matches the trusted whitelist."""
    domain = extract_domain(url_or_domain) if "/" in url_or_domain else url_or_domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    # Exact or subdomain match
    for trusted in TRUSTED_DOMAINS:
        if domain == trusted or domain.endswith("." + trusted):
            return True
    return False


def get_domain_reputation(url_or_domain: str) -> Dict[str, Any]:
    """Returns domain reputation assessment and score."""
    domain = extract_domain(url_or_domain) if "/" in url_or_domain else url_or_domain.lower()
    trusted = is_trusted_domain(domain)
    return {
        "domain": domain,
        "is_trusted": trusted,
        "credibility_weight": 1.0 if trusted else 0.40,
        "source_category": "Authoritative / Whitelisted" if trusted else "General Web Source",
    }
