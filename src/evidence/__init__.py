"""
CAGE-Fake Evidence Grounding Package
"""

from src.evidence.domain_whitelist import is_trusted_domain, extract_domain, TRUSTED_DOMAINS
from src.evidence.claim_extractor import ClaimExtractor
from src.evidence.retriever import EvidenceRetriever
from src.evidence.ranker import HybridEvidenceRanker
from src.evidence.nli_verifier import NLIVerifier
from src.evidence.entity_matcher import EntityNumberMatcher
from src.evidence.evidence_pipeline import EvidencePipeline

__all__ = [
    "is_trusted_domain",
    "extract_domain",
    "TRUSTED_DOMAINS",
    "ClaimExtractor",
    "EvidenceRetriever",
    "HybridEvidenceRanker",
    "NLIVerifier",
    "EntityNumberMatcher",
    "EvidencePipeline",
]
