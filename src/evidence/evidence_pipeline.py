"""
CAGE-Fake: Unified Evidence Grounding Pipeline
Coordinates claim extraction, multi-source retrieval, hybrid BM25+dense ranking,
NLI inference, entity/numeric consistency verification, and trusted domain accounting.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from src.utils import setup_logger
from src.evidence.claim_extractor import ClaimExtractor
from src.evidence.retriever import EvidenceRetriever
from src.evidence.ranker import HybridEvidenceRanker
from src.evidence.nli_verifier import NLIVerifier
from src.evidence.entity_matcher import EntityNumberMatcher
from src.evidence.domain_whitelist import is_trusted_domain, extract_domain

logger = setup_logger("evidence_pipeline")


class EvidencePipeline:
    """
    End-to-end evidence grounding pipeline for article triage.
    """

    def __init__(self, use_live_retrieval: bool = True):
        self.use_live_retrieval = use_live_retrieval
        self.claim_extractor = ClaimExtractor()
        self.retriever = EvidenceRetriever()
        self.ranker = HybridEvidenceRanker()
        self.nli_verifier = NLIVerifier()
        self.entity_matcher = EntityNumberMatcher()

    def verify_article(
        self,
        text: str,
        title: Optional[str] = None,
        custom_claims: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete evidence verification workflow for an article.
        """
        # 1. Extract claims
        claims = custom_claims or self.claim_extractor.extract_claims(text, title=title, max_claims=4)
        if not claims:
            return {
                "claim_verifications": [],
                "total_claims": 0,
                "supported_count": 0,
                "contradicted_count": 0,
                "unverified_count": 0,
                "percent_supported": 0.0,
                "trusted_domains_count": 0,
                "unique_trusted_domains": [],
                "has_any_contradiction": False,
                "has_numeric_conflict": False,
                "summary": "No check-worthy factual claims extracted.",
            }

        verifications = []
        supported_domains = set()
        contradicted_domains = set()
        any_numeric_conflict = False

        for claim in claims:
            # 2. Retrieve candidates
            docs = []
            if self.use_live_retrieval:
                try:
                    docs = self.retriever.retrieve_for_claim(claim)
                except Exception as e:
                    logger.debug(f"Retrieval failed for '{claim}': {e}")

            # 3. Rank sentences
            ranked_sentences = self.ranker.rank_sentences(claim, docs, top_k=2)

            if not ranked_sentences:
                # No evidence found online
                verifications.append({
                    "claim": claim,
                    "status": "Unverified",
                    "source": "No live sources found",
                    "evidence_text": "No corroborating external sources retrieved from live search.",
                    "similarity": 0.0,
                    "is_trusted_source": False,
                    "domain": "",
                    "url": "",
                    "conflicts": [],
                })
                continue

            # Pick best candidate
            best_cand = ranked_sentences[0]
            ev_text = best_cand["sentence"]
            ev_domain = best_cand.get("domain", "")
            ev_url = best_cand.get("url", "")
            is_trusted = best_cand.get("is_trusted", False) or is_trusted_domain(ev_domain)

            # 4. NLI Inference
            nli_res = self.nli_verifier.verify_pair(
                claim=claim,
                evidence_sentence=ev_text,
                factcheck_rating=best_cand.get("factcheck_rating")
            )
            status = nli_res["status"]

            # 5. Entity & Number cross-check
            matcher_res = self.entity_matcher.check_consistency(claim, ev_text)
            conflicts = matcher_res.get("conflicts", [])
            if matcher_res.get("has_numeric_conflict") or matcher_res.get("has_date_conflict"):
                any_numeric_conflict = True
                # If numbers or dates clash, upgrade to Contradicted!
                status = "Contradicted"
                conflicts.append("Linguistic/Numerical conflict flagged between claim and authoritative quote.")

            if status == "Supported" and is_trusted and ev_domain:
                supported_domains.add(ev_domain)
            elif status == "Contradicted" and is_trusted and ev_domain:
                contradicted_domains.add(ev_domain)

            verifications.append({
                "claim": claim,
                "status": status,
                "source": ev_domain or best_cand.get("source_type", "Web Source"),
                "evidence_text": ev_text,
                "similarity": best_cand.get("hybrid_score", 0.0),
                "is_trusted_source": is_trusted,
                "domain": ev_domain,
                "url": ev_url,
                "conflicts": conflicts,
                "nli_details": nli_res,
            })

        total = len(verifications)
        n_supp = sum(1 for v in verifications if v["status"] == "Supported")
        n_cont = sum(1 for v in verifications if v["status"] == "Contradicted")
        n_unver = sum(1 for v in verifications if v["status"] == "Unverified")
        pct_supp = float(n_supp / total) if total > 0 else 0.0

        return {
            "claim_verifications": verifications,
            "total_claims": total,
            "supported_count": n_supp,
            "contradicted_count": n_cont,
            "unverified_count": n_unver,
            "percent_supported": round(pct_supp, 3),
            "trusted_domains_count": len(supported_domains),
            "unique_trusted_domains": list(supported_domains),
            "contradicted_domains": list(contradicted_domains),
            "has_any_contradiction": (n_cont > 0),
            "has_numeric_conflict": any_numeric_conflict,
            "summary": f"{n_supp}/{total} claims supported across {len(supported_domains)} trusted domains.",
        }
