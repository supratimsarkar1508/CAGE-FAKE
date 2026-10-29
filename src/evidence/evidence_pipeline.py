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
        source_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete evidence verification workflow for an article.
        - Excludes self-corroborating sources matching the article's own domain or URL.
        - Enforces minimum 0.60 raw SBERT dense cosine similarity threshold.
        - Dynamic thresholding: any similarity >= 0.60 sets status = "Supported" unless explicitly contradicted.
        - Accurately tracks supported domains and independent trusted support rates.
        """
        # 1. Extract claims
        claims = custom_claims or self.claim_extractor.extract_claims(text, title=title, max_claims=4)
        if not claims:
            return {
                "claim_verifications": [],
                "total_claims": 0,
                "supported_count": 0,
                "trusted_supported_count": 0,
                "contradicted_count": 0,
                "trusted_contradicted_count": 0,
                "unverified_count": 0,
                "percent_supported": 0.0,
                "trusted_domains_count": 0,
                "unique_trusted_domains": [],
                "contradicted_domains": [],
                "has_any_contradiction": False,
                "has_numeric_conflict": False,
                "self_source_excluded": 0,
                "summary": "No check-worthy factual claims extracted.",
            }

        # Normalize source URL and domain for self-source exclusion
        source_domain = extract_domain(source_url) if source_url else ""
        norm_source_url = source_url.strip().lower().rstrip("/") if source_url else ""
        self_source_excluded = 0

        verifications = []
        supported_domains = set()
        contradicted_domains = set()
        any_numeric_conflict = False

        for claim in claims:
            # 2. Retrieve candidates
            docs = []
            if self.use_live_retrieval:
                try:
                    raw_docs = self.retriever.retrieve_for_claim(claim)
                except Exception as e:
                    logger.debug(f"Retrieval failed for '{claim}': {e}")
                    raw_docs = []

                # Filter out self-sources
                for doc in raw_docs:
                    doc_url = doc.get("url", "")
                    doc_dom = doc.get("domain", "") or extract_domain(doc_url)
                    norm_doc_url = doc_url.strip().lower().rstrip("/") if doc_url else ""

                    is_self = False
                    if norm_source_url and norm_doc_url and norm_doc_url == norm_source_url:
                        is_self = True
                    elif source_domain and doc_dom:
                        s_dom = source_domain.lower()
                        d_dom = doc_dom.lower()
                        if d_dom == s_dom or d_dom.endswith("." + s_dom) or s_dom.endswith("." + d_dom):
                            is_self = True

                    if is_self:
                        self_source_excluded += 1
                        logger.info(f"Excluded self-corroborating source: {doc_url} ({doc_dom}) matching article domain {source_domain}")
                    else:
                        docs.append(doc)

            # 3. Rank sentences
            ranked_sentences = self.ranker.rank_sentences(claim, docs, top_k=2) if docs else []

            # Drop candidates with dense_score < 0.60 (claim stays Unverified)
            valid_candidates = [c for c in ranked_sentences if c.get("dense_score", 0.0) >= 0.60]

            if not valid_candidates:
                # No evidence found online or candidates below similarity floor
                sim_val = round(float(ranked_sentences[0].get("dense_score", 0.0)), 3) if ranked_sentences else 0.0
                hyb_val = round(float(ranked_sentences[0].get("hybrid_score", 0.0)), 3) if ranked_sentences else 0.0
                verifications.append({
                    "claim": claim,
                    "status": "Unverified",
                    "source": "No live sources >= 0.60 similarity",
                    "evidence_text": "No corroborating external sources met the >= 0.60 semantic similarity floor.",
                    "similarity": sim_val,
                    "dense_score": sim_val,
                    "hybrid_score": hyb_val,
                    "is_trusted_source": False,
                    "domain": "",
                    "url": "",
                    "conflicts": [],
                })
                continue

            # Pick best candidate
            best_cand = valid_candidates[0]
            ev_text = best_cand["sentence"]
            ev_domain = best_cand.get("domain", "")
            ev_url = best_cand.get("url", "")
            is_trusted = bool(best_cand.get("is_trusted", False) or is_trusted_domain(ev_domain))
            dense_score = min(1.0, max(0.0, float(best_cand.get("dense_score", 0.0))))
            hybrid_score = float(best_cand.get("hybrid_score", 0.0))

            # 4. NLI Inference
            nli_res = self.nli_verifier.verify_pair(
                claim=claim,
                evidence_sentence=ev_text,
                factcheck_rating=best_cand.get("factcheck_rating")
            )
            status = nli_res["status"]

            # Dynamic similarity floor rule: any candidate with similarity >= 0.60 dynamically sets status = "Supported"
            # unless an explicit contradiction is detected
            if dense_score >= 0.60 and status != "Contradicted":
                status = "Supported"

            # 5. Entity, Number & Location cross-check
            matcher_res = self.entity_matcher.check_consistency(claim, ev_text, is_trusted=is_trusted)
            conflicts = list(matcher_res.get("conflicts", []))

            # Numeric or Date conflict on the same event from trusted source marks Contradicted
            if (matcher_res.get("has_numeric_conflict") or matcher_res.get("has_date_conflict")) and is_trusted and matcher_res.get("same_event"):
                status = "Contradicted"
                any_numeric_conflict = True
                conflicts.append("Linguistic/Numerical conflict flagged between claim and authoritative quote on same event.")
            elif matcher_res.get("location_mismatch"):
                conflicts.append("Geographical note: Claim and retrieved passage mention different locations, but core factual match is preserved.")

            if status == "Supported" and is_trusted and ev_domain:
                supported_domains.add(ev_domain)
            elif status == "Contradicted" and is_trusted and ev_domain:
                contradicted_domains.add(ev_domain)

            verifications.append({
                "claim": claim,
                "status": status,
                "source": ev_domain or best_cand.get("source_type", "Web Source"),
                "evidence_text": ev_text,
                "similarity": round(dense_score, 3),      # Raw SBERT cosine in [0, 1]
                "dense_score": round(dense_score, 3),
                "hybrid_score": round(hybrid_score, 3),
                "is_trusted_source": is_trusted,
                "domain": ev_domain,
                "url": ev_url,
                "conflicts": conflicts,
                "nli_details": nli_res,
                "location_mismatch": matcher_res.get("location_mismatch", False),
            })

        total = len(verifications)
        # Requirement 7:
        # percent_supported = Supported claims from TRUSTED, independent sources / total
        trusted_supp = sum(1 for v in verifications if v["status"] == "Supported" and v.get("is_trusted_source", False))
        n_supp_all = sum(1 for v in verifications if v["status"] == "Supported")
        n_cont_trusted = sum(1 for v in verifications if v["status"] == "Contradicted" and v.get("is_trusted_source", False))
        n_cont_all = sum(1 for v in verifications if v["status"] == "Contradicted")
        n_unver = sum(1 for v in verifications if v["status"] == "Unverified")
        pct_supp = float(trusted_supp / total) if total > 0 else 0.0
        has_any_contradiction = (n_cont_trusted > 0)

        return {
            "claim_verifications": verifications,
            "total_claims": total,
            "supported_count": n_supp_all,
            "trusted_supported_count": trusted_supp,
            "contradicted_count": n_cont_all,
            "trusted_contradicted_count": n_cont_trusted,
            "unverified_count": n_unver,
            "percent_supported": round(pct_supp, 3),
            "trusted_domains_count": len(supported_domains),
            "unique_trusted_domains": list(supported_domains),
            "contradicted_domains": list(contradicted_domains),
            "has_any_contradiction": has_any_contradiction,
            "has_numeric_conflict": any_numeric_conflict,
            "self_source_excluded": self_source_excluded,
            "summary": f"{trusted_supp}/{total} claims supported across {len(supported_domains)} trusted domains.",
        }
