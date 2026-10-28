"""
CAGE-Fake: Evidence Analysis and Claim Verification Engine
Extracts key claims from article text, semantically verifies each claim against
retrieved sources or an authoritative knowledge base using Sentence-BERT embeddings,
and marks claims as Supported, Contradicted, or Unverified.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np

from config import (
    SBERT_MODEL_NAME,
    EVIDENCE_SIMILARITY_THRESHOLD,
    EVIDENCE_CONTRADICTION_THRESHOLD,
    KNOWLEDGE_BASE_DIR,
)
from src.utils import setup_logger

logger = setup_logger("evidence_analysis")

# Pre-compiled regex patterns for claim extraction
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
NON_CLAIM_PATTERNS = [
    re.compile(r"^(subscribe|follow us|click here|sign up|read more|advertisement|copyright|all rights reserved)", re.IGNORECASE),
    re.compile(r"\b(photo by|image credit|source:|editor's note:)\b", re.IGNORECASE),
]

CONTRADICTION_CUES = {
    "false", "debunked", "untrue", "contrary", "denied", "no evidence",
    "disproven", "refuted", "hoax", "fabricated", "conspiracy", "incorrect",
    "myth", "fake", "not true", "never occurred", "impossible"
}

NEGATION_TOKENS = {"not", "never", "no", "neither", "nor", "none", "cannot", "without"}


class ClaimExtractor:
    """
    Extracts informative, assertive key claims from article headlines and bodies.
    Filters out conversational trivia, boilerplate, and rhetorical questions.
    """

    def __init__(self, min_words: int = 5, max_claims: int = 5):
        self.min_words = min_words
        self.max_claims = max_claims

    def extract_claims(self, text: str, title: Optional[str] = None) -> List[str]:
        """
        Segments text into sentences, scores factual assertiveness, and extracts top key claims.
        """
        raw_sentences = []
        if title and len(title.strip().split()) >= self.min_words:
            raw_sentences.append(title.strip())

        body_sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text or "") if s.strip()]
        raw_sentences.extend(body_sentences)

        candidate_claims = []
        for s in raw_sentences:
            s_clean = re.sub(r"\s+", " ", s).strip()
            words = s_clean.split()
            if len(words) < self.min_words:
                continue

            # Skip questions, greetings, and boilerplate
            if s_clean.endswith("?") or any(pat.search(s_clean) for pat in NON_CLAIM_PATTERNS):
                continue

            # Score informational assertiveness
            has_digits = bool(re.search(r"\d", s_clean))
            has_proper_nouns = sum(1 for w in words[1:] if w and w[0].isupper()) > 0
            has_assertive_verb = bool(re.search(r"\b(confirmed|announced|passed|discovered|showed|found|signed|reported|stated|ruled|approved|died|killed|launched|created|built)\b", s_clean, re.I))

            salience = len(words)
            if has_digits:
                salience += 15
            if has_proper_nouns:
                salience += 10
            if has_assertive_verb:
                salience += 20

            candidate_claims.append((salience, s_clean))

        if not candidate_claims:
            # Fallback to the first available non-empty sentence or full text snippet
            fallback = title if title and len(title.strip()) > 10 else text.strip()[:150]
            return [fallback] if fallback else ["Article makes general factual assertions."]

        # Sort by salience descending and keep unique claims
        candidate_claims.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        unique_claims = []
        for _, claim in candidate_claims:
            normalized = claim.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_claims.append(claim)
            if len(unique_claims) >= self.max_claims:
                break

        return unique_claims


class EvidenceKnowledgeBase:
    """
    Authoritative reference knowledge base containing verified facts and documented debunks
    across science, technology, public policy, medicine, geopolitics, and historical events.
    """

    DEFAULT_FACTS: List[Dict[str, Any]] = [
        # Science & Space
        {
            "statement": "The James Webb Space Telescope was successfully launched in December 2021 by NASA, ESA, and CSA and observes infrared astronomy.",
            "source": "NASA Goddard Space Flight Center",
            "is_debunk": False,
        },
        {
            "statement": "Earth is an oblate spheroid revolving around the Sun, confirmed by centuries of geodesy, astronomy, and satellite telemetry.",
            "source": "USGS / NASA Earth Observatory",
            "is_debunk": False,
        },
        {
            "statement": "Claims that the Earth is flat or bounded by an impenetrable ice wall are scientifically refuted by orbital satellite imagery and gravitational physics.",
            "source": "National Science Foundation Archive",
            "is_debunk": True,
            "debunks_topic": "flat earth",
        },
        {
            "statement": "Apollo 11 landed astronauts Neil Armstrong and Buzz Aldrin on the Moon on July 20, 1969.",
            "source": "NASA Historical Archive",
            "is_debunk": False,
        },
        {
            "statement": "Claims that the Apollo moon landings were filmed on a Hollywood soundstage have been thoroughly debunked by independent lunar telemetry and retroreflector arrays.",
            "source": "Smithsonian National Air and Space Museum",
            "is_debunk": True,
            "debunks_topic": "moon landing hoax hollywood stage",
        },
        {
            "statement": "There is no credible scientific evidence confirming the discovery of alien megastructures or extraterrestrial civilizations around Alpha Centauri.",
            "source": "SETI Institute / Royal Astronomical Society",
            "is_debunk": True,
            "debunks_topic": "alien megastructure extraterrestrial civilization",
        },
        # Public Policy, Geopolitics & Global Affairs
        {
            "statement": "International delegates and governments participate in global climate summits to negotiate emissions targets and renewable energy agreements.",
            "source": "United Nations Climate Change Secretariat (UNFCCC)",
            "is_debunk": False,
        },
        {
            "statement": "Congressional leaders and legislatures negotiate bipartisan infrastructure and budget funding for transportation, broadband, and clean energy investments.",
            "source": "Associated Press / Congressional Record",
            "is_debunk": False,
        },
        {
            "statement": "The European Union is an established economic and political union consisting of 27 democratic member nations.",
            "source": "European Commission Official Record",
            "is_debunk": False,
        },
        {
            "statement": "The Federal Reserve regulates monetary policy, sets benchmark federal funds rates, and conducts economic stabilization in the United States.",
            "source": "Federal Reserve Board of Governors",
            "is_debunk": False,
        },
        {
            "statement": "Elections in democratic nations utilize verified paper audit trails, decentralized local election boards, and cryptographic chain-of-custody protocols.",
            "source": "Cybersecurity and Infrastructure Security Agency (CISA)",
            "is_debunk": False,
        },
        # Health & Medicine
        {
            "statement": "COVID-19 and influenza vaccines undergo multi-phase controlled clinical safety trials regulated by the FDA, EMA, and WHO.",
            "source": "World Health Organization (WHO) Technical Briefing",
            "is_debunk": False,
        },
        {
            "statement": "Vaccines do not contain microchips, tracking transmitters, or toxic heavy metals; they contain biological antigens and standard stabilizers.",
            "source": "CDC Vaccine Safety Registry",
            "is_debunk": True,
            "debunks_topic": "vaccine microchip tracking",
        },
        {
            "statement": "Antibiotics treat bacterial infections and are ineffective against viral infections such as colds, the flu, or coronaviruses.",
            "source": "Centers for Disease Control and Prevention (CDC)",
            "is_debunk": False,
        },
        {
            "statement": "Injecting, consuming, or inhaling disinfectants, chlorine dioxide, or industrial bleach is toxic, causes organ damage, and is medically contraindicated.",
            "source": "FDA Consumer Health Warning Advisory",
            "is_debunk": True,
            "debunks_topic": "disinfectant cure bleach",
        },
        {
            "statement": "5G wireless technology utilizes non-ionizing radiofrequency radiation and is biologically incapable of transmitting viruses or biological pathogens.",
            "source": "International Commission on Non-Ionizing Radiation Protection (ICNIRP)",
            "is_debunk": True,
            "debunks_topic": "5G virus pathogen transmission",
        },
        # Technology & Computing
        {
            "statement": "Machine learning and artificial intelligence models process linguistic patterns and statistical correlations learned from training distributions.",
            "source": "ACM / IEEE Computer Society Transactions",
            "is_debunk": False,
        },
        {
            "statement": "HAARP is an ionospheric research facility in Alaska and possesses no scientific capability to control weather, trigger earthquakes, or manipulate human minds.",
            "source": "Geophysical Institute, University of Alaska Fairbanks",
            "is_debunk": True,
            "debunks_topic": "HAARP weather control earthquake mind",
        },
    ]

    def __init__(self, custom_records: Optional[List[Dict[str, Any]]] = None):
        self.records: List[Dict[str, Any]] = list(self.DEFAULT_FACTS)
        if custom_records:
            self.records.extend(custom_records)
        self._load_local_kb_files()
        self._model = None
        self._embeddings = None

    def _load_local_kb_files(self) -> None:
        """Loads additional knowledge base records from KNOWLEDGE_BASE_DIR if present."""
        if KNOWLEDGE_BASE_DIR.exists():
            import json
            for p in KNOWLEDGE_BASE_DIR.glob("*.json*"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and "statement" in item:
                                    self.records.append(item)
                except Exception as e:
                    logger.warning(f"Could not load knowledge base file {p}: {e}")

    def add_evidence(self, statement: str, source: str, is_debunk: bool = False, debunks_topic: str = "") -> None:
        """Adds a verified evidence record and invalidates the cached embeddings."""
        self.records.append({
            "statement": statement,
            "source": source,
            "is_debunk": is_debunk,
            "debunks_topic": debunks_topic,
        })
        self._embeddings = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                try:
                    self._model = SentenceTransformer(SBERT_MODEL_NAME, local_files_only=True)
                except Exception:
                    self._model = SentenceTransformer(SBERT_MODEL_NAME)
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformer for knowledge base: {e}")
                raise
        return self._model

    def get_embeddings(self) -> np.ndarray:
        """Returns normalized SBERT embeddings for all statements in the knowledge base."""
        if self._embeddings is None:
            model = self._get_model()
            statements = [r["statement"] for r in self.records]
            raw_emb = model.encode(statements, convert_to_numpy=True, show_progress_bar=False)
            norms = np.linalg.norm(raw_emb, axis=1, keepdims=True)
            self._embeddings = raw_emb / np.maximum(norms, 1e-9)
        return self._embeddings


class EvidenceVerifier:
    """
    Checks extracted claims against retrieved sources or the Knowledge Base using Sentence-BERT embeddings.
    Categorizes each claim as Supported, Contradicted, or Unverified with citations and semantic similarity.
    """

    def __init__(
        self,
        knowledge_base: Optional[EvidenceKnowledgeBase] = None,
        similarity_threshold: float = EVIDENCE_SIMILARITY_THRESHOLD,
        contradiction_threshold: float = EVIDENCE_CONTRADICTION_THRESHOLD,
    ):
        self.kb = knowledge_base or EvidenceKnowledgeBase()
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.claim_extractor = ClaimExtractor()
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = self.kb._get_model()
        return self._model

    def verify_claim(
        self,
        claim: str,
        retrieved_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Verifies a single claim using SBERT semantic similarity and contradiction detection.
        Checks against both retrieved sources (if provided) and the knowledge base.
        """
        model = self._get_model()
        claim_emb = model.encode([claim], convert_to_numpy=True, show_progress_bar=False)[0]
        norm = np.linalg.norm(claim_emb)
        if norm > 0:
            claim_emb = claim_emb / norm

        # Build candidate pool from KB + retrieved sources
        candidate_records = list(self.kb.records)
        kb_embs = self.kb.get_embeddings()

        if retrieved_sources:
            ret_statements = [r["statement"] for r in retrieved_sources]
            ret_emb = model.encode(ret_statements, convert_to_numpy=True, show_progress_bar=False)
            ret_norms = np.linalg.norm(ret_emb, axis=1, keepdims=True)
            ret_embs_norm = ret_emb / np.maximum(ret_norms, 1e-9)
            all_embs = np.vstack([kb_embs, ret_embs_norm])
            candidate_records.extend(ret_sources := retrieved_sources)
        else:
            all_embs = kb_embs

        similarities = np.dot(all_embs, claim_emb)
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        best_record = candidate_records[best_idx]

        # Check for contradiction vs support
        claim_lower = claim.lower()
        kb_lower = best_record["statement"].lower()

        is_contradiction = False
        if best_record.get("is_debunk", False):
            # If the matching evidence is a debunk of this specific topic, the claim is contradicted
            topic = best_record.get("debunks_topic", "").lower()
            if topic and any(t in claim_lower for t in topic.split()):
                is_contradiction = True
            elif best_sim >= self.contradiction_threshold:
                is_contradiction = True
        else:
            # Check polarity negation between claim and evidence
            claim_negated = any(re.search(rf"\b{w}\b", claim_lower) for w in NEGATION_TOKENS)
            kb_negated = any(re.search(rf"\b{w}\b", kb_lower) for w in NEGATION_TOKENS)
            has_contradiction_cue = any(re.search(rf"\b{w}\b", claim_lower) for w in CONTRADICTION_CUES)

            if (claim_negated != kb_negated and best_sim >= self.contradiction_threshold) or (has_contradiction_cue and best_sim >= self.contradiction_threshold):
                is_contradiction = True

        if best_sim >= self.similarity_threshold:
            if is_contradiction:
                status = "Contradicted"
            else:
                status = "Supported"
            source = best_record["source"]
            evidence_text = best_record["statement"]
        elif is_contradiction and best_sim >= self.contradiction_threshold:
            status = "Contradicted"
            source = best_record["source"]
            evidence_text = best_record["statement"]
        else:
            status = "Unverified"
            source = "No verified source found"
            evidence_text = "No corroborating evidence located in authoritative knowledge base or retrieved sources."

        return {
            "claim": claim,
            "status": status,
            "source": source,
            "evidence_text": evidence_text,
            "similarity": round(best_sim, 3),
        }

    def verify_article(
        self,
        text: str,
        title: Optional[str] = None,
        custom_claims: Optional[List[str]] = None,
        retrieved_sources: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extracts key claims and checks each against retrieved sources or the knowledge base.
        Computes summary statistics, evidence conflict score, and per-claim status.
        """
        # Standardize retrieved sources to list of dicts
        standardized_sources = None
        if retrieved_sources:
            standardized_sources = []
            for item in retrieved_sources:
                if isinstance(item, str) and item.strip():
                    is_deb = any(cue in item.lower() for cue in CONTRADICTION_CUES)
                    standardized_sources.append({
                        "statement": item.strip(),
                        "source": "Retrieved Document Source",
                        "is_debunk": is_deb,
                    })
                elif isinstance(item, dict) and "statement" in item:
                    standardized_sources.append(item)

        claims = custom_claims or self.claim_extractor.extract_claims(text, title=title)
        verifications = [self.verify_claim(c, retrieved_sources=standardized_sources) for c in claims]

        supported = [v for v in verifications if v["status"] == "Supported"]
        contradicted = [v for v in verifications if v["status"] == "Contradicted"]
        unverified = [v for v in verifications if v["status"] == "Unverified"]

        total = len(verifications)
        n_supp = len(supported)
        n_cont = len(contradicted)
        n_unver = len(unverified)

        has_supp = n_supp > 0 and n_cont == 0
        has_cont = n_cont > 0
        has_no_ev = (n_unver == total) or (total == 0)
        has_conf = n_supp > 0 and n_cont > 0

        # Conflict score: 0.95 for mixed support/contradiction, 0.70+ for pure contradiction, 0.50 for no evidence, 0.0 for pure support
        if has_conf:
            conflict_score = 0.95
        elif has_cont:
            conflict_score = min(1.0, 0.70 + 0.15 * n_cont)
        elif has_no_ev:
            conflict_score = 0.50
        else:
            conflict_score = 0.0

        return {
            "total_claims": total,
            "supported_count": n_supp,
            "contradicted_count": n_cont,
            "unverified_count": n_unver,
            "has_supporting_evidence": has_supp,
            "has_contradicting_evidence": has_cont,
            "has_no_evidence": has_no_ev,
            "has_conflicting_evidence": has_conf,
            "evidence_conflict_score": round(conflict_score, 3),
            "claim_verifications": verifications,
            "per_claim_evidence_status": verifications,
            "retrieved_sources_count": len(standardized_sources) if standardized_sources else 0,
        }
