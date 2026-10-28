"""
CAGE-Fake: spaCy-Powered Factual Claim Extractor
Extracts check-worthy factual sentences containing named entities, numerical assertions,
or core subject-verb-object propositions.
"""

from typing import List, Dict, Any, Optional
import re
import spacy

from src.utils import setup_logger

logger = setup_logger("claim_extractor")

# Discard subjective opinion, generic commentary, rhetorical questions
OPINION_MARKERS = {
    "i think", "i believe", "in my opinion", "arguably", "seemingly",
    "perhaps", "could be", "might be", "who knows", "what if"
}


class ClaimExtractor:
    """
    Extracts check-worthy factual assertions from article text using spaCy.
    Prioritizes sentences with:
    - Named entities (PERSON, ORG, GPE, LOC)
    - Quantitative metrics (CARDINAL, MONEY, PERCENT, QUANTITY)
    - Temporal anchors (DATE, TIME)
    - Transitive subject-verb-object structures
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except Exception as e:
            logger.warning(f"Could not load spaCy model {model_name}: {e}. Falling back to default blank parser.")
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

    def extract_claims(self, text: str, title: Optional[str] = None, max_claims: int = 5) -> List[str]:
        """
        Extracts top-k check-worthy factual claims from the text.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text[:10000])  # limit length for efficiency
        candidates = []

        # If title is provided, consider it high priority candidate
        if title and len(title.strip().split()) >= 4:
            clean_title = title.strip()
            if not clean_title.endswith((".", "?", "!")):
                clean_title += "."
            candidates.append((clean_title, 1.5))

        for sent in doc.sents:
            s_text = sent.text.strip()
            # Clean up whitespace
            s_text = re.sub(r"\s+", " ", s_text)
            words = s_text.split()

            # Filter out too short or overly verbose sentences
            if len(words) < 6 or len(words) > 55:
                continue

            # Skip questions or opinion commentary
            if s_text.endswith("?"):
                continue
            lower_s = s_text.lower()
            if any(marker in lower_s for marker in OPINION_MARKERS):
                continue

            # Score factual check-worthiness
            score = 0.0

            # 1. Named Entities (People, Orgs, Places)
            if hasattr(sent, "ents"):
                ents = list(sent.ents)
                for ent in ents:
                    if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "NORP"):
                        score += 0.4
                    elif ent.label_ in ("MONEY", "CARDINAL", "PERCENT", "QUANTITY"):
                        score += 0.5
                    elif ent.label_ in ("DATE", "TIME"):
                        score += 0.3

            # 2. Linguistic Root Verb Check
            has_verb = False
            for token in sent:
                if token.pos_ == "VERB" and token.dep_ in ("ROOT", "ccomp", "conj"):
                    has_verb = True
                    break
            if has_verb:
                score += 0.3

            # 3. Quoted speech or attribution (e.g. "said", "announced", "ruled", "testified")
            if any(v in lower_s for v in ["said", "stated", "ruled", "announced", "reported", "testified", "found", "sentenced"]):
                score += 0.4

            if score >= 0.5:
                candidates.append((s_text, score))

        # Sort by check-worthiness score descending
        # Deduplicate while preserving order
        candidates.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        claims = []
        for c_text, _ in candidates:
            norm_c = re.sub(r"[^\w\s]", "", c_text.lower())
            if norm_c not in seen:
                seen.add(norm_c)
                claims.append(c_text)
                if len(claims) >= max_claims:
                    break

        # Fallback if no candidate meets criteria: pick first 2 sentences
        if not claims:
            for sent in doc.sents:
                st = sent.text.strip()
                if len(st.split()) >= 5:
                    claims.append(st)
                    if len(claims) >= 2:
                        break

        return claims
