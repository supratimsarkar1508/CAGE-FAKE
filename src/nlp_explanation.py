"""
CAGE-Fake: Rule-Based and Semantic Linguistic Signal Summary Module
Generates evidence-grounded linguistic analysis and detects concrete red flags:
- Sensational or clickbait wording
- Emotional and loaded language
- Epistemic hedging ("reportedly", "sources say", "allegedly")
- Missing attribution and unanchored assertions
- Inconsistent named entities, dates, or numbers
- Contradictions within the text
- Semantic mismatch between headline and body
- Punctuation anomalies (shouting / sensational punctuation)
- Cognitive readability and lexical diversity
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import re
import string
import numpy as np

from config import (
    HEADLINE_BODY_MISMATCH_THRESHOLD,
    SBERT_MODEL_NAME,
)
from src.utils import setup_logger

logger = setup_logger("nlp_explanation")

CLICKBAIT_PHRASES = [
    re.compile(r"\b(you won'?t believe)\b", re.I),
    re.compile(r"\b(what happened next)\b", re.I),
    re.compile(r"\b(shocking truth)\b", re.I),
    re.compile(r"\b(they don'?t want you to know)\b", re.I),
    re.compile(r"\b(secret (revealed|exposed|leaked))\b", re.I),
    re.compile(r"\b(mind-?blowing)\b", re.I),
    re.compile(r"\b(jaw-?dropping)\b", re.I),
    re.compile(r"\b(must see)\b", re.I),
    re.compile(r"\b(won'?t tell you)\b", re.I),
    re.compile(r"\b(will blow your mind)\b", re.I),
    re.compile(r"\b(miracle cure)\b", re.I),
    re.compile(r"\b(mainstream media is silent)\b", re.I),
    re.compile(r"\b(the real reason why)\b", re.I),
]

SENSATIONAL_WORDS = {
    "shocking", "bombshell", "unbelievable", "devastating", "mind-blowing", "exposed",
    "secretly", "uncovered", "horrific", "explosive", "miraculous", "stunning", "panic",
    "insane", "disaster", "treason", "conspiracy", "scandal", "apocalypse", "nightmare",
    "catastrophic", "chaos", "terrifying", "bloodcurdling", "unprecedented", "fiasco",
}

EMOTIONAL_WORDS = {
    "outrageous", "disgusting", "vicious", "corrupt", "evil", "heroic", "furious",
    "threat", "danger", "frightening", "disgrace", "shameful", "sinister", "pure",
    "monstrous", "pathetic", "hero", "villain", "toxic", "destroy", "annihilate",
    "bloodbath", "traitor", "tyranny", "criminal", "despicable", "barbaric", "pure evil"
}

ATTRIBUTION_VERBS = {
    "said", "stated", "confirmed", "reported", "announced", "declared", "explained",
    "noted", "remarked", "testified", "admitted", "acknowledged", "cited", "asserted",
    "concluded", "emphasized", "insisted", "indicated", "revealed", "told", "warned",
    "spokesperson", "spokesman", "representative", "officials", "according to"
}

HEDGE_PHRASES = [
    re.compile(r"\b(sources say)\b", re.I),
    re.compile(r"\b(unconfirmed reports)\b", re.I),
    re.compile(r"\b(rumors suggest)\b", re.I),
    re.compile(r"\b(it is claimed)\b", re.I),
    re.compile(r"\b(claims indicate)\b", re.I),
    re.compile(r"\b(could possibly)\b", re.I),
]

HEDGE_WORDS = {
    "alleged", "allegedly", "apparently", "reportedly", "purportedly", "supposedly",
    "rumor", "rumored", "rumours", "claims", "claimed", "claiming", "suggests", "suggested",
    "could", "might", "maybe", "perhaps", "possibly", "possible", "potential",
    "seem", "seems", "seemed", "appear", "appears", "speculate", "assumed"
}

WORD_REGEX = re.compile(r"\b[a-zA-Z0-9_\'-]+\b")
SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+")
YEAR_REGEX = re.compile(r"\b(19\d\d|20[0-3]\d)\b")
NUMBER_WORD_REGEX = re.compile(r"\b(\d+([.,]\d+)?\s*(percent|%|million|billion|thousand|people|deaths|cases|patients)?)\b", re.I)


class NLPLinguisticExplainer:
    """
    Computes linguistic dimensions, performs semantic cross-validation (headline-body congruence,
    intra-text consistency), and generates explicit linguistic red flags.
    """

    def __init__(self):
        self._sbert = None

    def _get_sbert(self):
        if self._sbert is None:
            try:
                from sentence_transformers import SentenceTransformer
                try:
                    self._sbert = SentenceTransformer(SBERT_MODEL_NAME, local_files_only=True)
                except Exception:
                    self._sbert = SentenceTransformer(SBERT_MODEL_NAME)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer in NLPLinguisticExplainer: {e}")
                self._sbert = None
        return self._sbert

    def analyze(
        self,
        raw_text: str,
        clean_text: str,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes text and returns dimensional scores, detected red flags, and diagnostic summary.
        """
        raw = str(raw_text or "")
        clean = str(clean_text or "")
        headline = str(title or "").strip()

        total_chars = max(1, len(raw))
        tokens = [w.lower() for w in WORD_REGEX.findall(clean)]
        total_tokens = max(1, len(tokens))
        sentences = [s.strip() for s in SENTENCE_REGEX.split(raw) if s.strip()]
        total_sentences = max(1, len(sentences))

        red_flags: List[str] = []

        # 1. Sensationalism & Clickbait
        sensational_hits = [w for w in tokens if w in SENSATIONAL_WORDS]
        clickbait_hits = [m.group(0) for pat in CLICKBAIT_PHRASES for m in [pat.search(raw)] if m]
        if headline:
            clickbait_hits.extend([m.group(0) for pat in CLICKBAIT_PHRASES for m in [pat.search(headline)] if m])
        clickbait_hits = list(set(clickbait_hits))

        sensationalism_density = len(sensational_hits) / total_tokens * 100.0
        if clickbait_hits:
            red_flags.append(f"Clickbait phrasing detected: '{', '.join(clickbait_hits[:3])}'")
        if sensationalism_density > 1.2:
            red_flags.append(f"Elevated sensationalism ({sensationalism_density:.1f}%): terms like '{', '.join(list(set(sensational_hits))[:3])}'")

        # 2. Emotional & Loaded Language
        emotional_hits = [w for w in tokens if w in EMOTIONAL_WORDS]
        emotional_density = len(emotional_hits) / total_tokens * 100.0
        if emotional_density > 1.0:
            red_flags.append(f"Loaded / emotional rhetoric detected ({emotional_density:.1f}%): '{', '.join(list(set(emotional_hits))[:3])}'")

        # 3. Epistemic Hedging & Unverified Sourcing
        hedge_hits = [w for w in tokens if w in HEDGE_WORDS]
        phrase_hedges = [m.group(0) for pat in HEDGE_PHRASES for m in [pat.search(raw)] if m]
        hedging_density = len(hedge_hits) / total_tokens * 100.0
        if phrase_hedges:
            red_flags.append(f"Speculative sourcing phrases detected: '{', '.join(set(phrase_hedges))}'")
        elif hedging_density > 1.5:
            red_flags.append(f"Heavy speculative hedging ({hedging_density:.1f}%): '{', '.join(list(set(hedge_hits))[:3])}'")

        # 4. Source Attribution Check
        raw_lower = raw.lower()
        attr_hits = [w for w in tokens if w in ATTRIBUTION_VERBS]
        has_according_to = "according to" in raw_lower or "spokesperson" in raw_lower or "official" in raw_lower
        attr_count = len(attr_hits) + (2 if has_according_to else 0)
        attr_density = attr_count / total_tokens * 100.0
        has_quotes = raw.count('"') >= 2 or raw.count("“") >= 1 or raw.count("'") >= 2

        if attr_density < 0.25 and not has_quotes and total_tokens > 25:
            red_flags.append("Missing source attribution: article asserts claims without citing spokespersons, studies, or named sources")

        # 5. Inconsistent Named Entities, Dates, or Numbers
        # A. Chronological date discrepancies
        years_found = list(set(YEAR_REGEX.findall(raw)))
        if len(years_found) >= 2:
            try:
                numeric_years = [int(y) for y in years_found]
                year_span = max(numeric_years) - min(numeric_years)
                if year_span > 10 and not any(k in raw_lower for k in ["since", "history", "anniversary", "decades", "prior", "historical", "earlier", "compared to", "19"]):
                    red_flags.append(f"Chronological timeline discrepancy: conflicting dates found ({', '.join(years_found)})")
            except Exception:
                pass

        # B. Named Entity role and title inconsistencies
        entity_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", raw)
        if entity_matches:
            # Check for conflicting titles associated with the same person/entity
            entity_titles = {}
            title_keywords = ["president", "prime minister", "senator", "governor", "ceo", "dr", "professor", "spokesperson", "director"]
            for s in sentences:
                s_l = s.lower()
                for ent in set(entity_matches):
                    if ent in s:
                        surname = ent.split()[-1].lower()
                        for t in title_keywords:
                            if t in s_l:
                                entity_titles.setdefault(surname, set()).add(t)
            for surname, titles in entity_titles.items():
                if len(titles) >= 2 and not ({"president", "prime minister"} & titles and len(titles) == 1):
                    # Check if conflicting high-level offices are attributed to the same surname
                    if ("president" in titles and "senator" in titles) or ("prime minister" in titles and "governor" in titles):
                        red_flags.append(f"Inconsistent named entity titles: '{surname.title()}' assigned conflicting roles ({', '.join(titles)})")

        # C. Headline numbers vs body numbers and intra-body numerical discrepancies
        if headline:
            headline_numbers = [m[0] for m in NUMBER_WORD_REGEX.findall(headline)]
            body_numbers = [m[0] for m in NUMBER_WORD_REGEX.findall(raw)]
            for hn in headline_numbers:
                hn_clean = re.sub(r"[^\d]", "", hn)
                if hn_clean and len(hn_clean) >= 2:
                    if int(hn_clean) > 100 and hn not in raw and not any(hn_clean in bn for bn in body_numbers):
                        red_flags.append(f"Numerical inconsistency: Headline mentions '{hn}', which is not substantiated in the body text")
                        break

        # Intra-body order of magnitude contradiction
        all_numeric = [int(m) for m in re.findall(r"\b\d{2,7}\b", raw) if 10 <= int(m) <= 10000000]
        if len(all_numeric) >= 3:
            max_num, min_num = max(all_numeric), min(all_numeric)
            if max_num >= 10000 and min_num <= 50 and not any(w in raw_lower for w in ["percent", "%", "year", "age", "day", "hour", "rank", "step", "page"]):
                # Potential statistics collision
                pass

        # 6. Contradictions Within the Text (Semantic intra-textual check using SBERT)
        intra_contradiction = self._check_internal_contradiction(sentences)
        if intra_contradiction:
            red_flags.append(intra_contradiction)

        # 7. Headline and Body Semantic Mismatch
        headline_body_sim = None
        if headline and len(headline.split()) >= 3 and len(sentences) >= 1:
            headline_body_sim = self._compute_headline_body_similarity(headline, raw[:1500])
            if headline_body_sim is not None and headline_body_sim < HEADLINE_BODY_MISMATCH_THRESHOLD:
                red_flags.append(f"Headline-body mismatch: Headline diverges semantically from the body text (cosine similarity: {headline_body_sim:.2f})")

        # 8. Punctuation Intensity (Shouting / Emotional Punctuation)
        excl_count = raw.count("!")
        quest_count = raw.count("?")
        caps_tokens = sum(1 for tok in raw.split() if tok.isupper() and len(tok) > 2 and tok not in {"USA", "FDA", "CDC", "WHO", "NASA", "CEO", "GOP", "DNA", "SBERT", "COVID", "UN"})
        punct_intensity_score = (excl_count * 2.0 + quest_count * 1.5 + caps_tokens) / max(1, total_sentences)
        if caps_tokens >= 3 or excl_count >= 3:
            red_flags.append(f"Sensational punctuation / shouting: {excl_count} exclamation marks, {caps_tokens} ALL-CAPS words")

        # 9. Lexical Diversity
        unique_tokens = len(set(tokens))
        ttr = unique_tokens / total_tokens

        # 10. Readability Level
        words_per_sent = total_tokens / total_sentences
        readability_level = "Complex/Academic" if words_per_sent > 25 else ("Standard" if words_per_sent > 14 else "Simple/Conversational")

        # Tense Consistency
        past_verbs = sum(1 for w in tokens if w.endswith("ed") or w in {"was", "were", "had", "said", "did"})
        present_verbs = sum(1 for w in tokens if w.endswith("ing") or w in {"is", "are", "has", "says", "does"})
        total_tense = past_verbs + present_verbs
        tense_consistency = (max(past_verbs, present_verbs) / total_tense) if total_tense > 0 else 1.0

        quote_chars = raw.count('"') + raw.count("'") + raw.count("“") + raw.count("”")
        quote_density = quote_chars / total_chars * 100.0

        signals = {
            "sensationalism": {
                "score": round(sensationalism_density, 2),
                "level": "High" if (sensationalism_density > 1.2 or bool(clickbait_hits)) else ("Moderate" if sensationalism_density > 0.4 else "Low"),
                "examples": list(set(sensational_hits + clickbait_hits))[:4],
            },
            "hedging": {
                "score": round(hedging_density, 2),
                "level": "High" if (hedging_density > 1.5 or bool(phrase_hedges)) else ("Moderate" if hedging_density > 0.6 else "Low"),
                "examples": list(set(hedge_hits + phrase_hedges))[:4],
            },
            "attribution": {
                "score": round(attr_density, 2),
                "level": "High" if attr_density > 1.0 else ("Moderate" if attr_density > 0.3 else "Low"),
                "examples": list(set(attr_hits))[:4],
            },
            "emotional_language": {
                "score": round(emotional_density, 2),
                "level": "High" if emotional_density > 1.0 else ("Moderate" if emotional_density > 0.3 else "Low"),
                "examples": list(set(emotional_hits))[:4],
            },
            "punctuation_intensity": {
                "score": round(punct_intensity_score, 2),
                "level": "High" if punct_intensity_score > 1.5 else ("Moderate" if punct_intensity_score > 0.5 else "Low"),
                "details": f"{excl_count} exclamation marks, {quest_count} question marks, {caps_tokens} ALL-CAPS words",
            },
            "lexical_diversity": {
                "score": round(ttr, 3),
                "level": "High" if ttr > 0.6 else ("Moderate" if ttr > 0.4 else "Low (Repetitive)"),
            },
            "tense_consistency": {
                "score": round(tense_consistency, 2),
                "level": "High (Stable)" if tense_consistency > 0.75 else "Erratic / Mixed",
            },
            "readability": {
                "avg_sentence_len": round(words_per_sent, 1),
                "level": readability_level,
            },
            "quotation_density": {
                "score": round(quote_density, 2),
                "level": "High" if quote_density > 1.5 else ("Moderate" if quote_density > 0.5 else "Low (Few Quotes)"),
            },
            "headline_body_similarity": round(headline_body_sim, 3) if headline_body_sim is not None else None,
        }

        summary_text = self._generate_templated_summary(signals, red_flags)

        return {
            "signals": signals,
            "linguistic_red_flags": red_flags,
            "summary": summary_text,
        }

    def _check_internal_contradiction(self, sentences: List[str]) -> Optional[str]:
        """
        Detects intra-textual contradictions using semantic sentence embeddings (SBERT)
        and negation polarity checks rather than keyword matching alone.
        """
        if len(sentences) < 2:
            return None

        # Filter sentences with substantive content
        cand_sentences = [s.strip() for s in sentences if len(s.split()) >= 4][:12]
        if len(cand_sentences) < 2:
            return None

        negation_terms = {"not", "never", "no", "denied", "failed", "untrue", "debunked", "refuted", "false", "impossible"}

        # Try semantic embeddings first
        sbert = self._get_sbert()
        if sbert is not None:
            try:
                embs = sbert.encode(cand_sentences, convert_to_numpy=True, show_progress_bar=False)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-9)
                norm_embs = embs / norms
                sim_matrix = np.dot(norm_embs, norm_embs.T)

                for i in range(len(cand_sentences)):
                    s1 = cand_sentences[i].lower()
                    neg1 = any(re.search(rf"\b{w}\b", s1) for w in negation_terms)

                    for j in range(i + 1, len(cand_sentences)):
                        sim = float(sim_matrix[i, j])
                        if sim >= 0.52:
                            s2 = cand_sentences[j].lower()
                            neg2 = any(re.search(rf"\b{w}\b", s2) for w in negation_terms)
                            if neg1 != neg2:
                                return f"Internal contradiction: Sentence '{cand_sentences[i][:60]}...' directly conflicts in polarity with '{cand_sentences[j][:60]}...'"
            except Exception as e:
                logger.warning(f"Semantic internal contradiction check failed, falling back to lexical: {e}")

        # Lexical fallback
        for i in range(len(sentences)):
            s1 = sentences[i].lower()
            tokens1 = set(WORD_REGEX.findall(s1)) - {"the", "a", "an", "is", "was", "in", "on", "at", "to", "for", "and", "or"}
            if len(tokens1) < 4:
                continue

            for j in range(i + 1, min(len(sentences), i + 6)):
                s2 = sentences[j].lower()
                tokens2 = set(WORD_REGEX.findall(s2)) - {"the", "a", "an", "is", "was", "in", "on", "at", "to", "for", "and", "or"}
                if len(tokens2) < 4:
                    continue

                overlap = tokens1.intersection(tokens2)
                overlap_ratio = len(overlap) / min(len(tokens1), len(tokens2))

                if overlap_ratio >= 0.50:
                    neg1 = any(re.search(rf"\b{w}\b", s1) for w in negation_terms)
                    neg2 = any(re.search(rf"\b{w}\b", s2) for w in negation_terms)
                    if neg1 != neg2:
                        return f"Internal contradiction: Sentence '{sentences[i][:60]}...' directly conflicts in polarity with '{sentences[j][:60]}...'"

        return None

    def _compute_headline_body_similarity(self, headline: str, body: str) -> Optional[float]:
        """Computes semantic cosine similarity between headline and body using SBERT."""
        sbert = self._get_sbert()
        if sbert is None:
            return None
        try:
            embs = sbert.encode([headline, body], convert_to_numpy=True, show_progress_bar=False)
            norm0 = np.linalg.norm(embs[0])
            norm1 = np.linalg.norm(embs[1])
            if norm0 > 0 and norm1 > 0:
                return float(np.dot(embs[0], embs[1]) / (norm0 * norm1))
        except Exception as e:
            logger.warning(f"Failed to compute headline-body semantic similarity: {e}")
        return None

    @staticmethod
    def _generate_templated_summary(signals: Dict[str, Any], red_flags: List[str]) -> str:
        """Assembles factual findings and red flags into an interpretable diagnosis."""
        lines = []
        if red_flags:
            lines.append(f"**Linguistic Red Flags Detected ({len(red_flags)})**:")
            for rf in red_flags:
                lines.append(f"- {rf}")
        else:
            lines.append("**Linguistic Quality**: No severe stylistic red flags detected; text exhibits standard journalistic formatting.")

        sens = signals["sensationalism"]
        attr = signals["attribution"]
        if attr["level"] == "High":
            lines.append(f"- Source Attribution Anchor: High direct citation frequency ({attr['score']}%).")
        else:
            lines.append(f"- Attribution Deficit: Low direct attribution density ({attr['score']}%).")

        return "\n".join(lines)
