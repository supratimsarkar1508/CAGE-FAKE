"""
CAGE-Fake: Rule-Based NLP Linguistic Signal Summary Module
Generates fully deterministic, evidence-grounded linguistic analysis from computed values:
- Sensationalism & hyperbole
- Epistemic hedging
- Attribution verb density (source grounding)
- Emotional language & intensifiers
- Punctuation intensity (shouting / clickbait indicators)
- Cognitive readability
- Lexical diversity (TTR)
- Tense consistency
- Named entity / Proper noun density
- Direct quotation density

NO LLM HALLUCINATION: All observations are mathematically derived from verified text signals.
"""

from typing import Dict, Any, List
import re
import string

SENSATIONAL_WORDS = {
    "shocking", "bombshell", "unbelievable", "devastating", "mind-blowing", "exposed",
    "secretly", "uncovered", "horrific", "explosive", "miraculous", "stunning", "panic",
    "insane", "disaster", "treason", "conspiracy", "scandal", "apocalypse", "nightmare"
}

EMOTIONAL_WORDS = {
    "outrageous", "disgusting", "vicious", "corrupt", "evil", "heroic", "furious",
    "threat", "danger", "frightening", "disgrace", "shameful", "sinister", "pure",
    "monstrous", "pathetic", "hero", "villain", "toxic", "destroy", "annihilate"
}

ATTRIBUTION_VERBS = {
    "said", "stated", "confirmed", "reported", "announced", "declared", "explained",
    "noted", "remarked", "testified", "admitted", "acknowledged", "cited", "asserted",
    "concluded", "emphasized", "insisted", "indicated", "revealed", "told", "warned"
}

HEDGE_WORDS = {
    "alleged", "allegedly", "apparently", "reportedly", "purportedly", "supposedly",
    "rumor", "rumored", "rumours", "claims", "claimed", "claiming", "suggests", "suggested",
    "could", "might", "maybe", "perhaps", "possibly", "possible", "potential",
    "seem", "seems", "seemed", "appear", "appears", "speculate", "assumed"
}

WORD_REGEX = re.compile(r"\b[a-zA-Z0-9_\'-]+\b")
SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+")

class NLPLinguisticExplainer:
    """
    Computes 10 linguistic dimensions and formats them into an interpretable diagnostic report.
    """

    def analyze(self, raw_text: str, clean_text: str) -> Dict[str, Any]:
        """Analyzes text and returns dimensional scores and synthesized diagnostic text."""
        raw = str(raw_text or "")
        clean = str(clean_text or "")

        total_chars = max(1, len(raw))
        tokens = [w.lower() for w in WORD_REGEX.findall(clean)]
        total_tokens = max(1, len(tokens))
        sentences = [s.strip() for s in SENTENCE_REGEX.split(raw) if s.strip()]
        total_sentences = max(1, len(sentences))

        # 1. Sensationalism
        sensational_hits = [w for w in tokens if w in SENSATIONAL_WORDS]
        sensationalism_density = len(sensational_hits) / total_tokens * 100.0

        # 2. Epistemic Hedging
        hedge_hits = [w for w in tokens if w in HEDGE_WORDS]
        hedging_density = len(hedge_hits) / total_tokens * 100.0

        # 3. Attribution Density
        attr_hits = [w for w in tokens if w in ATTRIBUTION_VERBS]
        attr_density = len(attr_hits) / total_tokens * 100.0

        # 4. Emotional Language
        emotional_hits = [w for w in tokens if w in EMOTIONAL_WORDS]
        emotional_density = len(emotional_hits) / total_tokens * 100.0

        # 5. Punctuation Intensity
        excl_count = raw.count("!")
        quest_count = raw.count("?")
        caps_tokens = sum(1 for tok in raw.split() if tok.isupper() and len(tok) > 1)
        punct_intensity_score = (excl_count * 2.0 + quest_count * 1.5 + caps_tokens) / max(1, total_sentences)

        # 6. Lexical Diversity
        unique_tokens = len(set(tokens))
        ttr = unique_tokens / total_tokens

        # 7. Readability
        words_per_sent = total_tokens / total_sentences
        readability_level = "Complex/Academic" if words_per_sent > 25 else ("Standard" if words_per_sent > 14 else "Simple/Conversational")

        # 8. Tense Consistency
        past_verbs = sum(1 for w in tokens if w.endswith("ed") or w in {"was", "were", "had", "said", "did"})
        present_verbs = sum(1 for w in tokens if w.endswith("ing") or w in {"is", "are", "has", "says", "does"})
        total_tense = past_verbs + present_verbs
        tense_consistency = (max(past_verbs, present_verbs) / total_tense) if total_tense > 0 else 1.0

        # 9. Proper Noun / Named Entity Density
        capitalized_words = sum(1 for w in raw.split() if w and w[0].isupper() and not w.isupper())
        named_entity_density = capitalized_words / total_tokens * 100.0

        # 10. Direct Quotation Density
        quote_chars = raw.count('"') + raw.count("'") + raw.count("“") + raw.count("”")
        quote_density = quote_chars / total_chars * 100.0

        signals = {
            "sensationalism": {
                "score": round(sensationalism_density, 2),
                "level": "High" if sensationalism_density > 1.2 else ("Moderate" if sensationalism_density > 0.4 else "Low"),
                "examples": list(set(sensational_hits))[:4],
            },
            "hedging": {
                "score": round(hedging_density, 2),
                "level": "High" if hedging_density > 1.5 else ("Moderate" if hedging_density > 0.6 else "Low"),
                "examples": list(set(hedge_hits))[:4],
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
        }

        summary_text = self._generate_templated_summary(signals)
        return {"signals": signals, "summary": summary_text}

    @staticmethod
    def _generate_templated_summary(signals: Dict[str, Any]) -> str:
        """Assembles factual findings into an evidence-grounded diagnosis."""
        lines = []
        sens = signals["sensationalism"]
        attr = signals["attribution"]
        emot = signals["emotional_language"]
        punct = signals["punctuation_intensity"]
        hedge = signals["hedging"]

        if sens["level"] == "High" or emot["level"] == "High":
            lines.append(
                f"- **Sensational / Affective Tone Detected**: The article exhibits elevated emotional resonance "
                f"(emotional density {emot['score']}%, sensationalism {sens['score']}%). "
                f"Flagged terms: {', '.join(sens['examples'] or emot['examples'])}."
            )
        else:
            lines.append("- **Stylistic Tone**: Neutral journalistic presentation with constrained sensationalism.")

        if attr["level"] == "High":
            lines.append(
                f"- **Source Attribution Anchor**: Strong grounding with verified attribution markers "
                f"({attr['score']}% density; verbs: {', '.join(attr['examples'])})."
            )
        else:
            lines.append(
                f"- **Attribution Deficit**: Very low direct citation of sources ({attr['score']}% attribution density), "
                f"characteristic of uncorroborated commentary."
            )

        if punct["level"] == "High":
            lines.append(f"- **Punctuation Anomalies**: Noteworthy punctuation intensity ({punct['details']}).")

        if hedge["level"] == "High":
            lines.append(f"- **Speculative Hedging**: Elevated use of epistemic hedging ({hedge['score']}% density; {', '.join(hedge['examples'])}).")

        return "\n".join(lines)
