"""
CAGE-Fake: spaCy NER Entity, Number & Date Consistency Matcher
Performs fine-grained symbolic comparison between claim assertions and retrieved evidence.
Flags discrepancies in numeric figures, monetary sums, dates, and named entities.
"""

from typing import Dict, Any, List, Set, Tuple
import re
import spacy

from src.utils import setup_logger

logger = setup_logger("entity_matcher")


class EntityNumberMatcher:
    """
    Extracts named entities, cardinal numbers, monetary figures, and calendar dates
    from claim and evidence sentences, and evaluates cross-consistency.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except Exception:
            self.nlp = spacy.blank("en")
            if "ner" not in self.nlp.pipe_names:
                # Add basic entities if possible
                pass

    def _extract_elements(self, text: str) -> Dict[str, Set[str]]:
        doc = self.nlp(text)
        numbers = set()
        dates = set()
        entities = set()

        if hasattr(doc, "ents"):
            for ent in doc.ents:
                clean_ent = ent.text.strip().lower()
                if ent.label_ in ("CARDINAL", "MONEY", "PERCENT", "QUANTITY"):
                    numbers.add(clean_ent)
                elif ent.label_ in ("DATE", "TIME"):
                    dates.add(clean_ent)
                elif ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "NORP"):
                    entities.add(clean_ent)

        # Regex fallback for numbers and currency amounts
        num_matches = re.findall(r"\b(?:rs\.?|inr|\$|€|£)?\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:crore|lakh|million|billion|thousand))?\b", text.lower())
        for m in num_matches:
            if any(char.isdigit() for char in m):
                numbers.add(m.strip())

        return {
            "numbers": numbers,
            "dates": dates,
            "entities": entities,
        }

    def check_consistency(self, claim: str, evidence: str) -> Dict[str, Any]:
        """
        Compares claim and evidence elements.
        Detects conflicts if claim specifies numbers or dates that are contradicted in the evidence.
        """
        c_elem = self._extract_elements(claim)
        e_elem = self._extract_elements(evidence)

        conflicts = []
        has_num_conflict = False
        has_date_conflict = False

        # Entity overlap (Jaccard similarity on named entities)
        inter_ents = c_elem["entities"].intersection(e_elem["entities"])
        union_ents = c_elem["entities"].union(e_elem["entities"])
        overlap_score = len(inter_ents) / len(union_ents) if union_ents else 1.0

        # Check numeric discrepancies
        if c_elem["numbers"] and e_elem["numbers"]:
            # If evidence mentions different numbers on the same topic, check for direct clash
            common_nums = c_elem["numbers"].intersection(e_elem["numbers"])
            if not common_nums and len(inter_ents) >= 1:
                # Subject matches but all numbers differ
                has_num_conflict = True
                conflicts.append(f"Number mismatch: Claim cites {list(c_elem['numbers'])}, but Evidence cites {list(e_elem['numbers'])}")

        # Check date discrepancies
        if c_elem["dates"] and e_elem["dates"]:
            common_dates = c_elem["dates"].intersection(e_elem["dates"])
            if not common_dates and len(inter_ents) >= 1:
                has_date_conflict = True
                conflicts.append(f"Date mismatch: Claim mentions {list(c_elem['dates'])}, but Evidence mentions {list(e_elem['dates'])}")

        return {
            "entity_overlap_score": round(overlap_score, 3),
            "common_entities": list(inter_ents),
            "claim_entities": list(c_elem["entities"]),
            "evidence_entities": list(e_elem["entities"]),
            "has_numeric_conflict": has_num_conflict,
            "has_date_conflict": has_date_conflict,
            "conflicts": conflicts,
            "is_consistent": (not has_num_conflict and not has_date_conflict),
        }
