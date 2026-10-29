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


# Common locations to supplement spaCy NER
KNOWN_LOCATIONS = {
    "agra", "kasganj", "lucknow", "delhi", "mumbai", "kolkata", "chennai", "bengaluru",
    "hyderabad", "ahmedabad", "varanasi", "kanpur", "aligarh", "mathura", "meerut",
    "ghaziabad", "noida", "gurugram", "patna", "jaipur", "bhopal", "indore", "surat",
    "pune", "nagpur", "prayagraj", "allahabad", "uttar pradesh", "bihar", "maharashtra",
    "rajasthan", "punjab", "haryana", "gujarat", "karnataka", "tamil nadu", "kerala",
    "washington", "london", "new york", "paris", "tokyo", "beijing", "moscow",
}


class EntityNumberMatcher:
    """
    Extracts named entities, cardinal numbers, monetary figures, calendar dates,
    and geographical locations (GPE/LOC) from claim and evidence sentences, and evaluates cross-consistency.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except Exception:
            self.nlp = spacy.blank("en")
            if "ner" not in self.nlp.pipe_names:
                pass

    def _extract_elements(self, text: str) -> Dict[str, Set[str]]:
        doc = self.nlp(text)
        numbers = set()
        dates = set()
        entities = set()
        locations = set()

        if hasattr(doc, "ents"):
            for ent in doc.ents:
                clean_ent = ent.text.strip().lower()
                if ent.label_ in ("CARDINAL", "MONEY", "PERCENT", "QUANTITY"):
                    numbers.add(clean_ent)
                elif ent.label_ in ("DATE", "TIME"):
                    dates.add(clean_ent)
                elif ent.label_ in ("GPE", "LOC"):
                    locations.add(clean_ent)
                    entities.add(clean_ent)
                elif ent.label_ in ("PERSON", "ORG", "NORP"):
                    # Check if misclassified location
                    if clean_ent in KNOWN_LOCATIONS:
                        locations.add(clean_ent)
                    entities.add(clean_ent)

        # Keyword / regex fallback for known places
        text_lower = text.lower()
        for loc in KNOWN_LOCATIONS:
            if re.search(r"\b" + re.escape(loc) + r"\b", text_lower):
                locations.add(loc)
                entities.add(loc)

        # Regex fallback for numbers and currency amounts
        num_matches = re.findall(r"\b(?:rs\.?|inr|\$|€|£)?\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:crore|lakh|million|billion|thousand))?\b", text.lower())
        for m in num_matches:
            if any(char.isdigit() for char in m):
                numbers.add(m.strip())

        return {
            "numbers": numbers,
            "dates": dates,
            "entities": entities,
            "locations": locations,
        }

    def check_consistency(self, claim: str, evidence: str, is_trusted: bool = False) -> Dict[str, Any]:
        """
        Compares claim and evidence elements.
        Detects location mismatches, entity overlap, and numeric/date clashes.
        If locations clash -> location_mismatch=True -> caller must mark Unverified.
        """
        c_elem = self._extract_elements(claim)
        e_elem = self._extract_elements(evidence)

        conflicts = []
        has_num_conflict = False
        has_date_conflict = False

        # Entity overlap (Jaccard similarity on named entities)
        inter_ents = c_elem["entities"].intersection(e_elem["entities"])
        union_ents = c_elem["entities"].union(e_elem["entities"])
        overlap_score = len(inter_ents) / len(union_ents) if union_ents else 0.0

        # Location consistency check
        claim_locs = c_elem["locations"]
        ev_locs = e_elem["locations"]
        location_mismatch = False

        if claim_locs and ev_locs:
            # If both claim and evidence name places and NONE overlap
            loc_overlap = any(
                c_loc in e_loc or e_loc in c_loc
                for c_loc in claim_locs
                for e_loc in ev_locs
            )
            if not loc_overlap:
                location_mismatch = True
                conflicts.append(
                    f"Location mismatch: Claim names {sorted(claim_locs)} but Evidence names {sorted(ev_locs)}"
                )

        same_event = bool(overlap_score >= 0.50 and not location_mismatch)

        # Check numeric discrepancies - only mark Contradicted when same_event and source is trusted
        if c_elem["numbers"] and e_elem["numbers"]:
            common_nums = c_elem["numbers"].intersection(e_elem["numbers"])
            if not common_nums and same_event and is_trusted:
                has_num_conflict = True
                conflicts.append(f"Number mismatch on same event: Claim cites {sorted(c_elem['numbers'])}, but Evidence cites {sorted(e_elem['numbers'])}")

        # Check date discrepancies - only mark Contradicted when same_event and source is trusted
        if c_elem["dates"] and e_elem["dates"]:
            common_dates = c_elem["dates"].intersection(e_elem["dates"])
            if not common_dates and same_event and is_trusted:
                has_date_conflict = True
                conflicts.append(f"Date mismatch on same event: Claim mentions {sorted(c_elem['dates'])}, but Evidence mentions {sorted(e_elem['dates'])}")

        return {
            "entity_overlap_score": round(overlap_score, 3),
            "common_entities": list(inter_ents),
            "claim_entities": list(c_elem["entities"]),
            "evidence_entities": list(e_elem["entities"]),
            "claim_locations": list(claim_locs),
            "evidence_locations": list(ev_locs),
            "location_mismatch": location_mismatch,
            "same_event": same_event,
            "has_numeric_conflict": has_num_conflict,
            "has_date_conflict": has_date_conflict,
            "conflicts": conflicts,
            "is_consistent": (not has_num_conflict and not has_date_conflict and not location_mismatch),
        }
