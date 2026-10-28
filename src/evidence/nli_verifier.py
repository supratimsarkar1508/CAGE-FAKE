"""
CAGE-Fake: NLI-Based Evidence Verifier
Uses Natural Language Inference (NLI) models (such as MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
or cross-encoder/nli-distilroberta-base) to classify (claim, evidence) pairs into
Entailment (Supported), Contradiction (Contradicted), or Neutral (Unverified).
"""

from typing import Dict, Any, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import setup_logger

logger = setup_logger("nli_verifier")

DEBUNK_KEYWORDS = {"false", "fake", "hoax", "misleading", "pants on fire", "unproven", "fabricated", "incorrect"}
VERIFIED_KEYWORDS = {"true", "correct", "accurate", "verified", "authentic"}


class NLIVerifier:
    """
    Performs rigorous Natural Language Inference (NLI) to ascertain whether
    an authoritative evidence sentence logically entails or refutes a given factual claim.
    """

    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self._model is None:
            try:
                logger.info(f"Loading NLI model: {self.model_name} on {self.device}...")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()
            except Exception as e:
                logger.warning(f"Failed to load {self.model_name}: {e}. Trying lightweight cross-encoder/nli-distilroberta-base fallback.")
                fallback = "cross-encoder/nli-distilroberta-base"
                self._tokenizer = AutoTokenizer.from_pretrained(fallback)
                self._model = AutoModelForSequenceClassification.from_pretrained(fallback)
                self._model.to(self.device)
                self._model.eval()

    def verify_pair(self, claim: str, evidence_sentence: str, factcheck_rating: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs NLI forward pass on (premise=evidence_sentence, hypothesis=claim).
        Returns:
        {
            'status': 'Supported' | 'Contradicted' | 'Unverified',
            'entailment_prob': float,
            'contradiction_prob': float,
            'neutral_prob': float,
            'confidence': float,
        }
        """
        # 1. Fact-check rating shortcut if from verified fact-checking authority
        if factcheck_rating:
            rating_lower = str(factcheck_rating).lower()
            if any(k in rating_lower for k in DEBUNK_KEYWORDS):
                return {
                    "status": "Contradicted",
                    "entailment_prob": 0.05,
                    "contradiction_prob": 0.95,
                    "neutral_prob": 0.0,
                    "confidence": 0.95,
                    "source_cue": f"Fact-Check Verdict: {factcheck_rating}",
                }
            if any(k in rating_lower for k in VERIFIED_KEYWORDS):
                return {
                    "status": "Supported",
                    "entailment_prob": 0.95,
                    "contradiction_prob": 0.05,
                    "neutral_prob": 0.0,
                    "confidence": 0.95,
                    "source_cue": f"Fact-Check Verdict: {factcheck_rating}",
                }

        self._load_model()

        try:
            # Format: premise = evidence, hypothesis = claim
            inputs = self._tokenizer(
                evidence_sentence,
                claim,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

            # Map label index to entailment / neutral / contradiction
            # For MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli:
            # id2label usually: {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
            id2label = self._model.config.id2label
            prob_dict = {}
            for idx, prob in enumerate(probs):
                lbl = id2label.get(idx, str(idx)).lower()
                prob_dict[lbl] = float(prob)

            p_entail = prob_dict.get("entailment", 0.0)
            p_contra = prob_dict.get("contradiction", 0.0)
            p_neutral = prob_dict.get("neutral", 0.0)

            # Decision logic with confidence thresholds
            if p_contra >= 0.55:
                status = "Contradicted"
                conf = p_contra
            elif p_entail >= 0.55:
                status = "Supported"
                conf = p_entail
            else:
                status = "Unverified"
                conf = p_neutral

            return {
                "status": status,
                "entailment_prob": round(p_entail, 3),
                "contradiction_prob": round(p_contra, 3),
                "neutral_prob": round(p_neutral, 3),
                "confidence": round(conf, 3),
            }

        except Exception as e:
            logger.warning(f"NLI evaluation failed: {e}")
            return {
                "status": "Unverified",
                "entailment_prob": 0.33,
                "contradiction_prob": 0.33,
                "neutral_prob": 0.34,
                "confidence": 0.34,
            }
