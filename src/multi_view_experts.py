"""
CAGE-Fake: Multi-View Expert Ensemble
Replaces homogeneous SVD representations with three orthogonal domain-specific experts:
1. Lexical Expert: Logistic Regression on raw sparse TF-IDF (untruncated vocabulary)
2. Style Expert: XGBoost on 19 named stylometric & readability features (SHAP TreeExplainer)
3. Semantic Expert: Fine-tuned DistilRoBERTa / MuRIL Transformer (Integrated Gradients)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score
import xgboost as xgb
import shap
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import setup_logger, save_artifact, load_artifact

logger = setup_logger("multi_view_experts")


# =========================================================================
# 1. LEXICAL EXPERT (Raw Sparse TF-IDF + Logistic Regression)
# =========================================================================

class LexicalExpert:
    """
    Expert specialized strictly in sparse lexical vocabulary distributions.
    Directly evaluates raw uncompressed n-gram TF-IDF matrices without SVD compression.
    Explains predictions using linear coefficient-tfidf attribution: coef_i * tfidf_i.
    """

    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.model = LogisticRegression(C=C, max_iter=1000, random_state=random_state)
        self.feature_names: List[str] = []

    def fit(self, X_sparse, y, feature_names: Optional[List[str]] = None) -> "LexicalExpert":
        logger.info(f"Training Lexical Expert on raw sparse TF-IDF matrix (shape: {X_sparse.shape})...")
        self.model.fit(X_sparse, y)
        if feature_names is not None:
            self.feature_names = list(feature_names)
        logger.info("Lexical Expert training complete.")
        return self

    def predict_proba(self, X_sparse) -> np.ndarray:
        probas = self.model.predict_proba(X_sparse)
        return probas[:, 1]  # P(FAKE)

    def explain_instance(self, x_sparse_vec, top_k: int = 5) -> Dict[str, Any]:
        """
        Computes token-level linear attribution: contribution = w_i * x_i.
        Positive contribution drives toward FAKE; negative drives toward REAL.
        """
        p_fake = float(self.predict_proba(x_sparse_vec)[0])
        coefs = self.model.coef_[0]

        # Convert sparse vector to dictionary of active features
        if issparse(x_sparse_vec):
            cx = x_sparse_vec.tocoo()
            indices = cx.col
            values = cx.data
        else:
            arr = np.array(x_sparse_vec).flatten()
            indices = np.where(arr != 0)[0]
            values = arr[indices]

        contributions = []
        for idx, val in zip(indices, values):
            if idx < len(coefs):
                name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                impact = float(coefs[idx] * val)
                contributions.append({"token": name, "impact": impact})

        # Sort by magnitude of contribution
        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top_tokens = contributions[:top_k]

        return {
            "expert": "Lexical (TF-IDF + LR)",
            "p_fake": round(p_fake, 4),
            "top_features": top_tokens,
            "why": [f"'{t['token']}' ({t['impact']:+.3f})" for t in top_tokens],
        }


# =========================================================================
# 2. STYLE EXPERT (15 Stylometric + 4 Readability Features + XGBoost)
# =========================================================================

class StyleExpert:
    """
    Expert specialized strictly in writing style, syntax, and readability.
    Trains gradient boosted trees on 19 named stylometric metrics with early stopping.
    Explains predictions using exact SHAP TreeExplainer attributions.
    """

    def __init__(self, feature_names: List[str], random_state: int = 42):
        self.feature_names = list(feature_names)
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
        )
        self.explainer: Optional[shap.TreeExplainer] = None
        self.evals_result: Dict[str, Any] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, early_stopping_rounds: int = 50) -> "StyleExpert":
        logger.info(f"Training Style Expert on {len(self.feature_names)} named stylometric features...")
        eval_set = [(X_train, y_train), (X_val, y_val)]
        self.model.set_params(early_stopping_rounds=early_stopping_rounds)
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.evals_result = self.model.evals_result()
        logger.info("Fitting SHAP TreeExplainer on trained Style Expert...")
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("Style Expert training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = self.model.predict_proba(X)
        return probas[:, 1]

    def explain_instance(self, x_vec: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        p_fake = float(self.predict_proba(x_vec.reshape(1, -1))[0])
        shap_values = self.explainer.shap_values(x_vec.reshape(1, -1))
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # Positive class (FAKE)
        else:
            sv = shap_values[0]

        items = []
        for name, val, s in zip(self.feature_names, x_vec.flatten(), sv):
            items.append({
                "feature": name,
                "value": round(float(val), 4),
                "shap_impact": round(float(s), 4),
            })

        items.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
        top_items = items[:top_k]

        return {
            "expert": "Style (Stylometric & Readability XGBoost)",
            "p_fake": round(p_fake, 4),
            "top_features": top_items,
            "why": [f"{item['feature']} ({item['shap_impact']:+.3f})" for item in top_items],
        }

    def plot_training_curves(self, out_pdf: Path, out_png: Optional[Path] = None) -> None:
        """Plots validation logloss per boosting round and saves as PDF and PNG."""
        if not self.evals_result:
            return
        train_loss = self.evals_result.get("validation_0", {}).get("logloss", [])
        val_loss = self.evals_result.get("validation_1", {}).get("logloss", [])

        fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)
        ax.plot(train_loss, label="Training Log-Loss", color="#3b82f6", linewidth=1.5)
        ax.plot(val_loss, label="Validation Log-Loss", color="#ef4444", linewidth=1.5)
        ax.set_title("Style Expert (XGBoost) Training Curves with Early Stopping", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Boosting Round", fontsize=9)
        ax.set_ylabel("Binary Log-Loss", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_pdf), format="pdf", bbox_inches="tight")
        if out_png:
            fig.savefig(str(out_png), format="png", bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved Style Expert training curves to {out_pdf}")


# =========================================================================
# 3. SEMANTIC EXPERT (Fine-Tuned DistilRoBERTa / MuRIL Transformer)
# =========================================================================

class HeadTailDataset(Dataset):
    """Encodes texts using head+tail truncation: first 128 + last 384 tokens."""

    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer, max_len: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_len - 2:
            head = tokens[:128]
            tail = tokens[-(self.max_len - 2 - 128):]
            tokens = head + tail

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        padding_len = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_len
        attention_mask += [0] * padding_len

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class SemanticExpert:
    """
    Expert specialized strictly in contextual semantics and deep narrative flow.
    Fine-tunes DistilRoBERTa (or MuRIL) with head+tail truncation and early stopping.
    Explains predictions using token-level Integrated Gradients / Embedding Attributions.
    """

    def __init__(self, model_name: str = "distilroberta-base", max_len: int = 512):
        self.model_name = model_name
        self.max_len = max_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.to(self.device)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_f1s: List[float] = []

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 16,
    ) -> "SemanticExpert":
        logger.info(f"Fine-tuning Semantic Expert ({self.model_name}) on {self.device}...")
        train_dataset = HeadTailDataset(train_texts, train_labels, self.tokenizer, self.max_len)
        val_dataset = HeadTailDataset(val_texts, val_labels, self.tokenizer, self.max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        use_amp = (self.device == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val_f1 = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, len(train_loader))
            self.train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            total_val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                    total_val_loss += loss.item()

                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(labels.cpu().numpy())

            avg_val_loss = total_val_loss / max(1, len(val_loader))
            val_f1 = float(f1_score(all_targets, all_preds, average="macro"))
            self.val_losses.append(avg_val_loss)
            self.val_f1s.append(val_f1)

            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
            logger.info(f"Loaded best checkpoint with Val Macro-F1: {best_val_f1:.4f}")

        return self

    def predict_proba(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        self.model.eval()
        dataset = HeadTailDataset(texts, None, self.tokenizer, self.max_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        probas = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                probas.extend(probs)
        return np.array(probas, dtype=np.float32)

    def explain_instance(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Token-level gradient attribution (Integrated Gradients approximation).
        Identifies key tokens shifting the transformer attention toward FAKE.
        """
        self.model.eval()
        p_fake = float(self.predict_proba([text])[0])

        # Tokenize
        tokens = self.tokenizer.tokenize(text)[:self.max_len - 2]
        if not tokens:
            return {"expert": f"Semantic ({self.model_name})", "p_fake": round(p_fake, 4), "top_features": [], "why": []}

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([[self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]]).to(self.device)
        embed_layer = self.model.get_input_embeddings()
        embeddings = embed_layer(input_ids)
        embeddings.retain_grad()

        outputs = self.model(inputs_embeds=embeddings)
        fake_logit = outputs.logits[0, 1]
        fake_logit.backward()

        if embeddings.grad is not None:
            grads = embeddings.grad[0].cpu().numpy()
            # Gradient x Embedding attribution magnitude
            token_scores = np.sum(grads * embeddings[0].detach().cpu().numpy(), axis=1)
            # Exclude [CLS] and [SEP]
            token_scores = token_scores[1:-1]
        else:
            token_scores = np.zeros(len(tokens))

        scored_tokens = []
        for t, s in zip(tokens, token_scores):
            clean_t = t.replace("Ġ", "").strip()
            if len(clean_t) > 2 and clean_t.isalnum():
                scored_tokens.append({"token": clean_t, "attribution": float(s)})

        scored_tokens.sort(key=lambda x: abs(x["attribution"]), reverse=True)
        top_tokens = scored_tokens[:top_k]

        return {
            "expert": f"Semantic ({self.model_name})",
            "p_fake": round(p_fake, 4),
            "top_features": top_tokens,
            "why": [f"'{t['token']}' ({t['attribution']:+.3f})" for t in top_tokens],
        }

    def plot_training_curves(self, out_pdf: Path, out_png: Optional[Path] = None) -> None:
        if not self.train_losses:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), dpi=300)
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, "b-o", label="Train Loss")
        ax1.plot(epochs, self.val_losses, "r-o", label="Val Loss")
        ax1.set_title("Semantic Expert Loss Curves", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=9)
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=9)
        ax1.legend(fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.5)

        ax2.plot(epochs, self.val_f1s, "g-s", label="Val Macro-F1")
        ax2.set_title("Semantic Expert Validation Macro-F1", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=9)
        ax2.set_ylabel("Macro-F1", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.5)

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_pdf), format="pdf", bbox_inches="tight")
        if out_png:
            fig.savefig(str(out_png), format="png", bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved Semantic Expert training curves to {out_pdf}")


# =========================================================================
# 4. MULTI-VIEW EXPERT ORCHESTRATOR
# =========================================================================

class MultiViewEnsemble:
    """
    Coordinates Lexical, Style, and Semantic experts.
    Computes view-disagreement delta = std([p_lexical, p_style, p_semantic]).
    """

    def __init__(
        self,
        lexical_expert: LexicalExpert,
        style_expert: StyleExpert,
        semantic_expert: SemanticExpert,
    ):
        self.lexical = lexical_expert
        self.style = style_expert
        self.semantic = semantic_expert

    def evaluate_instance(
        self,
        raw_text: str,
        x_sparse_tf: Any,
        x_style_vec: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Runs all three experts on the input article and returns probabilities and attributions.
        """
        lex_exp = self.lexical.explain_instance(x_sparse_tf)
        style_exp = self.style.explain_instance(x_style_vec)
        sem_exp = self.semantic.explain_instance(raw_text)

        p_lex = lex_exp["p_fake"]
        p_sty = style_exp["p_fake"]
        p_sem = sem_exp["p_fake"]

        probs = [p_lex, p_sty, p_sem]
        view_disagreement = float(np.std(probs))
        mean_p_fake = float(np.mean(probs))

        return {
            "p_lexical": p_lex,
            "p_style": p_sty,
            "p_semantic": p_sem,
            "mean_p_fake": round(mean_p_fake, 4),
            "view_disagreement": round(view_disagreement, 4),
            "lexical_explanation": lex_exp,
            "style_explanation": style_exp,
            "semantic_explanation": sem_exp,
            "experts": [
                {"name": "Lexical Expert (TF-IDF + LR)", "p_fake": p_lex, "why": lex_exp["why"]},
                {"name": "Style Expert (Stylometrics + XGBoost)", "p_fake": p_sty, "why": style_exp["why"]},
                {"name": "Semantic Expert (Contextual Transformer)", "p_fake": p_sem, "why": sem_exp["why"]},
            ],
        }
