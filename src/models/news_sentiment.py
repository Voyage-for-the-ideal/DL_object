"""FinBERT-Chinese + LoRA dual-task news sentiment model.

Base: yiyanghkust/finbert-tone-chinese
LoRA: r=16, alpha=32, target: query/key/value
Dual heads: sentiment classification (3-class) + intensity regression (-1 to 1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class FinBERTSentimentModel(nn.Module):
    """FinBERT with dual-task heads and optional LoRA."""

    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone-chinese",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_lora: bool = True,
        max_length: int = 256,
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=["query", "key", "value"],
                )
                self.bert = get_peft_model(self.bert, lora_config)
            except ImportError:
                print("Warning: peft not installed, skipping LoRA")

        hidden_dim = self.bert.config.hidden_size
        self.classifier_head = nn.Linear(hidden_dim, 3)  # pos/neu/neg
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        sentiment_logits = self.classifier_head(pooled)  # [B, 3]
        intensity = torch.tanh(self.intensity_head(pooled)).squeeze(-1)  # [B]
        return sentiment_logits, intensity

    def predict(self, texts: list[str], device: str = "cpu") -> np.ndarray:
        """Predict sentiment for a list of texts.

        Returns:
            np.ndarray of shape [len(texts), 6]:
            [sentiment_class (0/1/2), prob_pos, prob_neu, prob_neg, intensity (-1 to 1)]
        """
        self.eval()
        self.to(device)
        results = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                logits, intensity = self.forward(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1)
                batch_results = torch.cat([
                    pred_class.unsqueeze(-1).float(),
                    probs,
                    intensity.unsqueeze(-1),
                ], dim=-1)
                results.append(batch_results.cpu().numpy())

        return np.concatenate(results, axis=0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls, path: str | Path, model_name: str = "yiyanghkust/finbert-tone-chinese"
    ) -> "FinBERTSentimentModel":
        instance = cls(model_name=model_name, use_lora=False)
        instance.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        return instance
