import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer
from src.utils.model_factory import BaseModelTrainer
from typing import Dict
import sklearn.metrics
import re
from tqdm import tqdm

class DistilBertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels: int, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)
        self.num_labels = num_labels
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Используем [CLS] токен для классификации
        hidden_state = outputs.last_hidden_state[:, 0]
        hidden_state = self.dropout(hidden_state)
        logits = self.classifier(hidden_state)
        
        return logits

class DistilBertFTTrainer(BaseModelTrainer):
    tokenizer: DistilBertTokenizer
    model: DistilBertForMultiLabelClassification

    def __init__(self, config, tags, logger=None):
        super().__init__(config, logger)
        self.tags = tags

    def _train_epoch(self, dataloader, epoch) -> Dict[str, float]:
        self.model.train()

        if not hasattr(self, "optimizer"):
            self.create_optimizer()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for x, y in tqdm(dataloader, desc="Training progress", leave=False):
            encoded_inputs = self.preprocess_data(x)
            outputs: torch.Tensor = self.model(**encoded_inputs)

            self.model.zero_grad()
            loss = F.binary_cross_entropy_with_logits(outputs, y.float(), reduction="sum")
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach())
            total_correct += ((F.sigmoid(outputs) > self.config.threshold) == y).sum()
            total_samples += torch.numel(y)

        return {"avg_loss": total_loss / len(dataloader.dataset), "accuracy": total_correct / total_samples}
    
    def _validate_epoch(self, dataloader, epoch) -> Dict[str, float]:
        self.model.eval()
        y_test = torch.empty([0])
        y_pred = torch.empty([0])
        total_loss = 0.0

        for x, y in tqdm(dataloader, desc="Validation progress", leave=False):
            encoded_inputs = self.preprocess_data(x)
            outputs: torch.Tensor = self.model(**encoded_inputs)

            loss = F.binary_cross_entropy_with_logits(outputs, y.float(), reduction="sum")

            total_loss += float(loss.detach())
            y_test = torch.cat([y_test, y])
            y_pred = torch.cat([y_pred, (F.sigmoid(outputs) > self.config.threshold)])

        accuracy = float(1 - sklearn.metrics.hamming_loss(y_test, y_pred))
        clf_report = sklearn.metrics.classification_report(
            y_test, y_pred, 
            output_dict=True, 
            target_names=self.tags,
            zero_division=1
        )
        clf_report = {
            re.sub(r"[^0-9A-Za-z_]", "", "_".join([cls, key]).replace(" ", "_")): val
            for cls, metrics in clf_report.items() for key, val in metrics.items() # type: ignore
        }
        return {"avg_loss": total_loss / len(dataloader.dataset), "accuracy": accuracy, **clf_report}

    def create_model(self):
        assert self.config.get("num_labels") or self.config.get("tags")
        num_labels = self.config.get("num_labels") or len(self.config.get("tags"))
        return DistilBertForMultiLabelClassification(
            num_labels=num_labels, model_name=self.config.model_name)
    
    def preprocess_data(self, data):
        if not hasattr(self, "tokenizer"):
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.config.model_name)
        return self.tokenizer(
            data,
            truncation=True,
            padding='longest',
            max_length=512,
            return_tensors='pt'
        )
    
    def create_optimizer(self):
        self.optimizer = torch.optim.Adam([
            {'params': self.model.classifier.parameters(), 'lr': self.config.get("classifier_learning_rate")},
            {'params': self.model.distilbert.parameters(), 'lr': self.config.get("distilbert_learning_rate")}
        ])
    
    def _log_base_params(self):
        super()._log_base_params()
        self.logger.log_params({
            "threshold": self.config.get("threshold", 0.5),
            "tags": self.tags,
            'classifier_lr': self.config.get("classifier_learning_rate"),
            'base_lr': self.config.get("distilbert_learning_rate")
        })