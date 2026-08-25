"""
Classifier implementations for AI text detection.

Implements three classifier types:
1. Classical ML (Logistic Regression, Random Forest, Gradient Boosting)
2. Transformer-based classifier
3. Hybrid ensemble combining both approaches

Design principles:
- Proper train/validation/test separation
- No data leakage
- Support for calibration
- Uncertainty estimation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ClassifierConfig:
    """Configuration for classifiers."""
    # Overall type
    type: str = "hybrid"  # classical, transformer, hybrid
    
    # Classical classifier settings
    classical_type: str = "gradient_boosting"
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    
    # Transformer settings
    encoder_name: str = "roberta-base"
    num_labels: int = 2
    dropout: float = 0.1
    fine_tune: bool = True
    batch_size: int = 16
    num_epochs: int = 3
    
    # Ensemble settings
    ensemble_method: str = "stacking"  # voting, stacking, weighted_average
    ensemble_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "classical": 0.3,
                "transformer": 0.5,
                "statistical": 0.2
            }


class ClassicalClassifier:
    """
    Classical ML classifier using statistical and linguistic features.
    
    Supports:
    - Logistic Regression (fast, interpretable baseline)
    - Random Forest (robust, handles non-linear patterns)
    - Gradient Boosting (often best performance)
    """
    
    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.model = None
        self.feature_scaler = None
        self.is_fitted = False
        
    def _create_model(self):
        """Create the underlying ML model based on config."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.pipeline import Pipeline
        
        if self.config.classical_type == "logistic_regression":
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    C=1.0, 
                    max_iter=1000,
                    class_weight='balanced'
                ))
            ])
            
        elif self.config.classical_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            )
            
        elif self.config.classical_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
            
        else:
            raise ValueError(f"Unknown classical classifier type: {self.config.classical_type}")
        
        return self.model
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Train the classical classifier.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (0=human, 1=AI)
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            self._create_model()
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        metrics = {}
        train_score = self.model.score(X, y)
        metrics['train_accuracy'] = train_score
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            metrics['val_accuracy'] = val_score
            
        logger.info(f"Classical classifier trained. Train acc: {train_score:.4f}")
        if 'val_accuracy' in metrics:
            logger.info(f"Val acc: {metrics['val_accuracy']:.4f}")
            
        return metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        probas = self.model.predict_proba(X)
        return probas
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def save(self, path: Path):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config
            }, f)
        logger.info(f"Classical classifier saved to {path}")
    
    def load(self, path: Path):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.config = data.get('config', self.config)
            self.is_fitted = True
        logger.info(f"Classical classifier loaded from {path}")


class TransformerClassifier:
    """
    Transformer-based classifier using pretrained encoder fine-tuning.
    
    Uses a pretrained language model (RoBERTa, DeBERTa, etc.) with
    a classification head fine-tuned on AI detection task.
    """
    
    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_fitted = False
        
    def _setup_device(self):
        """Setup computing device (CPU/GPU)."""
        import torch
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info("Using GPU for transformer classifier")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU for transformer classifier")
    
    def _load_model(self):
        """Load pretrained transformer with classification head."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_name)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.encoder_name,
            num_labels=self.config.num_labels,
            hidden_dropout_prob=self.config.dropout
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded transformer: {self.config.encoder_name}")
    
    def fit(self, texts: List[str], labels: List[int],
            val_texts: Optional[List[str]] = None,
            val_labels: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Fine-tune transformer classifier.
        
        Args:
            texts: Training texts
            labels: Training labels (0=human, 1=AI)
            val_texts: Optional validation texts
            val_labels: Optional validation labels
            
        Returns:
            Dictionary with training metrics
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AdamW, get_linear_schedule_with_warmup
        
        self._setup_device()
        self._load_model()
        
        # Tokenize
        train_encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(labels)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=2e-5,
            weight_decay=0.01
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        best_val_acc = 0
        
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            
            for batch in train_loader:
                input_ids, attention_mask, label = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = label.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_texts is not None:
                val_metrics = self.evaluate(val_texts, val_labels)
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    
        self.is_fitted = True
        return {'train_loss': avg_train_loss, 'best_val_accuracy': best_val_acc}
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict class probabilities for texts."""
        import torch
        
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        self.model.eval()
        all_probas = []
        
        with torch.no_grad():
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probas = torch.softmax(logits, dim=1).cpu().numpy()
                all_probas.append(probas)
        
        return np.vstack(all_probas)
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(texts)
        return np.argmax(probas, axis=1)
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """Evaluate model on texts."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.predict(texts)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1': f1_score(labels, predictions)
        }
    
    def save(self, path: Path):
        """Save model to disk."""
        if self.model is not None:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            with open(path / 'classifier_config.pkl', 'wb') as f:
                pickle.dump({'config': self.config}, f)
                
            logger.info(f"Transformer classifier saved to {path}")
    
    def load(self, path: Path):
        """Load model from disk."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        self._setup_device()
        
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()
        
        with open(path / 'classifier_config.pkl', 'rb') as f:
            data = pickle.load(f)
            self.config = data.get('config', self.config)
            
        self.is_fitted = True
        logger.info(f"Transformer classifier loaded from {path}")


class HybridClassifier:
    """
    Hybrid ensemble classifier combining classical and transformer models.
    
    Uses stacking or weighted averaging to combine predictions from:
    - Classical ML on statistical/linguistic features
    - Transformer encoder on raw text
    
    Design rationale:
    - Different models capture different signals
    - Ensemble reduces over-reliance on any single approach
    - More robust to distribution shift
    """
    
    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.classical_clf = ClassicalClassifier(config)
        self.transformer_clf = TransformerClassifier(config)
        self.meta_classifier = None  # For stacking
        self.is_fitted = False
        
    def fit(self, texts: List[str], feature_vectors: np.ndarray, 
            labels: List[int],
            val_texts: Optional[List[str]] = None,
            val_features: Optional[np.ndarray] = None,
            val_labels: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Train hybrid classifier.
        
        Args:
            texts: Raw texts for transformer
            feature_vectors: Statistical/linguistic features for classical ML
            labels: Binary labels (0=human, 1=AI)
            val_*: Optional validation data
            
        Returns:
            Training metrics
        """
        logger.info("Training classical classifier...")
        classical_metrics = self.classical_clf.fit(feature_vectors, labels, 
                                                   X_val=val_features, 
                                                   y_val=val_labels)
        
        logger.info("Training transformer classifier...")
        transformer_metrics = self.transformer_clf.fit(
            texts, labels,
            val_texts=val_texts,
            val_labels=val_labels
        )
        
        # If using stacking, train meta-classifier on validation predictions
        if self.config.ensemble_method == "stacking" and val_texts is not None:
            self._train_meta_classifier(
                texts, feature_vectors, labels,
                val_texts, val_features, val_labels
            )
        
        self.is_fitted = True
        
        return {
            'classical': classical_metrics,
            'transformer': transformer_metrics
        }
    
    def _train_meta_classifier(self, train_texts, train_features, train_labels,
                               val_texts, val_features, val_labels):
        """Train meta-classifier for stacking ensemble."""
        from sklearn.linear_model import LogisticRegression
        
        # Get base model predictions on validation set
        classical_proba = self.classical_clf.predict_proba(val_features)[:, 1]
        transformer_proba = self.transformer_clf.predict_proba(val_texts)[:, 1]
        
        # Stack predictions as features
        meta_features = np.column_stack([classical_proba, transformer_proba])
        
        # Train simple meta-classifier
        self.meta_classifier = LogisticRegression()
        self.meta_classifier.fit(meta_features, val_labels)
        
        logger.info("Meta-classifier trained for stacking ensemble")
    
    def predict_proba(self, texts: List[str], 
                     feature_vectors: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using ensemble.
        
        Args:
            texts: Raw texts
            feature_vectors: Statistical/linguistic features
            
        Returns:
            Class probabilities (n_samples, 2)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Get predictions from each model
        classical_proba = self.classical_clf.predict_proba(feature_vectors)
        transformer_proba = self.transformer_clf.predict_proba(texts)
        
        if self.config.ensemble_method == "voting":
            # Simple average
            combined = (classical_proba + transformer_proba) / 2
            
        elif self.config.ensemble_method == "weighted_average":
            w1 = self.config.ensemble_weights.get('classical', 0.5)
            w2 = self.config.ensemble_weights.get('transformer', 0.5)
            combined = w1 * classical_proba + w2 * transformer_proba
            
        elif self.config.ensemble_method == "stacking":
            if self.meta_classifier is None:
                # Fall back to weighted average if no meta-classifier
                return self.predict_proba(texts, feature_vectors)
            
            # Use meta-classifier
            classical_scores = classical_proba[:, 1].reshape(-1, 1)
            transformer_scores = transformer_proba[:, 1].reshape(-1, 1)
            meta_features = np.hstack([classical_scores, transformer_scores])
            
            ai_scores = self.meta_classifier.predict_proba(meta_features)[:, 1]
            combined = np.column_stack([1 - ai_scores, ai_scores])
            
        else:
            combined = (classical_proba + transformer_proba) / 2
        
        return combined
    
    def predict(self, texts: List[str], feature_vectors: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(texts, feature_vectors)
        return np.argmax(probas, axis=1)
    
    def save(self, path: Path):
        """Save all components."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.classical_clf.save(path / 'classical_model.pkl')
        self.transformer_clf.save(path / 'transformer_model')
        
        if self.meta_classifier is not None:
            with open(path / 'meta_classifier.pkl', 'wb') as f:
                pickle.dump(self.meta_classifier, f)
        
        with open(path / 'hybrid_config.pkl', 'wb') as f:
            pickle.dump({'config': self.config}, f)
        
        logger.info(f"Hybrid classifier saved to {path}")
    
    def load(self, path: Path):
        """Load all components."""
        path = Path(path)
        
        self.classical_clf.load(path / 'classical_model.pkl')
        self.transformer_clf.load(path / 'transformer_model')
        
        meta_path = path / 'meta_classifier.pkl'
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                self.meta_classifier = pickle.load(f)
        
        with open(path / 'hybrid_config.pkl', 'rb') as f:
            data = pickle.load(f)
            self.config = data.get('config', self.config)
        
        self.is_fitted = True
        logger.info(f"Hybrid classifier loaded from {path}")


def create_classifier(config: ClassifierConfig) -> Union[ClassicalClassifier, 
                                                          TransformerClassifier, 
                                                          HybridClassifier]:
    """Factory function to create appropriate classifier based on config."""
    if config.type == "classical":
        return ClassicalClassifier(config)
    elif config.type == "transformer":
        return TransformerClassifier(config)
    elif config.type == "hybrid":
        return HybridClassifier(config)
    else:
        raise ValueError(f"Unknown classifier type: {config.type}")
