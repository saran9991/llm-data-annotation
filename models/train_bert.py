import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from torch.optim.lr_scheduler import StepLR

#Custom class for BERT to align it with scikit-learn
class BertSentimentClassifier(BaseEstimator):
    def __init__(self, model_path: str = 'bert-base-uncased', device: Optional[torch.device] = None, epochs: int = 1):
        """
        Initialize the BertSentimentClassifier with the given parameters.

        Parameters:
        - model_path (str): The path to the BERT model or the model name (default is 'bert-base-uncased').
        - device (Optional[torch.device]): The device to use for model training (default is 'cuda' if available, else 'cpu').
        - epochs (int): The number of training epochs (default is 1).
        """
        self.model_path = model_path
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.max_len = 128
        self.epochs = epochs

    def fit(self, X, y, progress_callback=None) -> None:
        """
        Train the model using the given data.

        Parameters:
        - X: Input data.
        - y: Target labels.
        - progress_callback: A callback function to track training progress on UI (default is None).
        """
        self.classes_ = np.unique(y)

        train_data = CustomDataset(X, y, self.tokenizer, max_len=self.max_len)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        for epoch in range(self.epochs):
            train_acc, train_loss = train_epoch(self.model, train_loader, optimizer, self.device)
            print(f'Epoch {epoch + 1}/{self.epochs} - Train loss: {train_loss}, accuracy: {train_acc}')

            if progress_callback:
                progress_callback(epoch + 1, self.epochs)


    def predict(self, X) -> np.array:
        """
        Predict the class labels for the given data.

        Parameters:
        - X: Input data.

        Returns:
        - np.array: Predicted class labels.
        """
        X_list = X.tolist()
        encoding = self.tokenizer.batch_encode_plus(
            X_list,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

        return self.classes_[preds.cpu().numpy()]

    def predict_proba(self, X) -> np.array:
        """
        Predict the class probabilities for the given data.

        Parameters:
        - X: Input data.

        Returns:
        - np.array: Predicted class probabilities.
        """
        X_list = X.tolist()  # Convert to list
        encoding = self.tokenizer.batch_encode_plus(
            X_list,  # Updated this line
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        return probs.cpu().numpy()

    def score(self, X, y) -> float:
        """
        Compute the accuracy of the model on the given test data and labels.

        Parameters:
        - X: Input data.
        - y: Target labels.

        Returns:
        - float: Accuracy score.
        """
        y_pred = self.predict(X)
        accuracy = (y_pred == y).mean()
        return accuracy

    def set_model_weights(self, state_dict: dict) -> None:
        """
        Set the model weights from the given state dictionary.

        Parameters:
        - state_dict (dict): The state dictionary containing model weights.
        """
        self.model.load_state_dict(state_dict)

    def state_dict(self) -> dict:
        """
        Get the model's state dictionary.

        Returns:
        - dict: Model's state dictionary.
        """
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the model weights from the given state dictionary.

        Parameters:
        - state_dict (dict): The state dictionary containing model weights.
        """
        return self.model.load_state_dict(state_dict)


class CustomDataset(Dataset):
    def __init__(self, texts: List[str], targets: List[int], tokenizer, max_len: int):
        """
        Initialize the CustomDataset with texts, targets, tokenizer, and a maximum sequence length.

        Parameters:
        - texts (List[str]): A list of input texts.
        - targets (List[int]): A list of target labels or scores associated with the texts.
        - tokenizer: An NLP tokenizer.
        - max_len (int): The maximum sequence length to which the input texts should be truncated or padded.
        """
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
        - int: The number of items in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve and tokenize the text at the given index and return it with the associated target.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        - dict: A dictionary containing the following elements:
            - 'text' (str): The original text.
            - 'input_ids' (torch.Tensor): The tokenized and encoded input text as a flattened tensor.
            - 'attention_mask' (torch.Tensor): The attention mask indicating which tokens are part of the input text (flattened tensor).
            - 'targets' (torch.Tensor): The target label or score as a PyTorch tensor with dtype=torch.long, indicating it's for classification tasks.
        """
        text = str(self.texts[idx])
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device, scheduler: Optional[StepLR] = None) -> Tuple[float, float]:
    """
    Trains the model for one epoch.

    Parameters:
    - model (nn.Module): The PyTorch model to be trained.
    - data_loader (DataLoader): DataLoader providing the training data.
    - optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
    - device (torch.device): The device (e.g., 'cuda' or 'cpu') where the model and data should be loaded.
    - scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): An optional learning rate scheduler.

    Returns:
    - Tuple containing:
      1. Training accuracy.
      2. Average training loss.
    """
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == targets)

        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model: nn.Module,
               data_loader: DataLoader,
               device: torch.device,
               sentiments: Dict[str, int]) -> Tuple[float, str]:
    """
    Evaluate the model on a dataset.

    Parameters:
    - model: The model to be evaluated.
    - data_loader: DataLoader providing the evaluation data.
    - device: Device (e.g., 'cuda' or 'cpu') where the model and data should be loaded.
    - sentiments: Dictionary of sentiment classes.

    Returns:
    - Tuple containing:
      1. Evaluation accuracy.
      2. Classification report string.
    """
    model = model.eval()

    correct_predictions = 0
    predictions = []
    real_values = []

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds)
            real_values.extend(targets)
            correct_predictions += torch.sum(preds == targets)

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return correct_predictions.double() / len(data_loader.dataset), classification_report(real_values, predictions, target_names=sentiments.keys())


def train_bert(model_path: str, train_data_path: str, test_data_path: str, experiment_name: str,
               epoch_input: int, model_name_inp: str, progress_callback=None):
    """
    Train a BERT-based sentiment classifier and log metrics using MLflow.

    Parameters:
    - model_path (str): The path where the trained model will be saved.
    - train_data_path (str): Path to the training data CSV file.
    - test_data_path (str): Path to the testing data CSV file.
    - experiment_name (str): Name of the MLflow experiment.
    - epoch_input (int): Number of training epochs.
    - model_name_inp (str): Name for the registered MLflow model.
    - progress_callback: A callback function to track training progress (default is None).

    Returns:
    - Tuple containing:
      1. Path to the saved model.
      2. Accuracy on the test data.
      3. Trained BERTSentimentClassifier instance.
    """
    EXPERIMENT_NAME = experiment_name
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment_id is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment_id.experiment_id
    model_name = model_name_inp

    with mlflow.start_run(experiment_id=experiment_id):

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mlflow.log_param("epochs", epoch_input)
        mlflow.log_param("model_name", model_name)

        #sentiments = {'positive': 0, 'neutral': 1, 'negative': 2}
        train_data = pd.read_csv(train_data_path, encoding='unicode_escape')
        train_data = train_data.dropna()
        test_data = pd.read_csv(test_data_path, encoding='unicode_escape')
        test_data = test_data.dropna()

        raw_train_texts, raw_train_labels = train_data["text"].values, train_data["predicted_labels"].values
        raw_test_texts, raw_test_labels = test_data["text"].values, test_data["predicted_labels"].values

        # Label encoding the labels
        encoder = LabelEncoder()
        encoder.fit(raw_train_labels)
        train_labels = encoder.transform(raw_train_labels)
        test_labels = encoder.transform(raw_test_labels)

        # Define classifier and fit
        clf = BertSentimentClassifier(epochs= epoch_input)
        clf.fit(raw_train_texts, train_labels, progress_callback)
        initial_accuracy = clf.score(raw_test_texts, test_labels)
        print(f'Accuracy: {initial_accuracy:.4f}')

        mlflow.log_metric("accuracy", initial_accuracy)
    mlflow.pytorch.log_model(clf.model, "model")

    mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
        name=model_name
    )
    torch.save(clf, model_path)
    mlflow.end_run()
    return model_path, initial_accuracy, clf

