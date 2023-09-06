import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from cleanlab.classification import CleanLearning
from sklearn.preprocessing import LabelEncoder


class BertSentimentClassifier(BaseEstimator):
    def __init__(self, model_path='bert-base-uncased', device=None, epochs = 1):
        self.model_path = model_path
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.max_len = 128
        self.epochs = epochs

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        train_data = SentimentDataset(X, y, self.tokenizer, max_len=self.max_len)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        for epoch in range(self.epochs):
            train_acc, train_loss = train_epoch(self.model, train_loader, optimizer, self.device)
            print(f'Epoch {epoch + 1}/{self.epochs} - Train loss: {train_loss}, accuracy: {train_acc}')

    def predict(self, X):
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

    def predict_proba(self, X):
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

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = (y_pred == y).mean()
        return accuracy

    def set_model_weights(self, state_dict):
        self.model.load_state_dict(state_dict)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)


class SentimentDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
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

def train_epoch(model, data_loader, optimizer, device, scheduler=None):
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

def eval_model(model, data_loader, device, sentiments):
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


def train_bert(model_path, data_path, experiment_name, epoch_input, model_name_inp,
                             progress_callback=None):
    # MLflow setup
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
        data = pd.read_csv(data_path, encoding='unicode_escape')
        data = data.dropna()
        #data['predicted_labels'] = data['predicted_labels'].map(sentiments)

        raw_texts, raw_labels = data["text"].values, data["predicted_labels"].values
        raw_train_texts, raw_test_texts, raw_train_labels, raw_test_labels = train_test_split(raw_texts, raw_labels,
                                                                                              test_size=0.2)

        # Label encoding the labels
        encoder = LabelEncoder()
        encoder.fit(raw_train_labels)
        train_labels = encoder.transform(raw_train_labels)
        test_labels = encoder.transform(raw_test_labels)

        # Define classifier and fit
        clf = BertSentimentClassifier(epochs= epoch_input)
        clf.fit(raw_train_texts, train_labels)
        initial_accuracy = clf.score(raw_test_texts, test_labels)
        print(f'Accuracy: {initial_accuracy:.4f}')

        mlflow.log_metric("accuracy", initial_accuracy)
    torch.save(clf, model_path)
    mlflow.end_run()
    return model_path, initial_accuracy, clf

