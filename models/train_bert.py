import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from sklearn.preprocessing import LabelEncoder



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


def train_bert(model_path, data_path, experiment_name, epoch_input, model_name_inp, progress_callback=None):

    EXPERIMENT_NAME = experiment_name
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment_id is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment_id.experiment_id

    #model_name = "_".join(model_path.split("/")[-1].split("_")[:-2])
    #model_name = 'bert_sentiment_gpt35'
    model_name  = model_name_inp

    with mlflow.start_run(experiment_id=experiment_id):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        MODEL_NAME = 'bert-base-uncased'
        BATCH_SIZE = 16
        EPOCHS = epoch_input

        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("model_name", MODEL_NAME)

        sentiments = {'positive': 0, 'neutral': 1, 'negative': 2}

        data = pd.read_csv(data_path) #LLM Annotated Dataset
        data['predicted_labels'] = data['predicted_labels'].map(sentiments)

        train_texts, val_texts, train_targets, val_targets = train_test_split(data['text'], data['predicted_labels'], test_size=0.2)

        train_texts = train_texts.reset_index(drop=True)
        val_texts = val_texts.reset_index(drop=True)
        train_targets = train_targets.reset_index(drop=True)
        val_targets = val_targets.reset_index(drop=True)

        tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

        train_data = SentimentDataset(train_texts, train_targets, tokenizer, max_len=128)
        val_data = SentimentDataset(val_texts, val_targets, tokenizer, max_len=128)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(sentiments)).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=2e-5)
        for epoch in range(EPOCHS):
            if progress_callback:
                progress_callback((epoch + 1) / EPOCHS)
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)
            train_acc, train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
            print(f'Train loss: {train_loss}, accuracy: {train_acc}')
            val_acc, val_report = eval_model(model, val_loader, DEVICE, sentiments)
            print(f'Val accuracy: {val_acc}\n')

            mlflow.log_metric("train_acc", train_acc)
            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("val_acc", val_acc)
            #print(val_report)

        torch.save(model, model_path)
        result = mlflow.pytorch.log_model(model, "model")

        mlflow.pytorch.log_model(model, "model")
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name=model_name
        )
    mlflow.end_run()
    return model_path, val_acc
