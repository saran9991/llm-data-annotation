from fastapi import FastAPI, UploadFile, File
import pandas as pd
from annotation.annotate_gpt35 import analyze_gpt35
from annotation.data_versioning import get_next_version
from pathlib import Path

app = FastAPI()


@app.post("/annotate_dataset/")
async def annotate_dataset(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    sentiments, confidence_scores, all_logs = [], [], []

    for _, row in df.iterrows():
        text = row['text']
        sentiment, confidence, logs  = analyze_gpt35(text)
        sentiments.append(sentiment)
        confidence_scores.append(confidence)
        all_logs.extend(logs)

    df['predicted_labels'] = sentiments
    df['confidence_scores'] = confidence_scores
    version = get_next_version("data/annotated", 'annotated_')
    annotated_path = Path("data/annotated") / f"annotated_{version}.csv"
    df.to_csv(annotated_path, index=False)

    return {"status": "success", "path": str(annotated_path), "logs": all_logs}

