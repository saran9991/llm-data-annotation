from fastapi import FastAPI, UploadFile, File
import pandas as pd
from annotation.annotate_gpt35 import analyze_gpt35
from annotation.data_versioning import get_next_version
from pathlib import Path
from typing import List, Dict, Union, Tuple

app = FastAPI()


def annotate_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Annotate the dataframe along with confidence scores.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing text data.

    Returns:
    - Tuple[pd.DataFrame, List[str]]: A tuple containing the annotated DataFrame and a list of logs.
    """
    sentiments, confidence_scores, all_logs = [], [], []

    for _, row in df.iterrows():
        text = row['text']
        sentiment, confidence, logs = analyze_gpt35(text)
        sentiments.append(sentiment)
        confidence_scores.append(confidence)
        all_logs.extend(logs)

    df['predicted_labels'] = sentiments
    df['confidence_scores'] = confidence_scores
    return df, all_logs


def save_dataframe(path: str, prefix: str, df: pd.DataFrame) -> str:
    """
    Save dataframe to a versioned CSV file and return the file path.

    Parameters:
    - path (str): Directory path where the CSV file will be saved.
    - prefix (str): Prefix for the CSV file name.
    - df (pd.DataFrame): DataFrame to be saved.

    Returns:
    - str: File path of the saved CSV file.
    """
    version = get_next_version(path, prefix)
    file_path = Path(path) / f"{prefix}{version}.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@app.post("/annotate_dataset/")
async def annotate_dataset(file: UploadFile = File(...)) -> Dict[str, Union[str, List[str]]]:
    """
    Annotate a dataset with text data and return results.

    Parameters:
    - file (UploadFile): Uploaded CSV file containing text data.

    Returns:
    - Dict[str, Union[str, List[str]]]: A dictionary containing the status, file paths, and logs.
    """
    df = pd.read_csv(file.file)

    # Annotate the dataframe
    df, all_logs = annotate_dataframe(df)

    # Save the annotated dataframe
    annotated_path = save_dataframe("data/annotated", 'annotated_', df)

    # Filter the dataset where confidence_scores < 1 and save
    filtered_dataset = df[df['confidence_scores'] < 1]
    filtered_path = save_dataframe("data/filtered", 'filtered_', filtered_dataset)

    return {
        "status": "success",
        "path": annotated_path,
        "filtered_path": filtered_path,
        "logs": all_logs
    }
