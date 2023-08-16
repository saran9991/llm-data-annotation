from annotation.annotate_gpt35 import analyze_gpt35
import pandas as pd
from sklearn.metrics import accuracy_score
from models.train_bert import train_bert
from annotation.data_versioning import get_next_version
import os

def annotate_dataset(filepath):
    unannotated = pd.read_csv(filepath, encoding='unicode_escape', index_col=[0])
    original_dataset = pd.read_csv('data/original/train.csv',encoding='unicode_escape')

    sentiments_and_scores = unannotated['text'].apply(analyze_gpt35)
    unannotated['predicted_labels'] = [x[0] for x in sentiments_and_scores]
    unannotated['confidence_score'] = [x[1] for x in sentiments_and_scores]

    unannotated = unannotated[unannotated['predicted_labels'].isin(['positive', 'negative', 'neutral'])]

    version = get_next_version('data/annotated', 'annotated_')
    annotated_file_path = os.path.join('data', 'annotated', f"annotated_{version}.csv")
    unannotated.to_csv(annotated_file_path)
    print('Annotated dataset head: ', unannotated.head())

    accuracy = accuracy_score(
        original_dataset.loc[list(set(unannotated.index) & set(original_dataset.index)), 'sentiment'].values,
        unannotated.loc[list(set(unannotated.index) & set(original_dataset.index)), 'predicted_labels'].values
    )
    print(f"Accuracy of GPT 3.5's annotations: {accuracy}")
    return annotated_file_path

if __name__ == "__main__":
    annotate_dataset('data/unannotated/unannotated_50.csv')
    train_bert('models/bert_sentiment_gpt35_1000_model3.pt', 'data/annotated/gpt35_conf_scores_1000_preproc.csv')
