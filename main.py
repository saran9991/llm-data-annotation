from annotation.annotate_gpt35 import analyze_gpt35
import pandas as pd
from sklearn.metrics import accuracy_score
from models.train_bert import train_bert


def annotate_dataset(filepath):
    unannotated = pd.read_csv(filepath, encoding='unicode_escape', index_col=[0])
    original_dataset = pd.read_csv('data/original/train.csv',encoding='unicode_escape')

    num_rows = 50
    sentiments_and_scores = unannotated['text'].iloc[0:num_rows].apply(analyze_gpt35)
    unannotated.loc[unannotated.index[0:num_rows], 'predicted_labels'] = [x[0] for x in sentiments_and_scores]
    unannotated.loc[unannotated.index[0:num_rows], 'confidence_score'] = [x[1] for x in sentiments_and_scores]

    #unannotated['annotation_correct'] = (unannotated['predicted_labels'] == original_dataset['sentiment']).astype(str)
    #unannotated.to_csv('data/sentiment/annotated/gpt35/conf_scores_1000.csv')
    annotated_file_path = filepath.replace(".csv", "_annotated.csv")
    #unannotated.to_csv(annotated_file_path)

    print('Annotated dataset head: ', unannotated.head())
    accuracy = accuracy_score(
        original_dataset['sentiment'].iloc[0:num_rows].values,
        unannotated['predicted_labels'].iloc[0:num_rows].values
    )
    print(f"Accuracy of GPT 3.5's annotations: {accuracy}")
    return annotated_file_path

if __name__ == "__main__":
    annotate_dataset('data/unannotated/unannotated_50.csv')
    #train_bert('models/bert_sentiment_gpt35_1000_model_2.pt', 'data/annotated/conf_scores_1000_preproc.csv')
