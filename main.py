from annotation.annotate_davinci import analyze_davinci
import pandas as pd
from sklearn.metrics import accuracy_score
from models.train_bert import train_bert


def annotate_dataset():
    unannotated = pd.read_csv('data/sentiment/unannotated/unannotated_sentiment_dataset.csv',
                              encoding= 'unicode_escape', index_col=[0])

    original_dataset = pd.read_csv('data/sentiment/original/train.csv',
                                   encoding= 'unicode_escape') # Has original annotations

    num_rows = 1
    unannotated['predicted_labels'] = unannotated['text'].iloc[0:num_rows].apply(analyze_davinci)
    unannotated['annotation_correct'] = (unannotated['predicted_labels'] == original_dataset['sentiment']).astype(int)
    print('Annotated dataset head: ',unannotated.head())
    #unannotated.iloc[0:num_rows].to_csv('data/sentiment/davinci003_annotated_300.csv')

    accuracy = accuracy_score(
        original_dataset['sentiment'].iloc[0:num_rows].astype('str').values,
        unannotated['annotation_correct'].iloc[0:num_rows].astype('str').values
    )
    print(f"Accuracy of GPT 3.5's annotations: {accuracy}")

if __name__ == "__main__":
    annotate_dataset()
    train_bert('models/gpt35/bert_sentiment_gpt35_400_2_model.pt', 'data/sentiment/annotated/gpt35/gpt35_annotated_400.csv')

