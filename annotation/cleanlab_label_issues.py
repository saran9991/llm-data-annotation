from cleanlab.classification import CleanLearning
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get_dataframe_by_index(index, raw_train_texts, raw_train_labels, encoder, label_issues):
    df = pd.DataFrame(
        {
            "text": raw_train_texts,
            "given_label": raw_train_labels,
            "predicted_label": encoder.inverse_transform(label_issues["predicted_label"]),
            "quality": label_issues["label_quality"]
        }
    )

    return df.iloc[index]


def find_label_issues(clf, data_path):
    # Load data

    data = pd.read_csv(data_path, encoding='unicode_escape')
    raw_texts, raw_labels = data["text"].values, data["predicted_labels"].values
    raw_train_texts, raw_test_texts, raw_train_labels, raw_test_labels = train_test_split(raw_texts,
                                                                                          raw_labels,
                                                                                          test_size=0.2)
    cv_n_folds = 3
    cl = CleanLearning(clf, cv_n_folds=cv_n_folds)

    encoder = LabelEncoder()
    encoder.fit(raw_train_labels)
    train_labels = encoder.transform(raw_train_labels)
    test_labels = encoder.transform(raw_test_labels)

    separator = '-' * 40
    print(separator + ' Finding Label Issues: ' + separator + '\n')
    label_issues = cl.find_label_issues(X=raw_train_texts, labels=train_labels)
    #identified_issues = label_issues[label_issues["is_label_issue"] == True]
    lowest_quality_labels = label_issues["label_quality"].argsort().to_numpy()


    top_20_error_rows = get_dataframe_by_index(lowest_quality_labels[:20], raw_train_texts, raw_train_labels, encoder,
                                               label_issues)

    return top_20_error_rows



