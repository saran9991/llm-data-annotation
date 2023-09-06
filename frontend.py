import streamlit as st
import requests
import pandas as pd
from annotation.data_versioning import get_next_version
from models.train_bert import train_bert
from pathlib import Path
import streamlit as st
import pandas as pd
from cleanlab.classification import CleanLearning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models.train_bert import train_bert
import torch
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
import copy
from models.train_bert import BertSentimentClassifier

if "iteration" not in st.session_state:
    st.session_state.iteration = 2

if "initial_training" not in st.session_state:
    st.session_state.initial_training = False

if "top20_status" not in st.session_state:
    st.session_state.top20_status = False

def find_label_issues(clf, data_path):
    data = pd.read_csv(data_path, encoding='unicode_escape')
    raw_texts, raw_labels = data["text"].values, data["predicted_labels"].values
    raw_train_texts, raw_test_texts, raw_train_labels, raw_test_labels = train_test_split(raw_texts, raw_labels, test_size=0.2)
    cv_n_folds = 3
    model_copy = copy.deepcopy(clf)
    cl = CleanLearning(model_copy, cv_n_folds=cv_n_folds)

    encoder = LabelEncoder()
    encoder.fit(raw_train_labels)
    train_labels = encoder.transform(raw_train_labels)
    test_labels = encoder.transform(raw_test_labels)

    label_issues = cl.find_label_issues(X=raw_train_texts, labels=train_labels)
    lowest_quality_labels = label_issues["label_quality"].argsort().to_numpy()

    top_20_error_rows = get_dataframe_by_index(lowest_quality_labels[:20], raw_train_texts, raw_train_labels, encoder, label_issues)
    print('TOP 20 ERROR ROWS ')
    return top_20_error_rows

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

def load_your_model_method(path):
    model = torch.load(path, map_location=torch.device('cuda'))
    return model

st.set_page_config(page_title="Data Annotation", page_icon="🚀", layout="wide")

st.markdown(
    """
    <style>
        /* Centering the button */
        div.row-widget.stButton > button {
            margin: auto;
            display: block;
            transition: transform .2s; /* animation effect */
        }

        /* Enlarge the button w hen hovering */
        div.row-widget.stButton > button:hover {
            transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("LLM Seminar Data Annotation  ✏️")
st.write(
    """
    An interactive tool to annotate your dataset, preview annotations, and save changes.
    """
)
uploaded_file = st.file_uploader("Choose a dataset (CSV)", type="csv")

if uploaded_file:
    if st.button("Annotate"):
        files = {'file': uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/annotate_dataset/", files=files)

        if response.json()["status"] == "success":
            st.success(f"Dataset annotated successfully! Saved to {response.json()['path']}")
            st.session_state.annotated_path = response.json()["path"]
            st.session_state.dataset = pd.read_csv(uploaded_file).drop(columns='Unnamed: 0')

            filtered_dataset_url = response.json()['path'].replace('annotated', 'filtered')
            st.session_state.filtered_dataset = pd.read_csv(filtered_dataset_url).drop(columns='Unnamed: 0')
            low_confidence_rows = len(
                st.session_state.filtered_dataset[st.session_state.filtered_dataset['confidence_scores'] < 1])

            st.markdown(
                f"<div style='background-color: rgba(255,229,180,0.7); padding: 1rem; border: 1px solid rgba(0,0,0,0.6); border-radius: 0.5rem; color: rgba(0,0,0,0.9);'><strong>{low_confidence_rows}</strong> rows with annotation confidence less than 1, please annotate these manually</div>",
                unsafe_allow_html=True
            )

        else:
            st.error("Failed to annotate dataset.")

if "filtered_dataset" in st.session_state:
    st.dataframe(st.session_state.filtered_dataset, use_container_width=True)

    row_options = list(st.session_state.filtered_dataset.index)
    row_selection = st.selectbox("Edit label for row:", options=row_options)
    label_options = ["negative", "neutral", "positive"]
    new_label = st.selectbox("Select new label:", options=label_options)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Update Label"):
            st.session_state.filtered_dataset.loc[row_selection, "predicted_labels"] = new_label

    st.markdown(
        """
        <style>
            .customSaveButton > button {
                background-color: #FFDAB9;
                color: black;
                border: 2px solid black;
                border-radius: 8px;
                padding: 10px 30px;
                font-size: 20px;
                transition: transform .2s;
            }
            .customSaveButton > button:hover {
                transform: scale(1.05);
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button("Merge and Save", key="customSaveButton"):
        try:
            if 'dataset' not in st.session_state:
                st.warning("No original dataset available for merging.")

            annotated_dataset = pd.read_csv(st.session_state.annotated_path).drop(columns='Unnamed: 0')
            merged_dataset = annotated_dataset.merge(st.session_state.filtered_dataset[['text', 'predicted_labels']],
                                                     on='text', how='left')
            merged_dataset['predicted_labels'] = merged_dataset['predicted_labels_y'].combine_first(
                merged_dataset['predicted_labels_x'])

            merged_dataset = merged_dataset.drop(columns=['predicted_labels_x', 'predicted_labels_y'])

            version = get_next_version("data/merged", 'merged_')
            save_path = Path("data/merged") / f"merged_{version}.csv"
            merged_dataset.reset_index(drop=True)
            merged_dataset.to_csv(save_path, index=False)
            st.session_state.save_path = str(save_path)

            st.success(f"Dataset saved successfully at {save_path}")
            st.session_state.merged_successful = True

        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.session_state.get('merged_successful'):
        experiment_name = st.text_input("Enter the experiment name:", value="llm_seminar_data_annotation")
        epoch_input = int(st.text_input("Enter the number of Epochs for BERT Training:", value="1"))
        model_name_inp = st.text_input("Enter the model name:", value="bert_sentiment_gpt35_200.pt")

        if st.button("Train Model"):
            if not hasattr(st.session_state, 'save_path'):
                st.warning("No dataset available for training. Please upload, annotate, and then merge first.")
            else:
                model_path, val_acc, model = train_bert(f"models/{model_name_inp}", st.session_state.save_path, experiment_name, epoch_input, model_name_inp)
                st.success(f"Model trained successfully and saved at {model_path}", icon='✅')
                st.write(f"Current Model's trained Validation Accuracy: {val_acc:.2f}")
                st.session_state.model_path = model_path
                st.session_state.initial_model = model
                st.session_state.initial_training = True

    if st.session_state.get('initial_training'):
        st.subheader(f"Iteration: {st.session_state.iteration}")
        if st.session_state.iteration == 1 and st.session_state.initial_training == True:
            st.session_state.current_model = st.session_state.initial_model
            st.session_state.current_data_path = st.session_state.save_path
            print('data_path !!!!!!!!!!!!!:', st.session_state.current_data_path)

        else:
            model_path = f"models/model_cleanlab_{st.session_state.iteration - 1}.pt"
            #st.session_state.current_model = load_your_model_method(model_path)

            loaded_model = torch.load(model_path, map_location=torch.device('cuda'))
            loaded_model_weights = loaded_model.state_dict()
            st.session_state.current_model = BertSentimentClassifier()
            st.session_state.current_model.set_model_weights(loaded_model_weights)
            st.session_state.current_data_path = f"data/cleaned/cleaned_{st.session_state.iteration - 1}.csv"
            print('Model Path:', st.session_state.current_model)
            print('data_path:', st.session_state.current_data_path)

        # Button to find label issues
        if st.button("Find Label Issues", key="find_issues"):
            st.session_state.top_20 = find_label_issues(st.session_state.current_model,
                                                        st.session_state.current_data_path)
            st.dataframe(st.session_state.top_20,
                         use_container_width=True)  # Displaying the 20 label issues to the user
            st.session_state.top20_status = True

        if st.session_state.top20_status == True:
            st.subheader("Label Issues for Annotation")
            row_options = list(st.session_state.top_20.index)
            row_selection = st.selectbox("Edit label for row:", options=row_options, key="row_selection")
            label_options = ["negative", "neutral", "positive"]
            new_label = st.selectbox("Select new label:", options=label_options, key="new_label_selection")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("Update Label", key = 'update_iterative_button'):
                    st.session_state.top_20.loc[row_selection, 'predicted_labels'] = new_label

            st.write("----")
            # Button to merge and save cleaned data
            if st.button("Merge and Save Cleaned Data", key="merge_clean"):
                original_data = pd.read_csv(st.session_state.current_data_path)

                merged_dataset = original_data.merge(st.session_state.top_20[['text', 'predicted_labels']],
                                                     on='text', how='left')

                merged_dataset['predicted_labels'] = merged_dataset['predicted_labels_y'].combine_first(
                    merged_dataset['predicted_labels_x'])

                merged_dataset = merged_dataset.drop(columns=['predicted_labels_x', 'predicted_labels_y'])

                # Save the merged dataset
                save_cleaned_path = f"data/cleaned/cleaned_{st.session_state.iteration}.csv"
                merged_dataset.to_csv(save_cleaned_path, index=False)
                st.success(f"Cleaned data saved at: {save_cleaned_path}")
                st.session_state.save_cleaned_path = save_cleaned_path



