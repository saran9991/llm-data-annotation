import streamlit as st
import requests
import pandas as pd
from annotation.data_versioning import get_next_version
from models.train_bert import train_bert
from pathlib import Path
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from sklearn.base import BaseEstimator
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from cleanlab.classification import CleanLearning

from annotation.cleanlab_label_issues import get_dataframe_by_index
from annotation.cleanlab_label_issues import find_label_issues

if "new_save_path" not in st.session_state:
    st.session_state.new_save_path = None

if 'step' not in st.session_state:
    st.session_state.step = 1

# Check if 'n_iterations' is in session state, if not, initialize it
if 'n_iterations' not in st.session_state:
    st.session_state.n_iterations = 1

# Check if 'annotated_rows' is in session state, if not, initialize it
if 'annotated_rows' not in st.session_state:
    st.session_state.annotated_rows = {}

def merge_and_save(main_dataset_path, top_20_error_rows):
    main_dataset = pd.read_csv(main_dataset_path)

    # Merging logic (adapted from the earlier provided code)
    merged_dataset = main_dataset.merge(top_20_error_rows[['text', 'predicted_labels']], on='text', how='left')

    merged_dataset['predicted_labels'] = merged_dataset['predicted_labels_y'].combine_first(
        merged_dataset['predicted_labels_x'])
    merged_dataset = merged_dataset.drop(columns=['predicted_labels_x', 'predicted_labels_y'])
    merged_dataset = merged_dataset[
        merged_dataset['predicted_labels'].apply(lambda x: x.lower() in ['positive', 'negative', 'neutral'])]

    # Save the merged dataset
    version = get_next_version(Path("data/cleaned"), 'cleaned_')
    save_path = Path("data/cleaned") / f"cleaned_{version}.csv"
    merged_dataset.to_csv(save_path, index=False)

    return str(save_path)


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
    if st.button("Annotate", key="annotate_btn"):
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
        if st.button("Update Label", key="update_label_btn"):
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
            merged_dataset = merged_dataset[merged_dataset['predicted_labels'].apply(lambda x: x.lower() in ['positive', 'negative', 'neutral'])]

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

        if st.button("Train Model", key="train_model_btn"):
            if not hasattr(st.session_state, 'save_path'):
                st.warning("No dataset available for training. Please upload, annotate, and then merge first.")
            else:
                model_path, val_acc, clf = train_bert(f"models/{model_name_inp}", st.session_state.save_path, experiment_name, epoch_input, model_name_inp)
                st.success(f"Model trained successfully and saved at {model_path}", icon='✅')
                st.write(f"Current Model's trained Validation Accuracy: {val_acc:.2f}")
                st.session_state.initial_training_complete = True
                st.session_state.clf = clf



if st.session_state.get('initial_training_complete'):
    st.session_state.n_iterations = st.number_input('Number of iterations for CleanLab:', min_value=1)

    # Button to confirm number of iterations and proceed
    if st.button("Proceed with Iterations"):
        st.session_state.proceed = True
    else:
        st.session_state.proceed = False

    if 'current_iteration' not in st.session_state:
        st.session_state.current_iteration = 1

    if 'cleaned_dataset_path' not in st.session_state:
        st.session_state.cleaned_dataset_path = st.session_state.save_path  # the merged dataset path

    if st.session_state.get('proceed', False) and st.session_state.current_iteration <= st.session_state.n_iterations:
        st.write(f"Running iteration {st.session_state.current_iteration}...")

        # Identify label issues
        top_20_error_rows = find_label_issues(st.session_state.clf, st.session_state.cleaned_dataset_path)
        st.dataframe(top_20_error_rows, use_container_width=True)
        st.write(f"Please annotate the rows with label issues:")

        # Select row to annotate
        row_options = list(top_20_error_rows.index)
        row_selection = st.selectbox("Edit label for row:", options=row_options, key="row_selection_iterative_key")
        label_options = ["negative", "neutral", "positive"]
        new_label = st.selectbox("Select new label:", options=label_options, key="label_selection_iterative_key")

        # Update the label on button press
        with st.columns([1, 2, 1])[0]:
            if st.button("Update Label", key="update_label_iterative_btn"):
                top_20_error_rows.loc[row_selection, "predicted_labels"] = new_label
                # Display the updated dataframe again
                st.dataframe(top_20_error_rows, use_container_width=True)

        # Finish annotations and move to next step
        if st.button("Finish Annotations and Train", key="finish_annotations_btn"):
            # Update the main dataset and save
            new_dataset_path = merge_and_save(st.session_state.cleaned_dataset_path, top_20_error_rows)

            # Train a BERT model on this new dataset
            model_path, val_acc, clf = train_bert(f"models/bert_sentiment_{st.session_state.current_iteration}.pt",
                                                  new_dataset_path)
            st.write(
                f"Current Model's trained Validation Accuracy for iteration {st.session_state.current_iteration}: {val_acc:.2f}")

            # Increment the iteration and set the cleaned_dataset_path for the next iteration
            st.session_state.current_iteration += 1
            st.session_state.cleaned_dataset_path = new_dataset_path

    else:
        st.write("All iterations completed!")

