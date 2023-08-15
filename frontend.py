import streamlit as st
import requests
import pandas as pd
from io import StringIO
from annotation.data_versioning import get_next_version
from models.train_bert import train_bert
from pathlib import Path



# Set Page title and icon
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

# App Header
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

            # Displaying the low confidence row count with more visibility
            st.markdown(
                f"<div style='background-color: rgba(255,229,180,0.7); padding: 1rem; border: 1px solid rgba(0,0,0,0.6); border-radius: 0.5rem; color: rgba(0,0,0,0.9);'><strong>{low_confidence_rows}</strong> rows with annotation confidence less than 1, please annotate these manually</div>",
                unsafe_allow_html=True
            )

        else:
            st.error("Failed to annotate dataset.")

if "filtered_dataset" in st.session_state:
    st.dataframe(st.session_state.filtered_dataset, use_container_width=True)

    # Editing the labels
    row_options = list(st.session_state.filtered_dataset.index)
    row_selection = st.selectbox("Edit label for row:", options=row_options)
    label_options = ["negative", "neutral", "positive"]
    new_label = st.selectbox("Select new label:", options=label_options)

    # Place the Update Label button on the left
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Update Label"):
            st.session_state.filtered_dataset.loc[row_selection, "predicted_labels"] = new_label

    # Customized button for Merge and Save
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

    # Use the custom button styling for Merge and Save
    if st.button("Merge and Save", key="customSaveButton"):
        try:
            if 'dataset' not in st.session_state:
                st.warning("No original dataset available for merging.")

            annotated_dataset = pd.read_csv(st.session_state.annotated_path).drop(columns='Unnamed: 0')

            # Merge the filtered dataset labels into the annotated dataset
            merged_dataset = annotated_dataset.merge(st.session_state.filtered_dataset[['text', 'predicted_labels']],
                                                     on='text', how='left')

            # Replace the labels in the annotated_dataset with the labels from the filtered_dataset for the rows that have been updated
            merged_dataset['predicted_labels'] = merged_dataset['predicted_labels_y'].combine_first(
                merged_dataset['predicted_labels_x'])

            # Drop the extra columns
            merged_dataset = merged_dataset.drop(columns=['predicted_labels_x', 'predicted_labels_y'])

            # Automatically determine the save path for the merged dataset using data versioning
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

        if st.button("Train Model"):
            if not hasattr(st.session_state, 'save_path'):
                st.warning("No dataset available for training. Please upload, annotate, and then merge first.")
            else:
                # Modify the train_bert function to accept experiment_name
                model_path, best_run_id, best_val_acc = train_bert("models/bert_sentiment_gpt35_1000_model3.pt", st.session_state.save_path, experiment_name)
                st.success(f"Model trained successfully and saved at {model_path}")
                st.write(f"Best Model Run ID: {best_run_id}")
                st.write(f"Best Validation Accuracy: {best_val_acc:.2f}")

