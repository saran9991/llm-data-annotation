import streamlit as st
import requests
import pandas as pd
from io import StringIO

# Set Page title and icon
st.set_page_config(page_title="Data Annotation", page_icon="✏️", layout="wide")

# App Header
st.title("LLM Seminar Data Annotation")
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
            st.session_state.filtered_dataset = pd.read_csv(filtered_dataset_url)
        else:
            st.error("Failed to annotate dataset.")

if "filtered_dataset" in st.session_state:
    st.subheader("Filtered Dataset Preview")
    st.table(st.session_state.filtered_dataset)  # Use st.table for better styling

    st.subheader("Edit Labels")
    row_index = st.sidebar.number_input("Edit label for row:", min_value=0,
                                        max_value=st.session_state.filtered_dataset.shape[0] - 1, value=0, step=1)
    options = st.sidebar.selectbox("Select new label:", options=["negative", "neutral", "positive"])
    if st.sidebar.button("Update Label"):
        st.session_state.filtered_dataset.loc[row_index, "predicted_labels"] = options
        st.info("Label updated successfully!")

    save_path = st.sidebar.text_input("Enter the path where you want to save the merged dataset:")
    if st.sidebar.button("Merge and Save"):
        try:
            if 'dataset' not in st.session_state:
                st.warning("No original dataset available for merging.")

            annotated_dataset = pd.read_csv(st.session_state.annotated_path).drop(columns='Unnamed: 0')

            # Merge the filtered dataset labels into the annotated dataset
            merged_dataset = annotated_dataset.merge(st.session_state.filtered_dataset[['text', 'predicted_labels']],
                                                     on='text', how='left')

            # Replace the labels in the annotated_dataset with the labels from the filtered_dataset
            merged_dataset['predicted_labels'] = merged_dataset['predicted_labels_y'].combine_first(
                merged_dataset['predicted_labels_x'])

            # Drop the extra columns
            merged_dataset = merged_dataset.drop(columns=['predicted_labels_x', 'predicted_labels_y'])

            merged_dataset.to_csv(save_path + '.csv', index=False)
            st.success(f"Dataset saved successfully at {save_path}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

