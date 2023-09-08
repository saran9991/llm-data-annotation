import requests
from pathlib import Path
import streamlit as st
import pandas as pd
from models.train_bert import train_bert
import torch
from sklearn.model_selection import train_test_split
from annotation.data_versioning import get_next_version
from annotation.cleanlab_label_issues import find_label_issues

# Setting Streamlit page config
st.set_page_config(page_title="Data Annotation", page_icon="🚀", layout="wide")

# Inline CSS
st.markdown(
    """
    <style>
        div.row-widget.stButton > button {
            margin: auto;
            display: block;
            transition: transform .2s;
        }
        div.row-widget.stButton > button:hover {
            transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("LLM Seminar Data Annotation  ✏️")
st.write("An interactive tool to annotate your dataset, preview annotations, and save changes.")

# Session state initialization
session_keys = ["iteration", "initial_training", "top20_status", "stop_iterations", "display_top_20", "next_iteration"]
for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = 1 if key == "iteration" else False

# Helper functions
def cleanlab_style() -> str:
    """Loads cleanlab processing style from the frontend resources."""
    with open('frontend_resources/cleanlab_processing_style.html', 'r') as file:
        return file.read()

def train_model_style(epoch_value: int) -> str:
    """Returns the train model style string with formatted epoch input."""
    with open("frontend_resources/train_model_style.html", "r") as f:
        content = f.read()
    return content.format(epoch_input=epoch_value)


uploaded_file = st.file_uploader("Choose a dataset (CSV)", type="csv")

if uploaded_file:
    # This button is responsible for annotating the dataset using GPT 3.5
    if st.button("Annotate"):
        with st.spinner('Annotating rows using GPT 3.5...'):
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
                f"<div style='background-color: rgba(255,229,180,0.7); padding: 1rem; border: 1px solid rgba(0,0,0,"
                f"0.6); border-radius: 0.5rem; color: rgba(0,0,0,0.9);'><strong>{low_confidence_rows}</strong> rows "
                f"with annotation confidence less than 1, please annotate these manually</div>",
                unsafe_allow_html=True
            )

        else:
            st.error("Failed to annotate dataset.")

if "filtered_dataset" in st.session_state:
    # Shows the rows with low confidence values
    st.dataframe(st.session_state.filtered_dataset, use_container_width=True)

    row_options = list(st.session_state.filtered_dataset.index)
    row_selection = st.selectbox("Edit label for row:", options=row_options)
    label_options = ["negative", "neutral", "positive"]
    new_label = st.selectbox("Select new label:", options=label_options)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        # Human annotating the GPT 3.5 annotated dataset
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
    # Merge the human annotated low confidence rows with GPT 3.5 Annotated Dataset
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
            # get_next_version provides for rudimentary data versioning
            # The GPT 3.5 + Human annotated dataset ( merged dataset ) is saved under data/merged
            version = get_next_version("data/merged", 'merged_')
            save_path = Path("data/merged") / f"merged_{version}.csv"
            merged_dataset.reset_index(drop=True)
            merged_dataset.to_csv(save_path, index=False)
            st.session_state.save_path = str(save_path)

            st.success(f"Dataset saved successfully at {save_path}")
            st.session_state.merged_successful = True
            # To get more consistent test results across models, we use a static test set
            st.success('Allocating 20% of the rows as a hold-out test set')

            train_data, test_data = train_test_split(merged_dataset, test_size=0.2)

            # Train data is saved under data/trainsets
            train_version = get_next_version("data/trainsets", 'train_')
            train_save_path = Path("data/trainsets") / f"train_{train_version}.csv"
            train_data.reset_index(drop=True)
            train_data.to_csv(train_save_path, index=False)

            # Test data is saved under data/testsets
            test_version = get_next_version("data/testsets", 'test_')
            test_save_path = Path("data/testsets") / f"test_{test_version}.csv"
            test_data.reset_index(drop=True)
            test_data.to_csv(test_save_path, index=False)
            # Logging these to st.session_state for later use
            st.session_state.test_set_path = test_save_path
            st.session_state.train_save_path = train_save_path

        except Exception as e:
            st.error(f"An error occurred: {e}")
    # After the GPT 3.5 + Human annotated dataset has been saved, train BERT model on it
    if st.session_state.get('merged_successful'):
        st.write("----")
        st.session_state.experiment_name = st.text_input("Enter the experiment name:",
                                                         value="llm_seminar_data_annotation")

        epoch_input = int(st.text_input("Enter the number of Epochs for BERT Training:", value="1"))
        model_name_inp = st.text_input("Enter the model name:", value="bert_sentiment_gpt35")
        # Training BERT on the GPT 3.5 + Human annotated dataset for n epochs
        if st.button("Train Model"):
            training_message = train_model_style(epoch_input)
            st.markdown(training_message, unsafe_allow_html=True)
            progress_bar = st.empty()

            # This method is responsible for showing the model training progress bar
            def update_progress(current_epoch, total_epochs):
                progress = current_epoch / total_epochs
                progress_bar.progress(progress)


            if not hasattr(st.session_state, 'save_path'):
                st.warning("No dataset available for training. Please upload, annotate, and then merge first.")
            else:
                # Training BERT
                model_path, val_acc, model = train_bert(model_path=f"models/{model_name_inp}.pt",
                                                        train_data_path=st.session_state.train_save_path,
                                                        test_data_path=st.session_state.test_set_path,
                                                        experiment_name=st.session_state.experiment_name,
                                                        epoch_input=epoch_input,
                                                        model_name_inp=model_name_inp,
                                                        progress_callback=update_progress
                                                        )

                st.success(f"Model trained successfully and saved at {model_path}", icon='✅')
                st.write(f"Current Model's trained Validation Accuracy: {val_acc:.2f}")
                st.session_state.model_path = model_path
                st.session_state.initial_model = model
                st.session_state.initial_training = True

    if st.session_state.get('stop_iterations', False):
        st.stop()
    # After the initial BERT Training on the GPT 3.5 + Human annotated dataset, we use CleanLab to find label issues
    # This is done in an iterative manner to enhance the quality of the final dataset and the model test results
    if st.session_state.get('initial_training') and not getattr(st.session_state, 'stop_iterations', False):
        heading_style = cleanlab_style()
        st.markdown(heading_style, unsafe_allow_html=True)
        st.write("----")
        st.subheader(f"Iteration: {st.session_state.iteration}")
        # If it's the first iteration, we take the BERT model trained on the initial Human + GPT dataset
        if st.session_state.iteration == 1 and st.session_state.initial_training:
            st.session_state.current_model = st.session_state.initial_model
            st.session_state.current_data_path = st.session_state.train_save_path
            print('Iteration 1 data_path:', st.session_state.current_data_path)

        # Else we choose the previous iteration's dataset and model
        else:
            #
            model_path = f"models/model_cleanlab_{st.session_state.iteration - 1}.pt"
            loaded_model = torch.load(model_path)
            st.session_state.current_model = loaded_model
            st.session_state.current_data_path = f"data/cleaned/cleaned_{st.session_state.iteration - 1}.csv"
            print('Iteration:', st.session_state.iteration, ' data_path:', st.session_state.current_data_path)

        # Button to find label issues
        if st.button("Find Label Issues", key="find_issues"):
            st.session_state.top_20 = find_label_issues(st.session_state.current_model,
                                                        st.session_state.current_data_path)
            st.success('These are the top 20 labels in the dataset with lowest label quality:')
            st.session_state.display_top_20 = True
            st.session_state.top20_status = True

        if st.session_state.display_top_20:
            # If the CleanLab process was a success, we display 20 rows with the lowest label scores
            st.dataframe(st.session_state.top_20, use_container_width=True)

        if st.session_state.top20_status:
            # Providing an interface for the human to annotate these rows with label issues
            st.subheader("Label Issues for Annotation")
            row_options = list(st.session_state.top_20.index)
            row_selection = st.selectbox("Edit label for row:", options=row_options, key="row_selection")
            label_options = ["negative", "neutral", "positive"]
            new_label = st.selectbox("Select new label:", options=label_options, key="new_label_selection")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("Update Label", key='update_iterative_button'):
                    st.session_state.top_20.loc[row_selection, 'predicted_labels'] = new_label
                    st.session_state.display_top_20 = True

            # We finally merge these annotated rows with the dataset
            if st.button("Merge and Save Cleaned Data", key="merge_clean"):
                original_data = pd.read_csv(st.session_state.current_data_path)

                merged_dataset = original_data.merge(st.session_state.top_20[['text', 'predicted_labels']],
                                                     on='text', how='left')

                merged_dataset['predicted_labels'] = merged_dataset['predicted_labels_y'].combine_first(
                    merged_dataset['predicted_labels_x'])

                merged_dataset = merged_dataset.drop(columns=['predicted_labels_x', 'predicted_labels_y'])

                # The cleaned dataset is saved under data/cleaned/cleaned_i , where i denotes the iteration number
                save_cleaned_path = f"data/cleaned/cleaned_{st.session_state.iteration}.csv"
                merged_dataset.to_csv(save_cleaned_path, index=False)
                st.success(f"Cleaned data saved at: {save_cleaned_path}")
                st.session_state.save_cleaned_path = save_cleaned_path
                setattr(st.session_state, f'data_cleaning_{st.session_state.iteration}', True)

        # Once the cleaned dataset has been saved, we train BERT on it and get evaluation metrics
        if getattr(st.session_state, f'data_cleaning_{st.session_state.iteration}', False):
            st.write("----")
            epoch_input = int(
                st.text_input("Enter the number of Epochs for BERT Training:", value="1", key='ep_cl_inp'))
            model_name_inp = st.text_input("Enter the model name:", value="bert_sentiment_cleanlab", key='mname_cl_inp')
            if st.button("Train Model on Cleaned Data"):
                training_message = train_model_style(epoch_input)
                st.markdown(training_message, unsafe_allow_html=True)

                progress_bar = st.empty()

                # This method is responsible for showing the model training progress bar
                def update_progress(current_epoch, total_epochs):
                    progress = current_epoch / total_epochs
                    progress_bar.progress(progress)


                if not hasattr(st.session_state, 'save_cleaned_path'):
                    st.warning("No dataset available for training. Please upload, annotate, and then merge first.")
                else:
                    # Training BERT on cleaned dataset for iteration i
                    model_path, val_acc, model = train_bert(
                        model_path=f"models/model_cleanlab_{st.session_state.iteration}.pt",
                        train_data_path=st.session_state.save_cleaned_path,
                        test_data_path=st.session_state.test_set_path,
                        experiment_name=st.session_state.experiment_name,
                        epoch_input=epoch_input,
                        model_name_inp=model_name_inp,
                        progress_callback=update_progress)
                    st.success(f"Model trained successfully and saved at {model_path}", icon='✅')
                    st.write(f"Model's trained Validation Accuracy on Cleaned Data: {val_acc:.2f}")
                    st.session_state.model_path = model_path
                    st.session_state.current_model = model
                    st.session_state.bert_clean_training = True
                    setattr(st.session_state, f'iteration_{st.session_state.iteration}', True)

        # The iteration number is increased if data cleaning for previous iteration is completed
        # and if the entire iteration has processed
        if (getattr(st.session_state, f'iteration_{st.session_state.iteration}', False)
                and getattr(st.session_state, f'data_cleaning_{st.session_state.iteration}', False)):
            st.session_state.iteration += 1
            st.session_state.top20_status = False
            setattr(st.session_state, f'iteration_{st.session_state.iteration}', False)

            col_next, col_stop = st.columns(2)

            with col_next:
                if st.button("Next Iteration"):
                    st.write("----")
                    st.session_state.display_top_20 = False
                    st.session_state.next_iteration = True

            with col_stop:
                st.session_state.display_top_20 = False
                st.session_state.next_iteration = False
                if st.button('Stop Iterative CleanLab processing'):
                    st.write("----")
                    st.session_state.stop_iterations = True
