# Large Language Models for Efficient Data Annotation and Model Fine-Tuning

This project is about leveraging large language models (LLMs) like GPT-3.5 provided by OpenAI for annotation of task-specific training data sets, which can then be used to fine-tune smaller LLMs towards specific tasks.

## Features

1. **Dataset Uploading and Annotation**
    - Upload CSV datasets.
    - Automatically annotate datasets using the provided API.
    - Preview the annotations and highlight rows with low confidence scores.
    
2. **Manual Annotation Corrections**
    - Display the annotated dataset and enable row-wise corrections.
    - Update labels for specific rows.
    
3. **Data Versioning and Saving**
    - Merge user corrections into the annotated dataset.
    - Use data versioning to save datasets with unique versions.
    
4. **Model Training**
    - Train a BERT model on the annotated dataset.
    - Model versioning is managed using [MLflow](https://mlflow.org/), ensuring tracking and reproducibility of different model iterations.

## Setup

### Running the Tool

1. Start the FastAPI backend:
    ```bash
    uvicorn app:app --reload
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run frontend.py
    ```

3. Access the provided link in your web browser.

4. Follow the on-screen prompts to upload and annotate your dataset, make corrections, and train the model.


---