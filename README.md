# Large Language Models for Efficient Data Annotation and Model Fine-Tuning with Iterative Active Learning


This framework combines human expertise with the efficiency of Large Language Models (LLMs) like OpenAI's GPT-3.5 to simplify dataset annotation and model improvement. The iterative approach ensures the continuous improvement of data quality, and consequently, the performance of models fine-tuned using this data.  This not only saves time but also enables the creation of customized LLMs that leverage both human annotators and LLM-based precision.
<h2 align="center">Architecture</h2>
<p align="center">
  <img src="./architecture.png" alt="Architecture">
</p>

## Features

1. **Dataset Uploading and Annotation**
    - Upload CSV datasets.
    - Leverage GPT-3.5 to automatically annotate datasets.
    - Preview the annotations, highlighting low-confidence score rows.

2. **Manual Annotation Corrections**
    - Display the annotated dataset for user-based corrections.
    - User can update labels for specific rows.

3. **CleanLab: Confident Learning Approach**
    - Utilizes confident learning to identify and rectify label issues.
    - Automatically displays rows with potential label errors for user-based corrections.

4. **Data Versioning and Saving**
    - Merge user corrections with the annotated dataset.
    - Advanced data versioning ensures unique dataset versions are saved for every update.

5. **Model Training**
    - Train a BERT model on the cleaned dataset.
    - Track and reproduce model versions seamlessly using [MLflow](https://mlflow.org/).

## Setup

### Prerequisites

1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Tool

1. **Start the FastAPI backend**:
    ```bash
    uvicorn app:app --reload
    ```

2. **Run the Streamlit app**:
    ```bash
    streamlit run frontend.py
    ```

3. **Launch MLflow UI**:
   To view models, metrics, and registered models, you can access the MLflow UI with the following command:
    ```bash
    mlflow ui
    ```

4. **Access the provided links in your web browser**:
    - For the main application, access the Streamlit link.
    - For MLflow's tracking interface, by default, you can navigate to `http://127.0.0.1:5000`.

5. **Follow the on-screen prompts** to upload, annotate, correct, and train on your dataset.

## Why Confident Learning?

[Confident learning](https://arxiv.org/abs/1911.00068) has emerged as a groundbreaking technique in supervised learning and weak-supervision. It aims at characterizing label noise, finding label errors, and learning efficiently with noisy labels. By pruning noisy data and ranking examples to train with confidence, this method ensures a clean and reliable dataset, enhancing the overall model performance.

## License

This project is open-sourced under the [MIT License](LICENSE).

---

