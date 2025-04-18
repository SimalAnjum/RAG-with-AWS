# RAG-with-AWS

## Real-Time Document Q&A with LLMs

---

### 1. Introduction

This project implements a basic **Retrieval-Augmented Generation (RAG)** system. It allows users to ask questions about the content of uploaded documents (PDF/TXT) and get contextually accurate answers derived directly from that content. It utilises **Sentence-Transformer based embeddings** (specifically the BAAI/bge-small-en-v1.5 model via LlamaIndex's HuggingFace integration) for representing text, uses **LlamaIndex** for building a searchable vector index and performing information retrieval, and leverages a **language model API** (like Together AI) to generate answers based on the retrieved context.

The system serves as a prototype for applications requiring accurate information retrieval and question answering from structured document collections, applicable to various domains including potentially **legal** and **compliance**, where efficiently extracting information from large document sets is crucial.

---

### 2. Features

-   **Document Embedding:** Uses the `BAAI/bge-small-en-v1.5` model via LlamaIndex's integration with HuggingFace embeddings (which typically uses the Sentence-Transformers library underneath for local models).
-   **Vector Indexing & Retrieval:** Indexes and retrieves document chunks using **LlamaIndex**'s `VectorStoreIndex` and retriever components for efficient similarity search based on embeddings.
-   **Flexible Document Loading:** Supports loading content from individual **PDF** (using PyMuPDFReader) and **TXT** files, as well as structured data from paired **JSON and TXT** files.
-   **RAG Pipeline:** Implements the core Retrieval-Augmented Generation pipeline by combining retrieved document contexts with user queries and sending them to a Language Model (LLM) via an API.
-   **Language Model Integration:** Connects to a powerful LLM (like `meta-llama/Llama-3.3-70B-Instruct-Turbo` as configured) via the **Together AI API** for generating answers.
-   **Web API Interface:** Provides a **FastAPI** endpoint (`/rag`) for easy programmatic interaction, allowing users to upload files and submit queries.
-   **Persistent Index Storage:** Can persist the created document index to disk (`./storage` directory by default) to avoid re-indexing every time the application runs.
-   **Contextual Answering:** Generates answers that are strictly based on the provided context from the retrieved document chunks.

---

### 3. Project Structure

```
RAG-with-AWS-main/
├── app/
├── datasets/
│   └── cuad_raw/
│       └── setup.py           # Script to download CUAD dataset
├── utils/
│   └── preprocess_cuad.py     # Preprocess CUAD dataset for retrieval
├── final.py                   # Main RAG system (embedding, retrieval, generation)
├── test.py                    # Evaluation and testing script
└── README.md                  # Project documentation
```

---

### 4. Setup Instructions

#### Prerequisites
- Python 3.11
- Install required libraries:

```bash
pip install -r requirements.txt
```

> (The project expects `sentence-transformers`, `faiss-cpu`, `datasets`, `together`, `scikit-learn`, `rouge-score`, `bert-score`, and related dependencies.)

#### Steps

1.  **Create a Virtual Environment**
    It's highly recommended to work within a virtual environment to manage project dependencies and avoid conflicts.
    ```bash
    python -m venv .venv
    ```
    Activate the virtual environment:
    * On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    * On Windows:
        ```bash
        .venv\Scripts\activate
        ```

2.  **Install Backend Dependencies**
    With your virtual environment activated, navigate to the directory containing `final.py` and your `requirements.txt` file. Install the required Python libraries.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Together AI API Key**
    Add your Together AI API key in the dummy `.env` file:
    ```dotenv
    TOGETHER_API_KEY=your_api_key_here
    ```

4.  **Dataset Availability**
    The dataset (e.g., CUAD) is expected to be already located and preprocessed at the path `RAG-with-AWS/datasets/cuad_raw`. You typically do **not** need to run the setup script for normal operation.
    * *Troubleshooting:* In case of issues with the dataset files, you can try running the original setup script:
        ```bash
        python datasets/cuad_raw/setup.py
        ```

5.  **Run the Backend API**
    Navigate to the directory containing `final.py` (if you are not already there). Start the FastAPI server using uvicorn.
    ```bash
    python final.py
    ```
    This starts the RAG backend service, accessible at `http://0.0.0.0:8000`. Keep this terminal window open and running.

6.  **Run the Frontend Development Server**
    Open a **new** terminal window and navigate to your frontend project directory (e.g., `cd app` from your project root).
    Start the frontend server:
    ```bash
    # Run this first time or if frontend dependencies change
    # npm install
    npm run dev
    ```
    The frontend application should now be running and should attempt to connect to the backend API on port 8000. Check the terminal output for the URL where the frontend is accessible (typically `http://localhost:3000` or similar).

7.  **Use the Application via Frontend**
    Access the frontend URL in your web browser. Use the provided interface to upload a file from your local machine or download a file from `corpus/cuad` and submit your query.

8.  **Evaluate the System (Optional)**
    ```bash
    python test.py
    ```
---
### 5. Usage

The application consists of two main parts: a frontend and a backend API service (`final.py`).

* **Prerequisites:**
    * Ensure you have Node.js and npm installed for the frontend.
    * Ensure you have Python installed with the necessary libraries for the backend (these are listed in the `requirements.txt` file, which is installed using `pip install -r requirements.txt`).
    * Ensure you put your Together AI API key in the `.env` file:
        ```dotenv
        TOGETHER_API_KEY=your_api_key_here
        ```

* **Running the Backend API:**
    * Navigate to the directory where `final.py` is located in your terminal.
    * Start the FastAPI server:
        ```bash
        python final.py
        ```
    * This command tells uvicorn to run the `app` instance within the `final` module, reload the server on code changes, and listen on port 8000. The API will now be running and ready to accept requests from the frontend or other clients.

* **Running the Frontend:**
    * Navigate to your frontend project directory (by using  `cd app` from your project root).
    * Install frontend dependencies (if you haven't already):
        ```bash
        npm install
        ```
    * Start the frontend development server:
        ```bash
        npm run dev
        ```
    * This will start the frontend application, which should then be able to communicate with the backend API running on port 8000. The specific address where the frontend is accessible will be shown in your terminal (e.g., `http://localhost:3000`).

---

### 6. Performance Evaluation

The system supports a comprehensive evaluation of Q&A quality using multiple metrics:

| Metric                    | Description |
|:---------------------------|:------------|
| Exact Match (EM)           | Checks if the generated answer exactly matches the ground truth answer. |
| F1 Score                   | Measures overlap between generated and true answers, combining precision and recall. |
| Mean Reciprocal Rank (MRR) | Evaluates the rank position of the correct answer in retrieved results. |
| BERT Precision, Recall, F1 | Computes fine-grained semantic matching scores using BERT embeddings. |

**Evaluation Script:**  
All evaluations are automatically performed by the `evaluate_qna_system()` function in `test.py`.

**Result:**  
The evaluation returns an aggregated report summarizing all metrics.

---

### 7. Technologies Used

| Component                 | Library / Tool                                       |
| :------------------------ | :--------------------------------------------------- |
| Web Framework & API       | FastAPI                                              |
| ASGI Server               | Uvicorn                                              |
| Document Loading (PDF)    | LlamaIndex (specifically PyMuPDFReader)              |
| Document Loading (TXT)    | Python Standard I/O, pathlib                         |
| Document Chunking         | LlamaIndex (SentenceSplitter)                        |
| Embedding Model           | HuggingFace Embedding (using BAAI/bge-small-en-v1.5) |
| Embedding Library         | Sentence-Transformers (underlying library for HF local embeddings) |
| Vector Indexing & Storage | LlamaIndex (VectorStoreIndex, StorageContext)        |
| Information Retrieval     | LlamaIndex (Retriever)                               |
| Language Model API        | Together API (using model Llama-3.3-70B-Instruct-Turbo) |
| Environment Variables     | python-dotenv                                        |
| Utility Functions         | Python Standard Library (e.g., json, shutil, logging, pathlib) |

---

### 8. Notes

- **AWS Mention:** This prototype does not yet directly use AWS services in the code.
- **Language Model:** You must have valid API access for TogetherComputer to generate answers.
- **Prototype Status:** This project is a local prototype intended for experimental testing of RAG workflows.

---

### 9. Future Improvements (Optional)

- Integrate AWS S3 for model and dataset storage.
- Deploy as a FastAPI or Flask-based web service.
- Add user authentication and query logging.
- Enhance scalability for production environments.

---

### 10. Credits

- CUAD Dataset: [Contract Understanding Atticus Dataset](https://huggingface.co/datasets/cuad)

---

### 11. License

This project is for educational and experimental purposes.
