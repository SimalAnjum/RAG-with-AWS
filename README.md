# RAG-with-AWS

## Real-Time Document Q&A with LLMs and FAISS Retrieval

---

### 1. Introduction

This project implements a basic **Retrieval-Augmented Generation (RAG)** system that allows users to ask questions about a legal contracts dataset (CUAD) and get contextually accurate answers. It uses **sentence-transformer embeddings**, **FAISS** for retrieval, and a **language model API** to generate answers based on retrieved context.

The system is designed as a prototype for industries like **legal** and **compliance**, where rapid and accurate information retrieval from large document repositories is critical.

---

### 2. Features

- **Document Embedding:** Uses `all-MiniLM-L6-v2` model from Sentence-Transformers.
- **Vector Retrieval:** Indexes documents using **FAISS** for fast similarity search.
- **RAG Pipeline:** Combines retrieved documents with user queries and feeds into an LLM.
- **Dataset Support:** Built specifically around the **CUAD (Contract Understanding Atticus Dataset)**.
- **Simple Testing:** Provides basic scripts to test retrieval and answering.
- **Comprehensive Evaluation:** Implements multiple evaluation metrics to assess system performance.

---

### 3. Project Structure

```
RAG-with-AWS-main/
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
- Python 3.8+
- Install required libraries:

```bash
pip install -r requirements.txt
```

> (The project expects `sentence-transformers`, `faiss-cpu`, `datasets`, `together`, `scikit-learn`, `rouge-score`, `bert-score`, and related dependencies.)

#### Steps
1. **Download the Dataset**

```bash
python datasets/cuad_raw/setup.py
```

2. **Preprocess the Dataset**

```bash
python utils/preprocess_cuad.py
```

3. **Run the RAG System**

```bash
python final.py
```

4. **Evaluate the System**

```bash
python test.py
```

---

### 5. Usage

- When running `final.py`, it loads the preprocessed CUAD dataset.
- Embeds and indexes documents into FAISS.
- Accepts user questions.
- Retrieves top relevant contexts.
- Sends context + query to a language model API.
- Displays the generated answer.

- When running `test.py`, the system builds a fresh index, retrieves answers for questions, and automatically computes evaluation metrics.

---

### 6. Performance Evaluation

The system supports comprehensive evaluation of Q&A quality using multiple metrics:

| Metric                    | Description |
|:---------------------------|:------------|
| Exact Match (EM)           | Checks if the generated answer exactly matches the ground truth answer. |
| F1 Score                   | Measures overlap between generated and true answers, combining precision and recall. |
| Mean Reciprocal Rank (MRR) | Evaluates the rank position of the correct answer in retrieved results. |
| ROUGE-L                    | Evaluates the longest common subsequence overlap between prediction and ground truth. |
| Semantic Similarity (Cosine) | Computes embedding-based similarity between answers using Sentence-Transformers. |
| Gold Match Rate            | Checks whether the ground truth is fully contained inside the generated answer. |
| BERT Precision, Recall, F1 | Computes fine-grained semantic matching scores using BERT embeddings. |

**Evaluation Script:**  
All evaluations are automatically performed by the `evaluate_qna_system()` function in `test.py`.

**Result:**  
The evaluation returns an aggregated report summarizing all metrics.

---

### 7. Technologies Used

| Component            | Library / Tool                      |
|:---------------------|:-------------------------------------|
| Embedding             | Sentence-Transformers (MiniLM, BGE-small) |
| Vector Retrieval      | FAISS (CPU version)                  |
| Language Model API    | Together API                         |
| Dataset Management    | Huggingface Datasets (CUAD)           |
| Evaluation Metrics    | Scikit-learn, ROUGE, BERTScore, Sentence-Transformers |
| Preprocessing         | JSON, Python                         |

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
- Sentence-Transformers: [MiniLM model](https://www.sbert.net/docs/pretrained_models.html)

---

### 11. License

This project is for educational and experimental purposes.
