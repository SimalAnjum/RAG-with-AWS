# RAG-with-AWS

# Real-Time Document Q&A with LLMs on AWS

## Team Members
- **Dua e Sameen** (ds07138)
- **Muhammad Tahir Ghazi** (mg07593)
- **Simal Anjum** (sa07716)

## 1. Introduction
This project presents a scalable, real-time document question-answering (Q&A) system leveraging AWS services for efficient information retrieval from large document repositories. Our solution is designed for industries requiring rapid access to legal, compliance, and customer support documents.

## 2. Dataset & Use Cases
We will primarily utilize **CUAD (Contract Understanding Atticus Dataset)**, which is specifically designed for contract understanding, making it relevant for legal and compliance use cases.

### **Performance Metrics**
To demonstrate real-world applications, we will execute actual queries on these datasets and measure retrieval performance using:
- **Exact Match (EM):** Measures whether the retrieved answer is identical to the ground truth.
- **F1 Score:** Evaluates overlap between predicted and actual answers.
- **Mean Reciprocal Rank (MRR):** Measures ranking quality of retrieved documents.

## 3. System Architecture & Deployment Strategy
Our architecture integrates multiple AWS services:

### **Data Storage and Text Extraction**
- **Amazon S3:** Primary document storage.
- **Amazon Textract:** Extracts text from PDFs and images.

### **Embeddings and Vector Database**
- **Amazon SageMaker:** Converts extracted text into embeddings using pre-trained models.
- **FAISS (Facebook AI Similarity Search):** A vector database enabling efficient similarity searches.

### **LLM Integration and Query Processing**
- **Large Language Model (LLM):** Processes user queries and generates responses.
- **AWS Lambda & API Gateway:** Handles incoming queries and returns responses efficiently.

### **Deployment Considerations**
#### **Primary Plan:** Live deployment using AWS services for real-time scalability.
#### **Fallback Plan (Local Deployment):** If AWS costs are prohibitive, we will use:
- **Local OCR (Tesseract OCR)** instead of Amazon Textract.
- **Self-hosted embeddings** (e.g., SentenceTransformers) instead of SageMaker.
- **Local FAISS or ChromaDB** instead of Pinecone.
- **A Postman API collection or lightweight UI** for testing and demonstrations.

## 4. Setup Instructions
### **Week 1: Requirement Gathering & AWS Setup**
1. **Create an AWS account** (if not already done) and configure billing alerts.
2. **Set up IAM Roles & Permissions**:
   - Attach `AmazonS3FullAccess`, `AmazonTextractFullAccess`, and `AmazonSageMakerFullAccess` to an IAM role.
3. **Download CUAD Dataset** from [Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad-qa).
4. **Research Existing Solutions** such as Amazon Bedrock, OpenAI’s GPT, and Haystack.

### **Week 2: S3 Storage & Textract Integration**
1. **Create an S3 bucket** (`legal-document-storage`).
2. **Upload sample legal documents (PDFs) to S3.**
3. **Integrate Amazon Textract** to extract text from PDFs using Boto3.
4. **Automate Textract with AWS Lambda** so that text extraction runs when new documents are uploaded.

## 5. How to Run the Project
1. **Clone the repository** and install dependencies:
   ```sh
   git clone https://github.com/SimalAnjum/RAG-with-AWS.git
   cd RAG-WITH-AWS
   pip install boto3
   ```
2. **Set up AWS credentials**:
   ```sh
   aws configure
   ```
3. **Run the text extraction script**:
   ```sh
   python extract_text.py
   ```
4. **Deploy AWS Lambda function** using AWS Console or Terraform.

## 6. Expected Output
- Extracted text from documents stored in JSON format.
- Automatic processing of new uploads in S3 via AWS Lambda.
- Query processing with LLMs to retrieve relevant contract clauses.

## 7. Future Enhancements
- Implement **FAISS vector search** for improved retrieval efficiency.
- Fine-tune LLMs specifically for legal queries.
- Develop a **frontend UI** for better user experience.

---
This project aims to enhance document retrieval efficiency using LLMs and AWS services, focusing on legal and compliance queries with measurable improvements in retrieval accuracy and latency. A Postman API collection or lightweight UI will be provided for testing and demonstration, ensuring accessibility and usability.

