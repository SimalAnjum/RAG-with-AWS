import os
import minio
from minio import Minio
from minio.error import S3Error
import faiss
import json
from datasets import load_dataset
import pytesseract
from pdfminer.high_level import extract_text
from PIL import Image, ImageEnhance, ImageFilter
from sentence_transformers import SentenceTransformer
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize MinIO client
MINIO_ENDPOINT = "http://127.0.0.1:9000"  # Change if hosted elsewhere
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
BUCKET_NAME = "documents"

client = Minio(
    MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False  # Set to True if using HTTPS
)

# Load SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

document_texts = []  # Store document texts to retrieve later
cuad_questions = []  # Store CUAD questions
ground_truth_answers = []  # Store CUAD ground-truth answers

# Initialize FAISS index with IndexIDMap
D = 384  # Embedding dimension (depends on model)
index = faiss.IndexIDMap(faiss.IndexFlatIP(D))


def create_bucket():
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' created.")
    else:
        print(f"Bucket '{BUCKET_NAME}' already exists.")


def load_cuad_data():
    dataset = load_dataset("theatticusproject/cuad-qa", num_proc=1)
    
    for idx, item in enumerate(dataset["train"]):
        file_name = f"cuad_doc_{idx}.txt"
        local_file_path = f"/tmp/{file_name}"
        
        with open(local_file_path, "w", encoding="utf-8") as f:
            f.write(item["context"])
        
        upload_file(local_file_path)
        
        document_texts.append(item["context"])
        cuad_questions.append(item["question"])
        
        if item["answers"]["text"]:
            ground_truth_answers.append(item["answers"]["text"][0])
        else:
            ground_truth_answers.append("")
        
        store_embeddings(item["context"], idx)
    
    print("CUAD dataset uploaded to MinIO and indexed in FAISS.")


def upload_file(file_path):
    file_name = os.path.basename(file_path)
    client.fput_object(BUCKET_NAME, file_name, file_path)
    print(f"Uploaded {file_name} to MinIO bucket '{BUCKET_NAME}'.")

def download_file(file_name, download_path):
    client.fget_object(BUCKET_NAME, file_name, download_path)
    print(f"Downloaded {file_name} to {download_path}.")

def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = image.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    text = pytesseract.image_to_string(image)
    return text

def store_embeddings(text, idx):
    embedding = generate_embeddings(text)
    embedding = np.expand_dims(embedding, axis=0).astype(np.float32)
    
    if embedding.shape[1] != D:
        raise ValueError(f"Embedding dimension mismatch: Expected {D}, got {embedding.shape[1]}")
    
    index.add_with_ids(embedding, np.array([idx]))
    print(f"Stored embedding in FAISS with ID {idx}.")


def generate_embeddings(text):
    return embedding_model.encode(text, convert_to_numpy=True)


if __name__ == "__main__":
    create_bucket()
    load_cuad_data()
    print("FAISS index size:", index.ntotal)