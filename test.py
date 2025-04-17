import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.node_parser import SentenceSplitter
# from llama_hub.file.pymu_pdf import PyMuPDFReader
from llama_index.readers.file import PyMuPDFReader


from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import os
import together
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self, 
                 persist_dir: Optional[str] = "./storage",
                 embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the document retriever with local embeddings.
        
        Args:
            persist_dir: Directory to persist the index (optional)
            embedding_model_name: Name of the HuggingFace embedding model to use
        """
                
        self.persist_dir = Path(persist_dir)
        self.index = None

        # Initialize the local embedding model
        logger.info(f"Initializing local embedding model: {embedding_model_name}")
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

    def load_documents_from_json_and_text(self, json_folder: str, text_folder: str) -> List[Document]:
        json_folder = Path(json_folder)
        text_folder = Path(text_folder)
        documents = []

        for json_file in json_folder.glob("*.json"):
            txt_file = text_folder / f"{json_file.stem}.txt"
            if not txt_file.exists():
                logger.warning(f"Missing .txt for {json_file.name}")
                continue

            with open(json_file, 'r', encoding='utf-8') as jf:
                data = json.load(jf)

            with open(txt_file, 'r', encoding='utf-8') as tf:
                text = tf.read()

            qa_pairs = data.get("qa_pairs", [])
            for qa_pair in qa_pairs:
                question = qa_pair.get("question")
                answer = qa_pair.get("answer")

                # Minimal metadata to avoid chunking issues
                minimal_metadata = {
                    "file_name": json_file.name,
                    "document_id": data.get("document_id", json_file.stem),
                    "question": question,
                    "ground_truth": answer 
                }

                documents.append(Document(
                    text=text,
                    metadata=minimal_metadata
                ))

        logger.info(f"Loaded {len(documents)} documents with matching text and json")
        return documents

    def build_index(self, 
                   documents: List[Document], 
                   chunk_size: int = 2048, 
                   chunk_overlap: int = 100) -> VectorStoreIndex:
        """
        Build a vector index from documents with configurable chunking and local embeddings.
        
        Args:
            documents: List of Document objects
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            
        Returns:
            VectorStoreIndex object
        """        

        logger.info(f"Building index with chunk size {chunk_size}, overlap {chunk_overlap}, and local embeddings")

        # Create node parser with configurable chunk size
        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Build the index with local embeddings
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[node_parser],
            embed_model=self.embed_model,
            show_progress=True
        )

        # Persist the index if a directory is specified
        if self.persist_dir:
            self.persist_dir.mkdir(exist_ok=True, parents=True)
            index.storage_context.persist(persist_dir=str(self.persist_dir))
            logger.info(f"Index persisted to {self.persist_dir}")

        self.index = index
        return index

    def load_index(self) -> Optional[VectorStoreIndex]:
        """
        Load the index from disk if it exists.
        
        Returns:
            VectorStoreIndex object or None if it doesn't exist
        """

        if self.persist_dir.exists():
            logger.info(f"Loading index from {self.persist_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_dir))

            # Apply the same embedding model when loading
            self.index = load_index_from_storage(
                storage_context,
                embed_model=self.embed_model
            )
            return self.index
        else:
            logger.warning(f"No index found at {self.persist_dir}")
            return None

    def retrieve_context(self, 
                        query: str, 
                        top_k: int = 5, 
                        filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieve relevant context with metadata and scores for a query.
        
        Args:
            query: The user query
            top_k: Number of top results to return
            filters: Optional metadata filters
            
        Returns:
            Dictionary with retrieved nodes and their metadata
        """        

        if not self.index:
            raise ValueError("No index available. Please build or load an index first.")
        logger.info(f"Retrieving context for query: '{query}' (top_k={top_k})")

        # Set up the retriever with parameters
        retriever = self.index.as_retriever(similarity_top_k=top_k)

        # Apply metadata filters if provided
        if filters:
            metadata_filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key=key,
                        value=value,
                        operator=FilterOperator.EQ
                    ) for key, value in filters.items()
                ]
            )
            retriever.filters = metadata_filters

        # Retrieve nodes    
        retrieved_nodes = retriever.retrieve(query)

        # Format the response
        result = {
            "query": query,
            "num_results": len(retrieved_nodes),
            "retrieved_chunks": []
        }

        # Add each retrieved chunk with metadata
        for node in retrieved_nodes:
            result["retrieved_chunks"].append({
                "text": node.node.text,
                "score": float(node.score),
                "metadata": node.node.metadata,
                "node_id": node.node.id_
            })
        return result

together.api_key = os.getenv("TOGETHER_API_KEY")

def clean_answer(text):
    lines = text.splitlines()
    seen = set()
    clean_lines = []
    for line in lines:
        if line.strip() not in seen:
            seen.add(line.strip())
            clean_lines.append(line)
    return "\n".join(clean_lines)

def generate_answer(context_chunks, question, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    context_text = "\n\n".join(chunk["text"] for chunk in context_chunks)
    prompt = f"""
            Answer the following legal question based only on the provided context. 
            Be concise and do not include unnecessary commentary or sign-offs.

            Context:
            {context_text}

            Question:
            {question}

            Answer:
            """
    try:
        response = together.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=700,
            temperature=0.3,
            top_p=0.9,
            stop=["\n\n", "Best regards", "Thank you"]
        )
        return clean_answer(response.choices[0].text.strip())
    except Exception as e:
        return f"⚠️ Error generating answer: {str(e)}"



import numpy as np
from sklearn.metrics import f1_score
from collections import Counter

def exact_match_score(prediction, ground_truth):
    """
    Computes Exact Match (EM) score.
    """
    return int(prediction == ground_truth)

def f1_score_metric(prediction, ground_truth):
    """
    Computes F1 score based on precision and recall.
    """
    # Tokenize the prediction and ground truth
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())

    # Calculate Precision and Recall
    common = pred_tokens.intersection(gt_tokens)
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gt_tokens) if gt_tokens else 0

    # F1 is the harmonic mean of precision and recall
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def mrr_score(predictions, ground_truths):
    """
    Computes Mean Reciprocal Rank (MRR) score.
    """
    ranks = []
    for pred, gt in zip(predictions, ground_truths):
        if gt in pred:
            ranks.append(1 / (pred.index(gt) + 1))  # Rank starts from 1
        else:
            ranks.append(0)
    return np.mean(ranks)

from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

# Load once
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def semantic_score(pred, gt):
    emb1 = semantic_model.encode(pred, convert_to_tensor=True)
    emb2 = semantic_model.encode(gt, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)[0][0])

def rouge_l_score(pred, gt):
    return rouge.score(gt, pred)['rougeL'].fmeasure

from bert_score import score as bert_score_fn

def compute_bert_score(predictions, references):
    P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return float(P.mean()), float(R.mean()), float(F1.mean())

def contains_gold_answer(prediction, ground_truth):
    return int(ground_truth.lower() in prediction.lower())

def evaluate_qna_system(
    json_folder_path: str,
    text_folder_path: str,
    top_k: int = 5,
    chunk_size: int = 2048,
    chunk_overlap: int = 100,
    persist_dir: Optional[str] = "./storage",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    show_examples: bool = False #added for additional metrics
) -> Dict[str, Any]:
    """
    Evaluate the entire QnA system using Exact Match (EM), F1, and MRR metrics.
    """
    retriever = DocumentRetriever(
        persist_dir=persist_dir,
        embedding_model_name=embedding_model
    )

    # Load and build from matched QA + real text content
    documents = retriever.load_documents_from_json_and_text(json_folder_path, text_folder_path)
    retriever.build_index(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    ground_truth_answers = []
    predicted_answers = []

    for doc in documents:
        question = doc.metadata.get("question")
        ground_truth = doc.metadata.get("ground_truth")
        # print("DEBUG:", doc.metadata)

        if not question or not ground_truth:
            continue

        result = retriever.retrieve_context(question, top_k=top_k)
        retrieved_context = result["retrieved_chunks"]
        answer = generate_answer(retrieved_context, question)

        if show_examples:
            print("\n" + "-"*60)
            print(f"📌 Question: {question}")
            print(f"✅ Ground Truth: {ground_truth}")
            print(f"🤖 Predicted: {answer}")

        ground_truth_answers.append(ground_truth)
        predicted_answers.append(answer)

    if not ground_truth_answers or not predicted_answers:
        print("No valid predictions or ground truth answers.")
        return {
            "exact_match_score": 0,
            "f1_score": 0,
            "rougeL_score": 0,
            "semantic_similarity": 0
        }

    em_scores = [exact_match_score(pred, gt) for pred, gt in zip(predicted_answers, ground_truth_answers)]
    f1_scores = [f1_score_metric(pred, gt) for pred, gt in zip(predicted_answers, ground_truth_answers)]
    mrr = mrr_score(predicted_answers, ground_truth_answers)

    rouge_scores = [rouge_l_score(pred, gt) for pred, gt in zip(predicted_answers, ground_truth_answers)]
    semantic_scores = [semantic_score(pred, gt) for pred, gt in zip(predicted_answers, ground_truth_answers)]

    gold_match_flags = [contains_gold_answer(p, g) for p, g in zip(predicted_answers, ground_truth_answers)]

    bert_p, bert_r, bert_f1 = compute_bert_score(predicted_answers, ground_truth_answers)

    return {
        "exact_match_score": np.mean(em_scores),
        "f1_score": np.mean(f1_scores),
        "mrr_score": np.mean(mrr),
        "rougeL_score": np.mean(rouge_scores),
        "semantic_similarity": np.mean(semantic_scores),
        "contains_gold_match_rate": np.mean(gold_match_flags),
        "bert_precision": bert_p,
        "bert_recall": bert_r,
        "bert_f1": bert_f1
    }



if __name__ == "__main__":
    load_dotenv()
    together.api_key = os.getenv("TOGETHER_API_KEY")

    json_folder = "/Users/simalanjum/Desktop/legal-rag-qa/testjson"
    text_folder = "/Users/simalanjum/Desktop/legal-rag-qa/testtxt"

    # json_folder = "/Users/simalanjum/Desktop/legal-rag-qa/datasets/processed_json"
    # text_folder = "/Users/simalanjum/Desktop/legal-rag-qa/corpus/cuad"
    
    results = evaluate_qna_system(
        json_folder_path=json_folder,
        text_folder_path=text_folder,
        top_k=5,
        # show_examples=True
    )

    print("\n=== Evaluation Results ===")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
