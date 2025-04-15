# import json
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# import logging

# # LlamaIndex imports
# from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
# from llama_index.core.schema import Document, NodeWithScore
# from llama_index.core.node_parser import SentenceSplitter
# from llama_hub.file.pymu_pdf import PyMuPDFReader

# from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# import os
# import together
# from dotenv import load_dotenv

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class DocumentRetriever:
#     def __init__(self, 
#                  persist_dir: Optional[str] = "./storage",
#                  embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
#         """
#         Initialize the document retriever with local embeddings.
        
#         Args:
#             persist_dir: Directory to persist the index (optional)
#             embedding_model_name: Name of the HuggingFace embedding model to use
#         """
#         self.persist_dir = Path(persist_dir)
#         self.index = None
        
#         # Initialize the local embedding model
#         logger.info(f"Initializing local embedding model: {embedding_model_name}")
#         self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        
#     def load_documents(self, file_path: str) -> List[Document]:
#         """
#         Load documents from either PDF or TXT files.
        
#         Args:
#             file_path: Path to the file to load
            
#         Returns:
#             List of Document objects
#         """
#         file_path = Path(file_path)
#         ext = file_path.suffix.lower()
#         file_name = file_path.name
        
#         logger.info(f"Loading document: {file_name}")
        
#         if ext == ".pdf":
#             loader = PyMuPDFReader()
#             documents = loader.load(file=str(file_path))
#             # reader = PDFReader()
#             # documents = reader.load_data(file=str(file_path))
            
#             # Add file metadata to each document
#             for doc in documents:
#                 doc.metadata.update({
#                     "file_name": file_name,
#                     "file_type": "pdf",
#                     "source": str(file_path)
#                 })
                
#         elif ext == ".txt":
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 text = f.read()
#             documents = [Document(text=text, metadata={
#                 "file_name": file_name,
#                 "file_type": "txt",
#                 "source": str(file_path)
#             })]
#         else:
#             raise ValueError(f"Unsupported file type: {ext}")
            
#         logger.info(f"Loaded {len(documents)} document(s)")
#         return documents
    
#     def build_index(self, 
#                    documents: List[Document], 
#                    chunk_size: int = 1024, 
#                    chunk_overlap: int = 20) -> VectorStoreIndex:
#         """
#         Build a vector index from documents with configurable chunking and local embeddings.
        
#         Args:
#             documents: List of Document objects
#             chunk_size: Size of text chunks for splitting
#             chunk_overlap: Overlap between chunks
            
#         Returns:
#             VectorStoreIndex object
#         """
#         logger.info(f"Building index with chunk size {chunk_size}, overlap {chunk_overlap}, and local embeddings")
        
#         # Create node parser with configurable chunk size
#         node_parser = SentenceSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap
#         )
        
#         # Build the index with local embeddings
#         index = VectorStoreIndex.from_documents(
#             documents,
#             transformations=[node_parser],
#             embed_model=self.embed_model,  # Use the local embedding model
#             show_progress=True
#         )
        
#         # Persist the index if a directory is specified
#         if self.persist_dir:
#             self.persist_dir.mkdir(exist_ok=True, parents=True)
#             index.storage_context.persist(persist_dir=str(self.persist_dir))
#             logger.info(f"Index persisted to {self.persist_dir}")
            
#         self.index = index
#         return index
    
#     def load_index(self) -> Optional[VectorStoreIndex]:
#         """
#         Load the index from disk if it exists.
        
#         Returns:
#             VectorStoreIndex object or None if it doesn't exist
#         """
#         if self.persist_dir.exists():
#             logger.info(f"Loading index from {self.persist_dir}")
#             storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
#             # Apply the same embedding model when loading
#             self.index = load_index_from_storage(
#                 storage_context,
#                 embed_model=self.embed_model
#             )
#             return self.index
#         else:
#             logger.warning(f"No index found at {self.persist_dir}")
#             return None
    
#     def retrieve_context(self, 
#                         query: str, 
#                         top_k: int = 5, 
#                         filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Retrieve relevant context with metadata and scores for a query.
        
#         Args:
#             query: The user query
#             top_k: Number of top results to return
#             filters: Optional metadata filters
            
#         Returns:
#             Dictionary with retrieved nodes and their metadata
#         """
#         if not self.index:
#             raise ValueError("No index available. Please build or load an index first.")
        
#         logger.info(f"Retrieving context for query: '{query}' (top_k={top_k})")
        
#         # Set up the retriever with parameters
#         retriever = self.index.as_retriever(
#             similarity_top_k=top_k
#         )
        
#         # Apply metadata filters if provided
#         if filters:
#             metadata_filters = MetadataFilters(
#                 filters=[
#                     MetadataFilter(
#                         key=key,
#                         value=value,
#                         operator=FilterOperator.EQ
#                     ) for key, value in filters.items()
#                 ]
#             )
#             retriever.filters = metadata_filters
        
#         # Retrieve nodes
#         retrieved_nodes = retriever.retrieve(query)
        
#         # Format the response
#         result = {
#             "query": query,
#             "num_results": len(retrieved_nodes),
#             "retrieved_chunks": []
#         }
        
#         # Add each retrieved chunk with metadata
#         for node in retrieved_nodes:
#             result["retrieved_chunks"].append({
#                 "text": node.node.text,
#                 "score": float(node.score),  # Convert to float for JSON serialization
#                 "metadata": node.node.metadata,
#                 "node_id": node.node.id_
#             })
            
#         return result

# def process_file_and_query(
#     file_path: str, 
#     query: str, 
#     top_k: int = 5, 
#     chunk_size: int = 1024, 
#     chunk_overlap: int = 20,
#     persist_dir: Optional[str] = "./storage",
#     embedding_model: str = "BAAI/bge-small-en-v1.5"
# ) -> Dict[str, Any]:
#     """
#     Process a file and query, returning relevant context with metadata.
#     Uses local embedding model to avoid API rate limits.
    
#     Args:
#         file_path: Path to the PDF or TXT file
#         query: User query
#         top_k: Number of top results to return
#         chunk_size: Size of text chunks
#         chunk_overlap: Overlap between chunks
#         persist_dir: Directory to persist the index
#         embedding_model: HuggingFace model name for embeddings
        
#     Returns:
#         Dictionary with retrieved context and metadata
#     """
#     retriever = DocumentRetriever(
#         persist_dir=persist_dir,
#         embedding_model_name=embedding_model
#     )
    
#     # Try to load existing index first
#     if retriever.load_index() is None:
#         # If no index exists, build it
#         documents = retriever.load_documents(file_path)
#         retriever.build_index(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
#     # Retrieve context for the query
#     result = retriever.retrieve_context(query, top_k=top_k)
#     return result


# load_dotenv()
# together.api_key = os.getenv("TOGETHER_API_KEY")

# def generate_answer(context, question, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
#     prompt = f"""
# You are a helpful legal assistant. Carefully read the following contract excerpts and answer the question truthfully and concisely based only on the given context.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     try:
#         response = together.Completion.create(
#             model=model,
#             prompt=prompt,
#             max_tokens=700,
#             temperature=0.3,
#             top_p=0.9,
#         )
#         return response.choices[0].text.strip()
#     except Exception as e:
#         return f"⚠️ Error generating answer: {str(e)}"
    

# # Example usage
# if __name__ == "__main__":
#     # import argparse
    
#     # parser = argparse.ArgumentParser(description="Retrieve context from documents using LlamaIndex")
#     # parser.add_argument("file_path", help="Path to the PDF or TXT file")
#     # parser.add_argument("query", help="User query")
#     # parser.add_argument("--top-k", type=int, default=5, help="Number of top results to return")
#     # parser.add_argument("--chunk-size", type=int, default=1024, help="Size of text chunks")
#     # parser.add_argument("--chunk-overlap", type=int, default=20, help="Overlap between chunks")
#     # parser.add_argument("--persist-dir", default="./storage", help="Directory to persist the index")
    
#     # args = parser.parse_args()
    
#     # result = process_file_and_query(
#     #     args.file_path,
#     #     args.query,
#     #     top_k=args.top_k,
#     #     chunk_size=args.chunk_size,
#     #     chunk_overlap=args.chunk_overlap,
#     #     persist_dir=args.persist_dir
#     # )
    
#     query1= "Summarize the main points from the document."
#     result = process_file_and_query(
#     "/Users/simalanjum/Downloads/legal-rag-qa/ACCURAYINC_09_01_2010-EX-10.31-DISTRIBUTOR AGREEMENT.txt", 
#     query1,
#     top_k=5
#     )

#     # The returned context can be passed to your LLM
#     retrieved_context = result["retrieved_chunks"]
    
#     # Generate an answer using the retrieved context
#     answer = generate_answer(retrieved_context, query1)
    
#     # Print results
#     print("\n=== Retrieved Context ===")
#     print(json.dumps(result, indent=2))
    
#     print("\n=== Generated Answer ===")
#     print(answer)



import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.node_parser import SentenceSplitter
from llama_hub.file.pymu_pdf import PyMuPDFReader

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
        self.persist_dir = Path(persist_dir)
        self.index = None
        logger.info(f"Initializing local embedding model: {embedding_model_name}")
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

    def load_documents(self, file_path: str) -> List[Document]:
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        file_name = file_path.name
        logger.info(f"Loading document: {file_name}")

        if ext == ".pdf":
            loader = PyMuPDFReader()
            documents = loader.load(file=str(file_path))
            for doc in documents:
                doc.metadata.update({
                    "file_name": file_name,
                    "file_type": "pdf",
                    "source": str(file_path)
                })
        elif ext == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            documents = [Document(text=text, metadata={
                "file_name": file_name,
                "file_type": "txt",
                "source": str(file_path)
            })]
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.info(f"Loaded {len(documents)} document(s)")
        return documents

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
        logger.info(f"Building index with chunk size {chunk_size}, overlap {chunk_overlap}, and local embeddings")
        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[node_parser],
            embed_model=self.embed_model,
            show_progress=True
        )
        if self.persist_dir:
            self.persist_dir.mkdir(exist_ok=True, parents=True)
            index.storage_context.persist(persist_dir=str(self.persist_dir))
            logger.info(f"Index persisted to {self.persist_dir}")
        self.index = index
        return index

    def load_index(self) -> Optional[VectorStoreIndex]:
        if self.persist_dir.exists():
            logger.info(f"Loading index from {self.persist_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
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
        if not self.index:
            raise ValueError("No index available. Please build or load an index first.")
        logger.info(f"Retrieving context for query: '{query}' (top_k={top_k})")
        retriever = self.index.as_retriever(similarity_top_k=top_k)
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
        retrieved_nodes = retriever.retrieve(query)
        result = {
            "query": query,
            "num_results": len(retrieved_nodes),
            "retrieved_chunks": []
        }
        for node in retrieved_nodes:
            result["retrieved_chunks"].append({
                "text": node.node.text,
                "score": float(node.score),
                "metadata": node.node.metadata,
                "node_id": node.node.id_
            })
        return result

def process_file_and_query(
    file_path: str, 
    query: str, 
    top_k: int = 5, 
    chunk_size: int = 1024, 
    chunk_overlap: int = 20,
    persist_dir: Optional[str] = "./storage",
    embedding_model: str = "BAAI/bge-small-en-v1.5"
) -> Dict[str, Any]:
    retriever = DocumentRetriever(
        persist_dir=persist_dir,
        embedding_model_name=embedding_model
    )
    if retriever.load_index() is None:
        documents = retriever.load_documents(file_path)
        retriever.build_index(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    result = retriever.retrieve_context(query, top_k=top_k)
    return result

def evaluate_documents(
    json_folder_path: str, 
    query: str, 
    top_k: int = 5, 
    chunk_size: int = 1024, 
    chunk_overlap: int = 20,
    persist_dir: Optional[str] = "./storage",
    embedding_model: str = "BAAI/bge-small-en-v1.5"
) -> Dict[str, Any]:
    """
    Evaluate a list of documents by retrieving context and generating answers for each document.
    
    Args:
        json_folder_path: Path to the folder containing JSON documents
        query: User query
        top_k: Number of top results to return
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        persist_dir: Directory to persist the index
        embedding_model: HuggingFace model name for embeddings
        
    Returns:
        Dictionary with retrieved context and answers for each document
    """
    retriever = DocumentRetriever(
        persist_dir=persist_dir,
        embedding_model_name=embedding_model
    )
    
    # Load documents, build index and retrieve context for each document
    results = {}
    
    # Load documents from the JSON folder
    documents = retriever.load_documents_from_json(json_folder_path)
    
    # Build index from documents
    retriever.build_index(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Retrieve context for the query
    result = retriever.retrieve_context(query, top_k=top_k)
    retrieved_context = result["retrieved_chunks"]
    
    # Generate an answer using the retrieved context
    answer = generate_answer(retrieved_context, query)
    
    # Store the result
    results[json_folder_path] = {
        "context": result,
        "answer": answer
    }
    
    return results

# load_dotenv()
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
Answer the following legal question based only on the provided context. Be concise and do not include unnecessary commentary or sign-offs.

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

def evaluate_qna_system(
    json_folder_path: str,
    text_folder_path: str,
    top_k: int = 5,
    chunk_size: int = 2048,
    chunk_overlap: int = 100,
    persist_dir: Optional[str] = "./storage",
    embedding_model: str = "BAAI/bge-small-en-v1.5"
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

        print(f"\nQuestion: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Predicted Answer: {answer}")

        ground_truth_answers.append(ground_truth)
        predicted_answers.append(answer)

    if not ground_truth_answers or not predicted_answers:
        print("No valid predictions or ground truth answers.")
        return {"exact_match_score": 0, "f1_score": 0, "mrr_score": 0}

    em_scores = [exact_match_score(pred, gt) for pred, gt in zip(predicted_answers, ground_truth_answers)]
    f1_scores = [f1_score_metric(pred, gt) for pred, gt in zip(predicted_answers, ground_truth_answers)]
    mrr = mrr_score(predicted_answers, ground_truth_answers)

    return {
        "exact_match_score": np.mean(em_scores),
        "f1_score": np.mean(f1_scores),
        "mrr_score": mrr
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
        top_k=5
    )

    print("\n=== Evaluation Results ===")
    print(f"Exact Match Score: {results['exact_match_score']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"MRR Score: {results['mrr_score']:.4f}")
