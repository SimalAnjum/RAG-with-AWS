#!/usr/local/bin/python3.11

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
# from llama_hub.file.pymu_pdf import PyMuPDFReader
from llama_index.readers.file import PyMuPDFReader

from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import os
import together
from dotenv import load_dotenv
import uvicorn

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import shutil
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    def load_documents(self, file_path: str) -> List[Document]:
        """
        Load documents from either PDF or TXT files.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            List of Document objects
        """

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
        result = {
            "query": query,
            "num_results": len(retrieved_nodes),
            "retrieved_chunks": []
        }

        # Format the response
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
    """
    Process a file and query, returning relevant context with metadata.
    Uses local embedding model to avoid API rate limits.
    
    Args:
        file_path: Path to the PDF or TXT file
        query: User query
        top_k: Number of top results to return
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        persist_dir: Directory to persist the index
        embedding_model: HuggingFace model name for embeddings
        
    Returns:
        Dictionary with retrieved context and metadata
    """       

    retriever = DocumentRetriever(
        persist_dir=persist_dir,
        embedding_model_name=embedding_model
    )

    # Try to load existing index first
    if retriever.load_index() is None:
        # If no index exists, build it
        documents = retriever.load_documents(file_path)
        retriever.build_index(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Retrieve context for the query    
    result = retriever.retrieve_context(query, top_k=top_k)
    return result


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

@app.post("/rag")
async def rag_endpoint(file: UploadFile, query: str = Form(...)):
    temp_path = f"./uploads/{file.filename}"
    os.makedirs("./uploads", exist_ok=True)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    retriever = DocumentRetriever()
    if retriever.load_index() is None:
        docs = retriever.load_documents(temp_path)
        retriever.build_index(docs)

    context = retriever.retrieve_context(query)
    answer = generate_answer(context["retrieved_chunks"], query)
    return JSONResponse(content={"answer": answer})

if __name__ == "__main__":
    load_dotenv()
    together.api_key = os.getenv("TOGETHER_API_KEY")
    uvicorn.run(app, host="0.0.0.0", port=8000)