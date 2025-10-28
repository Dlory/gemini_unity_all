import google.genai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import uuid
from sentence_transformers import SentenceTransformer
# --- 1. Configuration and Initialization ---

gemini_client = genai.Client() 

# Qdrant Client Settings (ensure Qdrant service is running)
qdrant_client =  QdrantClient( url="https://f82ce5af-b90f-42f9-9289-84dd659000ed.us-east-1-1.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.N-RrLg9mUtbOdbTo9V7KHjIk498rCFCYmKMcOPKrusI",)

print(qdrant_client.get_collections())
PDF_FILE_PATH = "knowledge_base.pdf" 
collection_name = f"pdf_rag_collection_{PDF_FILE_PATH}"
vector_size = 768 # Dimension of 'all-mpnet-base-v2'

embed_model  = SentenceTransformer("all-mpnet-base-v2")


def get_local_embeddings(chunks):
    embeddings = embed_model.encode(chunks, convert_to_numpy=True).tolist()
    return embeddings

# --- 2. Load and Split PDF File ---
def process_pdf_to_chunks(pdf_path: str):
    """Loads a PDF, extracts text, and splits it into small chunks."""
    
    print(f"-> Loading PDF file: {pdf_path}")
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    # Initialize the text splitter
    # The recursive splitter generally works better as it tries to maintain the semantic structure of the text.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Maximum number of characters per chunk
        chunk_overlap=200,     # Overlap between chunks
        separators=["\n\n", "\n", " ", ""], 
        length_function=len,
    )
    
    # Split the text
    chunks = text_splitter.split_text(full_text)
    print(f"-> Text has been split into {len(chunks)} chunks.")
    
    return chunks

# --- 3. Embed and Store in Qdrant ---

def index_chunks_to_qdrant(chunks: list[str], qdrant_client: QdrantClient):
    """Embeds text chunks and uploads them to Qdrant."""
    
    # Recreate Qdrant Collection
    if qdrant_client.collection_exists(collection_name=collection_name):
            # If the collection exists, delete it first
            qdrant_client.delete_collection(collection_name=collection_name)
            print(f"-> Existing Qdrant collection '{collection_name}' has been deleted.")
            
    # Create Qdrant Collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"-> Qdrant collection '{collection_name}' has been created.")
    
    points = []
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        chunk_batch = chunks[i : i + batch_size]
        
        embeddings_batch = get_local_embeddings(chunk_batch)
        
        # Construct PointStruct
        for (text, vector) in zip(chunk_batch, embeddings_batch):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text}
            ))

    # Batch upload to Qdrant
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True
    )
    print(f"-> Successfully indexed {len(points)} text chunks.")

# --- 4. RAG Process ---

def rag_query(query: str, client: genai.Client, qdrant_client: QdrantClient):
    """Executes the RAG process: retrieves relevant context and generates an answer with Gemini."""
    
    # 1. Generate query embedding
    # query_embedding_result = client.models.embed_content(
    #     model="gemini-embedding-001",
    #     content=query,
    #     task_type="RETRIEVAL_QUERY" # Use RETRIEVAL_QUERY for retrieval queries
    # )
    query_vector = get_local_embeddings(query)
        
    # 2. Search Qdrant
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3,
    )
    
    # 3. Construct RAG prompt
    context = "\n---\n".join([hit.payload['text'] for hit in search_results])
    
    rag_prompt = f'''
    Please use the following provided context to answer the user's question.
    If some information in the context is relevant to the question, please ask the user for more details.
    If you cannot find an answer relevant to the question in the context at all, please politely state that you could not find the information in the provided document.
    
    Context:
    ---
    {context}
    ---
    
    Question: {query}
    '''
    
    # 4. Generate the final answer using Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=rag_prompt
    )
    
    # Print the retrieved sources
    print("\n--- Retrieved Source Snippets ---")
    for i, hit in enumerate(search_results):
        print(f"Source {i+1} (Similarity: {hit.score:.4f}): {hit.payload['text'][:80]}...")
        
    print("\n--- Gemini RAG Answer ---")
    return response.text

# --- Main Execution Logic ---
if __name__ == "__main__":
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: File '{PDF_FILE_PATH}' not found. Please make sure the file exists.")
    else:
        if qdrant_client.collection_exists(collection_name=collection_name)==False:
            # Steps 1, 2 & 3: Process PDF and index
            chunks = process_pdf_to_chunks(PDF_FILE_PATH)
            index_chunks_to_qdrant(chunks, qdrant_client)

        # Step 4: Ask a question
        user_question = "Tell me Daily maintenance after power-on"
        answer = rag_query(user_question, gemini_client, qdrant_client)
        print(answer)