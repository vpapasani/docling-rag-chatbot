import os
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from sqlalchemy import create_engine, text

from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# 1. Define your documents + domains
# -----------------------------
docs_info = [
    {"path": "/Users/xai/documents/RAG/Abu Dhabi Procurement Standards.pdf", "domain": "procurement"},
    {"path": "/Users/xai/documents/RAG/Procurement Manual (Ariba Aligned).pdf", "domain": "procurement"},
    {"path": "/Users/xai/documents/RAG/Procurement Manual (Business Process).pdf", "domain": "procurement"},
    {"path": "/Users/xai/documents/RAG/HR Bylaws.pdf", "domain": "hr"},
    {"path": "/Users/xai/documents/RAG/Inforamation Security.pdf", "domain": "security"},
]

# -----------------------------
# 2. Initialize converter, splitter, embeddings
# -----------------------------
converter = DocumentConverter()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "mps"},   # Apple M1/M2 GPU, fallback: "cpu"
    encode_kwargs={"batch_size": 32}
)

# Postgres DB URL
DATABASE_URL = os.getenv("DATABASE_URL")

# A collection name for pgvector (like a namespace)
COLLECTION_NAME = "company_documents"

# -----------------------------
# 3. Process documents
# -----------------------------
all_texts = []
all_metadata = []

for doc_info in docs_info:
    filepath = doc_info["path"]
    domain = doc_info["domain"]
    filename = os.path.basename(filepath)

    try:
        # Convert document
        result = converter.convert(filepath)
        doc = result.document
        markdown_text = doc.export_to_markdown()

        # Split into chunks
        chunks = splitter.split_text(markdown_text)

        # Add metadata for each chunk
        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadata.append({
                "domain": domain,
                "source": filename,
                "chunk_id": idx
            })

        print(f"✅ Processed {filename} ({len(chunks)} chunks)")

    except Exception as e:
        print(f"⚠️ Failed to process {filename}: {e}")

# -----------------------------
# 4. Store in Postgres (pgvector)
# -----------------------------
print("\n💾 Storing chunks into pgvector...")

vectordb = PGVector.from_texts(
    embedding=embeddings,
    texts=all_texts,
    metadatas=all_metadata,
    collection_name=COLLECTION_NAME,
    connection_string=os.getenv("DATABASE_URL")  # your env var
)



print("✅ All documents stored in Postgres pgvector!")

# -----------------------------
# 5. Example Query
# -----------------------------
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

query = "What are the procurement tender rules?"
results = retriever.get_relevant_documents(query)

print("\n🔎 Query Results:")
for r in results:
    print(f"- ({r.metadata['domain']}) {r.metadata['source']}: {r.page_content[:200]}...")
