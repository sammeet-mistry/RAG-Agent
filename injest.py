import fitz
import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

PDF_PATH = "data/handbook.pdf"
DB_PATH = "vector_db"
COLLECTION_NAME = "employee_handbook"

SKIP_PAGES = 3


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# EXTRACT PDF PAGES
# ==========================================

def extract_pages(pdf_path):

    pdf = fitz.open(pdf_path)

    pages = []

    for page_number in range(len(pdf)):

        page = pdf[page_number]

        # Skip table of contents pages
        if page_number < SKIP_PAGES:
            continue

        # Explicitly request normal text
        raw_text = page.get_text("text")

        # Make sure Pylance knows this is a string
        text = str(raw_text).strip()

        # Ignore empty pages
        if not text:
            continue

        pages.append({
            "page": page_number + 1,
            "text": text
        })

    pdf.close()

    return pages


# ==========================================
# CHUNK TEXT
# ==========================================

def chunk_text(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================================
# READ PDF
# ==========================================

print("Reading PDF...")

pages = extract_pages(
    PDF_PATH
)

print(
    f"Pages indexed: {len(pages)}"
)


# ==========================================
# CREATE CHUNKS
# ==========================================

all_chunks = []

for page in pages:

    chunks = chunk_text(
        page["text"]
    )

    for chunk_number, chunk in enumerate(
        chunks
    ):

        all_chunks.append({

            "text": chunk,

            "page": page["page"],

            "document": "handbook.pdf",

            "chunk_id":
                f"page_{page['page']}_chunk_{chunk_number}",

            "chunk_position":
                chunk_number

        })


print(
    f"Chunks created: {len(all_chunks)}"
)


# ==========================================
# CREATE EMBEDDINGS
# ==========================================

print("Creating embeddings...")

texts = [

    chunk["text"]

    for chunk in all_chunks

]


embeddings = model.encode(

    texts,

    normalize_embeddings=True

)


print(
    "Embeddings created."
)


# ==========================================
# CONNECT TO CHROMADB
# ==========================================

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(

    path=DB_PATH

)


# ==========================================
# DELETE OLD COLLECTION
# ==========================================

try:

    client.delete_collection(

        name=COLLECTION_NAME

    )

    print(
        "Old collection deleted."
    )

except Exception:

    pass


# ==========================================
# CREATE NEW COLLECTION
# ==========================================

collection = client.create_collection(

    name=COLLECTION_NAME

)


# ==========================================
# PREPARE DATABASE DATA
# ==========================================

ids = []

documents = []

metadatas = []


for chunk in all_chunks:

    ids.append(
        chunk["chunk_id"]
    )

    documents.append(
        chunk["text"]
    )

    metadatas.append({

        "document":
            chunk["document"],

        "page":
            chunk["page"],

        "chunk_position":
            chunk["chunk_position"]

    })


# ==========================================
# SAVE TO CHROMADB
# ==========================================

print(
    "Saving data to ChromaDB..."
)


collection.add(

    ids=ids,

    documents=documents,

    embeddings=embeddings.tolist(),

    metadatas=metadatas

)


# ==========================================
# FINISHED
# ==========================================

print()
print("================================")
print("INGESTION COMPLETE")
print("================================")

print(
    f"Chunks stored: {collection.count()}"
)

print(
    f"Skipped first {SKIP_PAGES} pages."
)

print(
    f"Database location: {DB_PATH}"
)