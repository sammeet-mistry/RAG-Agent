import chromadb


# ==========================================
# CONFIGURATION
# ==========================================

DB_PATH = "vector_db"
COLLECTION_NAME = "employee_handbook"


# ==========================================
# CONNECT TO CHROMADB
# ==========================================

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=DB_PATH
)


# ==========================================
# GET COLLECTION
# ==========================================

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ==========================================
# GET DATABASE DATA
# ==========================================

results = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)


# ==========================================
# SAFELY GET DOCUMENTS
# ==========================================

documents = results.get("documents")

if documents is None:
    documents = []


# ==========================================
# SAFELY GET METADATA
# ==========================================

metadatas = results.get("metadatas")

if metadatas is None:
    metadatas = []


# ==========================================
# DATABASE SUMMARY
# ==========================================

print()
print("==============================")
print("DATABASE CHECK")
print("==============================")

print(
    f"Total chunks: {len(documents)}"
)


# ==========================================
# SHOW FIRST FEW CHUNKS
# ==========================================

number_to_show = min(
    5,
    len(documents)
)


for i in range(number_to_show):

    print()
    print("==============================")
    print(f"CHUNK {i + 1}")
    print("==============================")

    print(
        f"Text:\n{documents[i][:500]}"
    )

    if i < len(metadatas):

        metadata = metadatas[i]

        if metadata is None:
            print("Metadata: None")
        else:
            print(
                f"Metadata: {metadata}"
            )


# ==========================================
# FINISHED
# ==========================================

print()
print("==============================")
print("DATABASE CHECK COMPLETE")
print("==============================")