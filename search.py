import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

DB_PATH = "vector_db"
COLLECTION_NAME = "employee_handbook"

DEFAULT_RESULTS = 5


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# CONNECT TO CHROMADB
# ==========================================

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ==========================================
# VECTOR SEARCH
# ==========================================

def search(
    query,
    n_results=DEFAULT_RESULTS,
    page=None
):

    print()
    print("Creating query embedding...")

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )


    # ======================================
    # SEARCH
    # ======================================

    if page is not None:

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where={
                "page": page
            }
        )

    else:

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )


    # ======================================
    # GET RESULTS
    # ======================================

    documents = results.get(
        "documents"
    ) or [[]]

    metadatas = results.get(
        "metadatas"
    ) or [[]]

    distances = results.get(
        "distances"
    ) or [[]]


    documents = documents[0]

    metadatas = metadatas[0]

    distances = distances[0]


    # ======================================
    # DISPLAY
    # ======================================

    print()
    print("================================")
    print("VECTOR SEARCH RESULTS")
    print("================================")


    for i in range(len(documents)):

        print()
        print("--------------------------------")
        print(
            f"RESULT {i + 1}"
        )
        print("--------------------------------")

        print(
            f"Distance: "
            f"{distances[i]:.4f}"
        )

        print(
            f"Page: "
            f"{metadatas[i]['page']}"
        )

        print(
            f"Chunk: "
            f"{metadatas[i]['chunk_position']}"
        )

        print()

        print(
            documents[i]
        )


# ==========================================
# USER INTERFACE
# ==========================================

print()
print("================================")
print("HANDBOOK VECTOR SEARCH")
print("================================")

print("Type 'exit' to quit.")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    query = input(
        "\nAsk a question: "
    )


    if query.lower() == "exit":

        print("Goodbye!")

        break


    if not query.strip():

        continue


    number_input = input(
        "Number of results [5]: "
    )


    if number_input.strip():

        try:

            n_results = int(
                number_input
            )

        except ValueError:

            print(
                "Invalid number. Using 5."
            )

            n_results = 5

    else:

        n_results = 5


    page_input = input(
        "Page filter "
        "(press Enter to skip): "
    )


    page = None


    if page_input.strip():

        try:

            page = int(
                page_input
            )

        except ValueError:

            print(
                "Invalid page. "
                "Ignoring filter."
            )


    search(
        query=query,
        n_results=n_results,
        page=page
    )