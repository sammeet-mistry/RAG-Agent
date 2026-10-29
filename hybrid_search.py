import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ==========================================
# CONFIGURATION
# ==========================================

DB_PATH = "vector_db"
COLLECTION_NAME = "employee_handbook"

VECTOR_RESULTS = 10
FINAL_RESULTS = 5

VECTOR_WEIGHT = 0.70
KEYWORD_WEIGHT = 0.30


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
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
# LOAD ALL DOCUMENTS
# ==========================================

print("Loading documents...")

data = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)

documents = data["documents"] or []

metadatas = data["metadatas"] or []


print(
    f"Documents loaded: "
    f"{len(documents)}"
)


# ==========================================
# CREATE BM25 INDEX
# ==========================================

print("Creating BM25 index...")

tokenized_documents = [

    document.lower().split()

    for document in documents

]

bm25 = BM25Okapi(
    tokenized_documents
)


# ==========================================
# NORMALIZE
# ==========================================

def normalize_scores(scores):

    if not scores:

        return {}


    maximum = max(
        scores.values()
    )


    if maximum == 0:

        return {
            index: 0.0
            for index in scores
        }


    return {

        index:
            score / maximum

        for index, score in scores.items()

    }


# ==========================================
# HYBRID SEARCH
# ==========================================

def hybrid_search(query):

    # ======================================
    # VECTOR SEARCH
    # ======================================

    print(
        "\nRunning vector search..."
    )


    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )


    vector_results = collection.query(

        query_embeddings=
            query_embedding.tolist(),

        n_results=
            VECTOR_RESULTS

    )


    vector_documents = (
        vector_results.get(
            "documents"
        ) or [[]]
    )[0]


    vector_distances = (
        vector_results.get(
            "distances"
        ) or [[]]
    )[0]


    vector_metadatas = (
        vector_results.get(
            "metadatas"
        ) or [[]]
    )[0]


    # ======================================
    # VECTOR SCORES
    # ======================================

    vector_scores = {}


    for i in range(
        len(vector_documents)
    ):

        distance = (
            vector_distances[i]
        )


        similarity = (
            1.0 / (1.0 + distance)
        )


        page = (
            vector_metadatas[i]["page"]
        )


        chunk_position = (
            vector_metadatas[i]
            ["chunk_position"]
        )


        for index, metadata in enumerate(
            metadatas
        ):

            if (
                metadata["page"]
                == page
                and
                metadata["chunk_position"]
                == chunk_position
            ):

                vector_scores[index] = (
                    similarity
                )

                break


    # ======================================
    # KEYWORD SEARCH
    # ======================================

    print(
        "Running keyword search..."
    )


    query_tokens = (
        query.lower().split()
    )


    bm25_scores = bm25.get_scores(
        query_tokens
    )


    keyword_scores = {}


    for index, score in enumerate(
        bm25_scores
    ):

        if score > 0:

            keyword_scores[index] = (
                float(score)
            )


    # ======================================
    # NORMALIZE
    # ======================================

    vector_scores = normalize_scores(
        vector_scores
    )


    keyword_scores = normalize_scores(
        keyword_scores
    )


    # ======================================
    # COMBINE
    # ======================================

    combined_scores = {}


    all_indices = set(
        vector_scores.keys()
    ).union(
        keyword_scores.keys()
    )


    for index in all_indices:

        vector_score = (
            vector_scores.get(
                index,
                0.0
            )
        )


        keyword_score = (
            keyword_scores.get(
                index,
                0.0
            )
        )


        combined_score = (

            VECTOR_WEIGHT
            * vector_score

            +

            KEYWORD_WEIGHT
            * keyword_score

        )


        combined_scores[index] = (
            combined_score
        )


    # ======================================
    # SORT
    # ======================================

    ranked_results = sorted(

        combined_scores.items(),

        key=lambda item: item[1],

        reverse=True

    )


    # ======================================
    # DISPLAY
    # ======================================

    print()
    print("================================")
    print("HYBRID SEARCH RESULTS")
    print("================================")


    for rank, (
        index,
        score
    ) in enumerate(
        ranked_results[:FINAL_RESULTS],
        start=1
    ):

        print()
        print("--------------------------------")
        print(
            f"RESULT {rank}"
        )
        print("--------------------------------")

        print(
            f"Hybrid score: "
            f"{score:.4f}"
        )

        print(
            f"Page: "
            f"{metadatas[index]['page']}"
        )

        print(
            f"Chunk: "
            f"{metadatas[index]['chunk_position']}"
        )

        print()

        print(
            documents[index]
        )


# ==========================================
# USER INTERFACE
# ==========================================

print()
print("================================")
print("HYBRID RAG SEARCH")
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


    hybrid_search(
        query
    )