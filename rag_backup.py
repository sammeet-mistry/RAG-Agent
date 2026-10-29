import chromadb
import ollama

from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

DB_PATH = "vector_db"
COLLECTION_NAME = "employee_handbook"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"

TOP_K = 5

MAX_DISTANCE = 1.10

MAX_MEMORY_MESSAGES = 6


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
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
# MEMORY
# ==========================================

conversation_history = []


# ==========================================
# MEMORY TEXT
# ==========================================

def build_memory():

    if not conversation_history:
        return "No previous conversation."

    parts = []

    for message in conversation_history:

        parts.append(
            f"{message['role']}: "
            f"{message['content']}"
        )

    return "\n".join(parts)


# ==========================================
# REWRITE QUERY
# ==========================================

def rewrite_query(question):

    if not conversation_history:

        return question

    memory = build_memory()

    prompt = f"""
Rewrite the user's question into a standalone
search query.

Do not answer the question.

Use the conversation to understand words
such as "it", "this", "that", "they", or
"the policy".

Previous conversation:

{memory}

Current question:

{question}

Return ONLY the standalone search query.
"""

    response = ollama.chat(

        model=OLLAMA_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    rewritten = response["message"]["content"].strip()

    return rewritten


# ==========================================
# VECTOR SEARCH
# ==========================================

def retrieve(query):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    results = collection.query(

        query_embeddings=query_embedding.tolist(),

        n_results=TOP_K

    )

    documents = results ["documents"][0]

    metadatas = results ["metadatas"][0]

    distances = results ["distances"][0]

    return (
        documents,
        metadatas,
        distances
    )


# ==========================================
# BUILD CONTEXT
# ==========================================

def build_context(
    documents,
    metadatas
):

    parts = []

    for i in range(
        len(documents)
    ):

        page = metadatas[i]["page"]

        text = documents[i]

        parts.append(
            f"""
SOURCE: handbook.pdf
PAGE: {page}

{text}
"""
        )

    return "\n".join(parts)


# ==========================================
# GENERATE ANSWER
# ==========================================

def generate_answer(
    question,
    context
):

    memory = build_memory()

    prompt = f"""
You are an employee handbook assistant.

Answer the user's question using ONLY
the employee handbook context below.

If the answer is not contained in the
context, say:

"I couldn't find that information
in the employee handbook."

Do not use outside knowledge.

Do not invent information.

Previous conversation:

{memory}


EMPLOYEE HANDBOOK CONTEXT:

{context}


CURRENT QUESTION:

{question}


Give a clear and concise answer.
"""

    print()
    print("Sending context to Ollama...")

    response = ollama.chat(

        model=OLLAMA_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ==========================================
# SOURCES
# ==========================================

def get_sources(metadatas):

    sources = []

    for metadata in metadatas:

        page = metadata["page"]

        source = (
            f"handbook.pdf - Page {page}"
        )

        if source not in sources:

            sources.append(source)

    return sources


# ==========================================
# SAVE MEMORY
# ==========================================

def save_memory(
    question,
    answer
):

    conversation_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    conversation_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    while len(
        conversation_history
    ) > MAX_MEMORY_MESSAGES:

        conversation_history.pop(0)


# ==========================================
# RAG
# ==========================================

def rag(question):

    # --------------------------------------
    # QUERY REWRITE
    # --------------------------------------

    search_query = rewrite_query(
        question
    )

    print()
    print(
        "SEARCH QUERY:"
    )

    print(
        search_query
    )


    # --------------------------------------
    # SEARCH
    # --------------------------------------

    print()
    print(
        "Searching handbook..."
    )

    (
        documents,
        metadatas,
        distances
    ) = retrieve(
        search_query
    )


    # --------------------------------------
    # DISTANCE
    # --------------------------------------

    best_distance = min(
        distances
    )

    print()
    print(
        f"Best vector distance: "
        f"{best_distance:.4f}"
    )


    # --------------------------------------
    # SHOW CHUNKS
    # --------------------------------------

    print()
    print(
        "RETRIEVED CHUNKS"
    )

    print(
        "=============================="
    )


    for i in range(
        len(documents)
    ):

        print()
        print(
            f"CHUNK {i + 1}"
        )

        print(
            f"Page: "
            f"{metadatas[i]['page']}"
        )

        print(
            f"Distance: "
            f"{distances[i]:.4f}"
        )

        print(
            documents[i][:400]
        )


    # --------------------------------------
    # CONFIDENCE CHECK
    # --------------------------------------

    if best_distance > MAX_DISTANCE:

        answer = (
            "I couldn't find that information "
            "in the employee handbook."
        )

        print()
        print(
            "================================"
        )

        print(
            "ANSWER"
        )

        print(
            "================================"
        )

        print(answer)

        save_memory(
            question,
            answer
        )

        return


    # --------------------------------------
    # BUILD CONTEXT
    # --------------------------------------

    context = build_context(
        documents,
        metadatas
    )


    # --------------------------------------
    # GENERATE
    # --------------------------------------

    answer = generate_answer(
        question,
        context
    )


    # --------------------------------------
    # SAVE MEMORY
    # --------------------------------------

    save_memory(
        question,
        answer
    )


    # --------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------

    print()
    print(
        "================================"
    )

    print(
        "ANSWER"
    )

    print(
        "================================"
    )

    print(answer)


    # --------------------------------------
    # SOURCES
    # --------------------------------------

    sources = get_sources(
        metadatas
    )

    print()
    print(
        "================================"
    )

    print(
        "SOURCES"
    )

    print(
        "================================"
    )

    for source in sources:

        print(
            f"- {source}"
        )


# ==========================================
# START
# ==========================================

print()
print(
    "================================"
)

print(
    "EMPLOYEE HANDBOOK RAG"
)

print(
    "================================"
)

print(
    f"Ollama model: {OLLAMA_MODEL}"
)

print(
    "Query rewriting: ON"
)

print(
    "Vector search: ON"
)

print(
    "Memory: ON"
)

print(
    "Type 'clear' to clear memory."
)

print(
    "Type 'exit' to quit."
)


# ==========================================
# CHAT LOOP
# ==========================================

while True:

    question = input(
        "\nAsk the handbook: "
    )


    if question.lower() == "exit":

        print(
            "Goodbye!"
        )

        break


    if question.lower() == "clear":

        conversation_history.clear()

        print(
            "Memory cleared."
        )

        continue


    if not question.strip():

        continue


    try:

        rag(question)

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(error)