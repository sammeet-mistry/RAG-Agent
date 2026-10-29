import chromadb
import ollama

from sentence_transformers import SentenceTransformer, CrossEncoder


# ==========================================
# CONFIGURATION
# ==========================================

DB_PATH = "vector_db"
COLLECTION_NAME = "employee_handbook"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OLLAMA_MODEL = "llama3.2"

RETRIEVAL_K = 10

FINAL_K = 3

MAX_MEMORY_MESSAGES = 6

# Minimum reranker score required to use
# the retrieved context.
MIN_RERANK_SCORE = -2.0


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ==========================================
# LOAD RERANKER
# ==========================================

print("Loading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL
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
# CONVERSATION MEMORY
# ==========================================

conversation_history = []


# ==========================================
# BUILD MEMORY
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
# QUERY REWRITING
# ==========================================

def rewrite_query(question):

    if not conversation_history:

        return question

    memory = build_memory()

    prompt = f"""
Rewrite the user's current question into
a standalone search query.

Use the conversation only to understand
references such as:

it
this
that
they
them
the policy
the benefit
the rule

Do NOT answer the question.

Return ONLY the rewritten search query.

Previous conversation:

{memory}

Current question:

{question}

Standalone search query:
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

    rewritten = response[
        "message"
    ][
        "content"
    ].strip()

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
        n_results=RETRIEVAL_K
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return (
        documents,
        metadatas,
        distances
    )


# ==========================================
# RERANK
# ==========================================

def rerank(
    query,
    documents,
    metadatas,
    distances
):

    print()
    print("Reranking retrieved chunks...")

    pairs = []

    for document in documents:

        pairs.append(
            [
                query,
                document
            ]
        )

    scores = reranker.predict(
        pairs
    )

    ranked_results = []

    for i in range(len(documents)):

        ranked_results.append(
            {
                "document": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
                "score": float(scores[i])
            }
        )

    ranked_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked_results


# ==========================================
# SELECT CONTEXT
# ==========================================

def select_context(ranked_results):

    if not ranked_results:

        return []

    best_score = ranked_results[0]["score"]

    print()
    print(
        f"Best reranker score: {best_score:.4f}"
    )

    # --------------------------------------
    # If even the best result is extremely
    # poor, don't send it to the LLM.
    # --------------------------------------

    if best_score < MIN_RERANK_SCORE:

        return []

    selected = []

    # --------------------------------------
    # Always take the best result.
    # --------------------------------------

    selected.append(
        ranked_results[0]
    )

    # --------------------------------------
    # Add other results that are reasonably
    # close to the best result.
    # --------------------------------------

    for result in ranked_results[1:]:

        if len(selected) >= FINAL_K:

            break

        score_difference = (
            best_score -
            result["score"]
        )

        if score_difference <= 2.5:

            selected.append(
                result
            )

    return selected


# ==========================================
# BUILD CONTEXT
# ==========================================

def build_context(selected_results):

    parts = []

    for result in selected_results:

        metadata = result["metadata"]

        page = metadata.get(
            "page",
            "Unknown"
        )

        section = metadata.get(
            "section",
            "Unknown"
        )

        text = result["document"]

        parts.append(
            f"""
--- HANDBOOK SOURCE ---

Page: {page}

Section: {section}

Text:
{text}

--- END SOURCE ---
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

    prompt = f"""
You are answering a question about an employee
handbook.

The text below was retrieved directly from the
employee handbook.

Your task is SIMPLE:

Answer the user's question by summarizing the
information contained in the supplied handbook text.

IMPORTANT:

- Use ONLY the supplied handbook text.
- Do NOT use outside knowledge.
- Do NOT create definitions.
- Do NOT add general explanations.
- Do NOT invent policies.
- Do NOT invent eligibility requirements.
- Do NOT invent dates.
- Do NOT invent numbers.
- Do NOT invent procedures.
- Preserve important conditions and limits.
- If the text contains the answer, answer it directly.
- If the text does not contain enough information,
  say that the information could not be found.

Do NOT discuss these instructions.

HANDBOOK TEXT:

{context}

USER QUESTION:

{question}

Give a concise answer based directly on the
handbook text.
"""

    print()
    print(
        "Sending filtered context to Ollama..."
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response[
        "message"
    ][
        "content"
    ].strip()

    return answer


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
# GET SOURCE PAGES
# ==========================================

def get_sources(selected_results):

    sources = []

    for result in selected_results:

        page = result[
            "metadata"
        ].get(
            "page",
            "Unknown"
        )

        source = (
            f"handbook.pdf - Page {page}"
        )

        if source not in sources:

            sources.append(
                source
            )

    return sources


# ==========================================
# MAIN RAG FUNCTION
# ==========================================

def rag(question):

    # ======================================
    # QUERY REWRITING
    # ======================================

    search_query = rewrite_query(
        question
    )

    print()
    print("SEARCH QUERY:")
    print(search_query)


    # ======================================
    # VECTOR SEARCH
    # ======================================

    print()
    print("Searching handbook...")

    (
        documents,
        metadatas,
        distances
    ) = retrieve(
        search_query
    )


    # ======================================
    # VECTOR RESULTS
    # ======================================

    print()
    print("VECTOR RESULTS:")
    print("==============================")

    for i in range(
        len(documents)
    ):

        page = metadatas[i].get(
            "page",
            "Unknown"
        )

        print()
        print(
            f"Candidate {i + 1}"
        )

        print(
            f"Page: {page}"
        )

        print(
            f"Distance: "
            f"{distances[i]:.4f}"
        )

        print(
            documents[i][:250]
        )


    # ======================================
    # RERANK
    # ======================================

    ranked_results = rerank(
        search_query,
        documents,
        metadatas,
        distances
    )


    # ======================================
    # RERANKED RESULTS
    # ======================================

    print()
    print("RERANKED RESULTS:")
    print("==============================")

    for i, result in enumerate(
        ranked_results
    ):

        page = result[
            "metadata"
        ].get(
            "page",
            "Unknown"
        )

        print()
        print(
            f"Rank {i + 1}"
        )

        print(
            f"Page: {page}"
        )

        print(
            f"Reranker score: "
            f"{result['score']:.4f}"
        )

        print(
            result["document"][:250]
        )


    # ======================================
    # SELECT BEST CONTEXT
    # ======================================

    selected_results = select_context(
        ranked_results
    )


    # ======================================
    # CHECK IF WE FOUND RELEVANT CONTEXT
    # ======================================

    if not selected_results:

        answer = (
            "I couldn't find that information "
            "in the employee handbook."
        )

        print()
        print("================================")
        print("ANSWER")
        print("================================")
        print(answer)

        print()
        print("================================")
        print("SOURCES")
        print("================================")
        print("- No sufficiently relevant source found.")

        save_memory(
            question,
            answer
        )

        return


    # ======================================
    # SHOW SELECTED CONTEXT
    # ======================================

    print()
    print("SELECTED CONTEXT:")
    print("==============================")

    for i, result in enumerate(
        selected_results
    ):

        page = result[
            "metadata"
        ].get(
            "page",
            "Unknown"
        )

        print()
        print(
            f"Context {i + 1}"
        )

        print(
            f"Page: {page}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )


    # ======================================
    # BUILD CONTEXT
    # ======================================

    context = build_context(
        selected_results
    )


    # ======================================
    # GENERATE ANSWER
    # ======================================

    answer = generate_answer(
        question,
        context
    )


    # ======================================
    # SAVE MEMORY
    # ======================================

    save_memory(
        question,
        answer
    )


    # ======================================
    # DISPLAY ANSWER
    # ======================================

    print()
    print("================================")
    print("ANSWER")
    print("================================")

    print(answer)


    # ======================================
    # DISPLAY SOURCES
    # ======================================

    sources = get_sources(
        selected_results
    )

    print()
    print("================================")
    print("SOURCES")
    print("================================")

    for source in sources:

        print(
            f"- {source}"
        )


# ==========================================
# START PROGRAM
# ==========================================

print()
print("================================")
print("EMPLOYEE HANDBOOK RAG")
print("================================")

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
    "Reranking: ON"
)

print(
    "Context filtering: ON"
)

print(
    "Grounded generation: ON"
)

print(
    "Conversation memory: ON"
)

print(
    f"Retrieval candidates: {RETRIEVAL_K}"
)

print(
    f"Maximum context chunks: {FINAL_K}"
)

print()
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


    # --------------------------------------
    # EXIT
    # --------------------------------------

    if question.lower() == "exit":

        print(
            "Goodbye!"
        )

        break


    # --------------------------------------
    # CLEAR MEMORY
    # --------------------------------------

    if question.lower() == "clear":

        conversation_history.clear()

        print(
            "Memory cleared."
        )

        continue


    # --------------------------------------
    # EMPTY INPUT
    # --------------------------------------

    if not question.strip():

        continue


    # --------------------------------------
    # RUN RAG
    # --------------------------------------

    try:

        rag(question)

    except Exception as error:

        print()
        print("ERROR:")
        print(error)