from flask import Flask, render_template, request, jsonify
import subprocess
import sys
import os


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RAG_FILE = os.path.join(
    BASE_DIR,
    "rag.py"
)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# RUN RAG
# ==========================================

def run_rag(question):

    try:

        # ----------------------------------
        # Run your existing rag.py
        # ----------------------------------

        result = subprocess.run(
            [
                sys.executable,
                "-u",
                RAG_FILE
            ],

            input=(
                question
                + "\n"
                + "exit"
                + "\n"
            ),

            capture_output=True,

            text=True,

            cwd=BASE_DIR,

            timeout=180
        )


        output = result.stdout

        error = result.stderr


        # ----------------------------------
        # DEBUG OUTPUT IN TERMINAL
        # ----------------------------------

        print()
        print("================================")
        print("RAG OUTPUT")
        print("================================")
        print(output)

        if error:

            print()
            print("================================")
            print("RAG ERROR")
            print("================================")
            print(error)


        # ----------------------------------
        # CHECK FOR PROCESS ERROR
        # ----------------------------------

        if result.returncode != 0:

            return (
                "There was an error while "
                "running the handbook AI.\n\n"
                + error
            ), []


        # ----------------------------------
        # EXTRACT ANSWER
        # ----------------------------------

        answer = extract_answer(
            output
        )


        # ----------------------------------
        # EXTRACT SOURCES
        # ----------------------------------

        sources = extract_sources(
            output
        )


        return answer, sources


    except subprocess.TimeoutExpired:

        return (
            "The handbook AI took too long "
            "to respond. Please try again."
        ), []


    except Exception as e:

        print()
        print("RAG CONNECTION ERROR:")
        print(e)

        return (
            "Something went wrong while "
            "connecting to the handbook AI."
        ), []


# ==========================================
# EXTRACT ANSWER
# ==========================================

def extract_answer(output):

    # --------------------------------------
    # Look for ANSWER section
    # --------------------------------------

    if "ANSWER" in output:

        answer_part = output.split(
            "ANSWER",
            1
        )[1]


        # ----------------------------------
        # Remove separator lines
        # ----------------------------------

        answer_part = (
            answer_part
            .replace(
                "================================",
                ""
            )
            .strip()
        )


        # ----------------------------------
        # Remove SOURCES section
        # ----------------------------------

        if "SOURCES" in answer_part:

            answer_part = answer_part.split(
                "SOURCES",
                1
            )[0]


        # ----------------------------------
        # Remove SOURCE PAGE if present
        # ----------------------------------

        if "SOURCE PAGE:" in answer_part:

            answer_part = answer_part.split(
                "SOURCE PAGE:",
                1
            )[0]


        answer = answer_part.strip()


        if answer:

            return answer


    # --------------------------------------
    # Fallback
    # --------------------------------------

    return (
        "I couldn't generate an answer "
        "from the handbook."
    )


# ==========================================
# EXTRACT SOURCES
# ==========================================

def extract_sources(output):

    sources = []


    # --------------------------------------
    # Find SOURCES section
    # --------------------------------------

    if "SOURCES" not in output:

        return sources


    sources_part = output.split(
        "SOURCES",
        1
    )[1]


    # --------------------------------------
    # Remove separator lines
    # --------------------------------------

    sources_part = (
        sources_part
        .replace(
            "================================",
            ""
        )
        .strip()
    )


    # --------------------------------------
    # Read source lines
    # --------------------------------------

    for line in sources_part.splitlines():

        line = line.strip()


        if not line:

            continue


        if line.startswith("-"):

            source = line[1:].strip()

            if source:

                sources.append(
                    source
                )


    return sources


# ==========================================
# ASK HANDBOOK
# ==========================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "answer": "No question received.",
                "sources": []
            })


        question = data.get(
            "question",
            ""
        ).strip()


        if not question:

            return jsonify({
                "answer": "Please enter a question.",
                "sources": []
            })


        print()
        print("================================")
        print("QUESTION FROM WEBSITE")
        print("================================")
        print(question)


        # ----------------------------------
        # RUN ACTUAL RAG
        # ----------------------------------

        answer, sources = run_rag(
            question
        )


        # ----------------------------------
        # SEND ANSWER TO WEBSITE
        # ----------------------------------

        return jsonify({

            "answer": answer,

            "sources": sources

        })


    except Exception as e:

        print()
        print("FLASK ERROR:")
        print(e)


        return jsonify({

            "answer": (
                "An error occurred while "
                "processing your question."
            ),

            "sources": []

        }), 500


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    print()
    print("================================")
    print("EMPLOYEE HANDBOOK AI")
    print("================================")

    print()
    print("RAG FILE:")
    print(RAG_FILE)

    print()
    print("Website:")
    print("http://127.0.0.1:5000")

    print()
    print("================================")
    print()


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )

