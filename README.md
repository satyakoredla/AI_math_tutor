# MathMentor AI — Multimodal JEE Math Tutor

## Overview

MathMentor AI is an end‑to‑end AI system that solves JEE‑style math problems using a multi‑agent architecture combined with Retrieval Augmented Generation (RAG), symbolic mathematics, and memory. The application supports text, image, and audio inputs and produces step‑by‑step explanations similar to a human tutor.

The system is designed to demonstrate practical AI engineering concepts including multimodal processing, agent pipelines, retrieval‑based reasoning, human‑in‑the‑loop verification, and memory‑driven learning.

---

# System Architecture

User Input (Text / Image / Audio)
→ Multimodal Processing (OCR / Speech Recognition)
→ Parser Agent
→ Intent Router Agent
→ RAG Retrieval (Knowledge Base)
→ Solver Agent
→ Verifier Agent
→ Explainer Agent
→ Memory Storage
→ Streamlit UI Output

---

# Implementation Workflow

## 1. Multimodal Input Handling

The system accepts three types of user input:

### Text Input

Users can directly type math problems in the interface.
Example:

```
Solve: x^2 - 5x + 6 = 0
```

### Image Input

Users upload a photo or screenshot of a math question.

Workflow:

1. Image uploaded through Streamlit UI
2. OCR extracts text from the image
3. Extracted text is shown to the user
4. User can edit or approve the text

Tool used:

* EasyOCR

### Audio Input

Users can upload an audio recording containing a math question.

Workflow:

1. Audio uploaded
2. Speech‑to‑text conversion
3. Transcript displayed for confirmation

Tool used:

* Whisper speech recognition model

---

## 2. Parser Agent

Purpose: Convert raw text into structured problem format.

Responsibilities:

* Clean OCR or ASR output
* Identify math topic
* Extract variables
* Detect ambiguity

Example Output:

```
{
 "problem_text": "Find derivative of x^3 + 2x^2",
 "topic": "calculus",
 "variables": ["x"],
 "method": "derivative",
 "needs_clarification": false
}
```

Model Used:

* HuggingFace LLM (Mistral / Llama / Gemma)

---

## 3. Intent Router Agent

Purpose: Determine the correct solving strategy.

Responsibilities:

* Classify problem type
* Select solving method
* Generate RAG query

Example:

Problem: derivative of x^3

Router Output:

```
topic: calculus
solver_mode: symbolic_math
rag_query: derivative rules
```

---

## 4. Retrieval Augmented Generation (RAG)

The system uses a curated knowledge base containing mathematical formulas and concepts.

Knowledge Base Examples:

* Algebra formulas
* Derivative rules
* Integral formulas
* Probability identities
* Trigonometric identities

### RAG Workflow

1. Knowledge documents are chunked
2. Embeddings generated
3. Stored in vector database
4. Query retrieves top‑k relevant formulas

Example retrieved context:

```
Derivative rule:
d/dx(x^n) = n*x^(n-1)
```

---

## 5. Solver Agent

Purpose: Compute the solution using symbolic mathematics and reasoning.

The solver combines:

* SymPy symbolic math engine
* HuggingFace language model

Example using SymPy:

```
from sympy import symbols, diff
x = symbols('x')
diff(x**3 + 2*x**2, x)
```

Output:

```
3x^2 + 4x
```

---

## 6. Verifier Agent

Purpose: Validate the correctness of the computed solution.

Checks performed:

* Mathematical correctness
* Domain constraints
* Invalid numeric results (NaN / infinity)
* Edge cases

If confidence is low, the system triggers Human‑in‑the‑Loop verification.

---

## 7. Explainer Agent

Purpose: Generate step‑by‑step explanation for students.

Example Output:

Step 1: Apply derivative rule

```
d/dx(x^n) = n*x^(n-1)
```

Step 2:

```
d/dx(x^3) = 3x^2
```

Step 3:

Final Answer:

```
3x^2 + 4x
```

---

## 8. Human‑in‑the‑Loop (HITL)

HITL is triggered when:

* OCR confidence is low
* Speech transcription unclear
* Parser detects ambiguity
* Verifier confidence below threshold

User actions:

* Approve result
* Edit extracted text
* Provide corrected answer

---

## 9. Memory System

The system stores solved problems to improve future responses.

Stored Data:

* Original question
* Parsed structure
* Retrieved documents
* Final answer
* Explanation
* User feedback

Memory enables:

* Reusing previous solutions
* Faster responses
* Learning from corrections

Database:

* SQLite

---

## 10. Streamlit Application UI

Features:

* Input selector (Text / Image / Audio)
* OCR / Transcript preview
* Agent pipeline trace
* Retrieved knowledge panel
* Final answer display
* Confidence score
* Step‑by‑step explanation
* User feedback buttons

---

# Tech Stack

## Programming Language

* Python

## AI Models

Gemini flash
## Embeddings

* sentence‑transformers/all‑MiniLM‑L6‑v2

## Symbolic Math

* SymPy

## OCR

* EasyOCR

## Speech Recognition

* Whisper

## Vector Database

* FAISS

## Backend Framework

* Python

## Frontend

* Streamlit

## Database

* SQLite

## Deployment

* HuggingFace Spaces or Streamlit Cloud

---

# Project Structure

```
math‑mentor‑ai
│
├── app.py
├── requirements.txt
├── .env.example
│
├── agents
│   ├── parser_agent.py
│   ├── router_agent.py
│   ├── solver_agent.py
│   ├── verifier_agent.py
│   └── explainer_agent.py
│
├── rag
│   ├── embeddings.py
│   └── retriever.py
│
├── input_processing
│   ├── image_ocr.py
│   └── speech_to_text.py
│
├── memory
│   └── memory_store.py
│
└── data
    └── math_docs
```

---

# Setup Instructions

### 1 Install dependencies

```
pip install -r requirements.txt
```

### 2 Add environment variables

Create `.env` file:

```
HUGGINGFACEHUB_API_TOKEN=your_token
```

### 3 Run the application

```
streamlit run app.py
```

---

# Demo Capabilities

The application demonstrates:

* Image to math solution
* Audio to math solution
* RAG‑based formula retrieval
* Multi‑agent reasoning
* Human‑in‑the‑loop correction
* Memory reuse for repeated problems

---

# Future Improvements

* Support handwritten math recognition
* Add graph visualization for functions
* Improve math reasoning models
* Add evaluation benchmarks

---

# Author

AI Engineer Assignment — Multimodal Math Mentor
