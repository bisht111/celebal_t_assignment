# Document Question Answering System using RAG

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that answers user questions based on the content of PDF documents.

Instead of relying only on the language model's pre-trained knowledge, the system first retrieves relevant information from the uploaded document and then generates an answer using that retrieved context. This approach improves answer accuracy and allows users to ask questions about private or custom documents.

---

# What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI technique that combines:

- **Retrieval** – finding the most relevant information from a document.
- **Generation** – using a Large Language Model (LLM) to generate an answer based on the retrieved information.

Unlike a normal chatbot, a RAG system does not depend only on the model's memory. It first searches the document and then answers using the retrieved content.

---

# Objectives

- Understand the concept of Retrieval-Augmented Generation.
- Perform question answering over PDF documents.
- Learn document chunking and embeddings.
- Build a vector database using FAISS.
- Improve answer accuracy using retrieved context.

---

# Technologies Used

- Python
- LangChain
- FAISS
- Hugging Face Embeddings
- Groq API (Llama 3)
- PyPDF
- Sentence Transformers

---

# Project Structure

```
Document-QA-RAG/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── data/
│      notes.pdf
│
└── faiss_index/
```

---

# How the System Works

## Step 1: Load Document

The system loads the PDF document using **PyPDFLoader**.

↓

## Step 2: Text Chunking

The document is divided into smaller chunks.

Large documents cannot be processed efficiently at once, so chunking improves retrieval performance.

↓

## Step 3: Create Embeddings

Each chunk is converted into a numerical vector using the Hugging Face embedding model.

These vectors represent the semantic meaning of the text.

↓

## Step 4: Store in FAISS

The embeddings are stored inside a FAISS vector database.

This enables very fast similarity search.

↓

## Step 5: User Query

The user asks a question.

↓

## Step 6: Retrieve Relevant Chunks

The question is converted into an embedding.

FAISS compares it with all stored vectors and retrieves the most relevant document chunks.

↓

## Step 7: Generate Answer

The retrieved chunks are sent to the Llama language model.

The model generates a final answer using only the retrieved information.

---

# Workflow

```
PDF Document
      │
      ▼
Load PDF
      │
      ▼
Split into Chunks
      │
      ▼
Generate Embeddings
      │
      ▼
Store in FAISS
      │
      ▼
User Question
      │
      ▼
Similarity Search
      │
      ▼
Retrieve Relevant Chunks
      │
      ▼
Large Language Model
      │
      ▼
Generated Answer
```

---

# What is FAISS?

FAISS (Facebook AI Similarity Search) is a vector database developed by Meta.

Its purpose is to quickly search millions of vector embeddings and return the most similar ones.

Without FAISS, the system would have to compare every document chunk manually, which would be much slower.

---

# Is the FAISS Index Created Automatically?

Yes.

The first time you run the application:

- The PDF is loaded.
- Text is split into chunks.
- Embeddings are generated.
- A FAISS index is created automatically.
- The index is saved inside the **faiss_index/** folder.

On future runs, the saved FAISS index is loaded directly, avoiding the need to regenerate embeddings every time.

---

# Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

---

# API Key Setup

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

You can obtain a free API key from:

https://console.groq.com/

---

# Running the Project

Place your PDF inside:

```
data/
```

Run:

```bash
python app.py
```

Example:

```
Ask a question:
What are the objectives of this project?
```

Output:

```
The objectives are:

• Understand Retrieval-Augmented Generation
• Learn document embeddings
• Build a vector database
• Generate answers using retrieved context
```

---

# Features

- PDF Question Answering
- Automatic Document Chunking
- Semantic Search
- FAISS Vector Database
- Hugging Face Embeddings
- Llama 3 Answer Generation
- Automatic FAISS Index Creation
- Interactive Command Line Interface

---

# Future Improvements

- Support multiple PDF documents
- Hybrid keyword and vector search
- Re-ranking retrieved documents
- Web interface using Streamlit
- Chat history support
- OCR support for scanned PDFs

---

# Applications

- Enterprise Knowledge Base
- Research Paper Assistant
- Resume Question Answering
- Educational Chatbots
- AI Documentation Assistant
- Customer Support Systems

---

# Conclusion

This project demonstrates how Retrieval-Augmented Generation combines semantic search and Large Language Models to answer questions based on custom documents.

By retrieving relevant document chunks before generating responses, the system produces more accurate and context-aware answers than a traditional language model.