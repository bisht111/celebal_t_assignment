import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain.chains import RetrievalQA

# -----------------------------
# Load Environment Variables
# -----------------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# -----------------------------
# Load PDF
# -----------------------------

PDF_PATH = "data/WEEK01.pdf" # pkease enter according you file name 

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"{PDF_PATH} not found.")

print("Loading PDF...")

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"Loaded {len(documents)} pages")

# -----------------------------
# Split into Chunks
# -----------------------------

print("Splitting document...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

docs = text_splitter.split_documents(documents)

print(f"Created {len(docs)} chunks")

# -----------------------------
# Create Embeddings
# -----------------------------

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Create / Load FAISS
# -----------------------------

DB_PATH = "faiss_index"

if os.path.exists(DB_PATH):
    print("Loading existing FAISS index...")

    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

else:
    print("Building FAISS index...")

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    vectorstore.save_local(DB_PATH)

    print("FAISS index saved.")

# -----------------------------
# Retriever
# -----------------------------

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# -----------------------------
# QA Chain
# -----------------------------

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

print("\n==============================")
print(" Document Question Answering ")
print("==============================")

while True:

    question = input("\nAsk a question (type exit): ")

    if question.lower() == "exit":
        break

    result = qa.invoke({"query": question})

    print("\nAnswer:\n")
    print(result["result"])

    print("\nRetrieved Chunks:\n")

    for i, doc in enumerate(result["source_documents"], 1):
        print("=" * 60)
        print(f"Chunk {i}")
        print("=" * 60)
        print(doc.page_content[:700])