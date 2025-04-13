import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Load CSV and clean ===
df = pd.read_csv("csuf_course_professor_merged.csv")
df.columns = df.columns.str.strip()
df = df.fillna('')

# === Convert rows to LangChain Documents ===
def row_to_doc(row):
    text = "\n".join([f"{col}: {row[col]}" for col in row.index])
    return Document(page_content=text)

documents = [row_to_doc(row) for _, row in df.iterrows()]

# === Chunk documents ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = splitter.split_documents(documents)

# === Embed and save to Chroma ===
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(splits, embedding=embeddings, persist_directory="chroma_db")
vectorstore.persist()

print("? Chroma vectorstore created with", len(splits), "chunks.")
