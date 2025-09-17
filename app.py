import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from google import genai
from google.genai.types import EmbedContentConfig

# Load env vars
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_store")
COLLECTION = os.getenv("COLLECTION", "health_reports")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not set. Please set the env variable.")
    st.stop()

# Initialize GenAI client
genai_client = genai.Client()  # uses GOOGLE_API_KEY from env

# Function to get embeddings using Gemini embedding model
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",  # use the Gemini embedding model
        client=genai_client
    )

# Function to get LLM (Gemini chat model)
def get_llm():
    # You can choose available Gemini model
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # or other model ID you have access to
        client=genai_client,
        temperature=0,  # or adjust
    )

# Ingest PDFs into vector DB
def ingest_pdfs(files):
    texts = []
    metadatas = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for f in files:
        reader = PdfReader(f)
        full_text_pages = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                full_text_pages.append(txt)
        full_text = "\n".join(full_text_pages)
        if not full_text.strip():
            continue
        chunks = splitter.split_text(full_text)
        for i, ch in enumerate(chunks):
            texts.append(ch)
            metadatas.append({"file": f.name, "chunk": i})

    if not texts:
        return None

    embeddings = get_embeddings()
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    return vectordb

# Load or create vector store
def load_chroma():
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

# Streamlit UI
st.title("📄 Health Report RAG with Gemini")

uploaded = st.file_uploader("Upload health report PDFs", accept_multiple_files=True, type=["pdf"])
if uploaded and st.button("Ingest to Vector DB"):
    vectordb = ingest_pdfs(uploaded)
    if vectordb:
        st.success("✅ Ingested successfully!")
    else:
        st.warning("⚠️ No text extracted or no PDFs provided.")

# Load existing DB
try:
    vectordb = load_chroma()
except Exception as e:
    st.error(f"Failed to load vector DB: {e}")
    vectordb = None

if vectordb:
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    llm = get_llm()

    qa_chain = None
    if llm:
        from langchain.chains import ConversationalRetrievalChain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
    else:
        st.warning("⚠️ LLM (Gemini chat) not initialized. Check your API key / model permissions.")

    if qa_chain:
        if "history" not in st.session_state:
            st.session_state.history = []

        query = st.text_input("Ask a question about the reports:")
        if st.button("Ask") and query:
            try:
                result = qa_chain({"question": query, "chat_history": st.session_state.history})
                answer = result["answer"]
                st.subheader("Answer")
                st.write(answer)

                # show sources
                with st.expander("Sources"):
                    for doc in result["source_documents"]:
                        fname = doc.metadata.get("file", "Unknown file")
                        chunk_idx = doc.metadata.get("chunk", None)
                        st.write(f"📄 {fname} — chunk {chunk_idx}")
                        st.text(doc.page_content[:500] + "...")
                
                # update history
                st.session_state.history.append((query, answer))

            except Exception as e:
                st.error(f"Error during QA: {e}")

