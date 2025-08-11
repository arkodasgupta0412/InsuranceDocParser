import os, asyncio
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def get_vectorstore(chunks: List[Document]) -> Chroma:

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        task_type="RETRIEVAL_DOCUMENT"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="hackrx"
    )

    return vectordb
