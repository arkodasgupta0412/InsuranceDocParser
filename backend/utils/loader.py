import os
import requests
import tempfile
from urllib.parse import urlparse
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader, TextLoader
from langchain.schema import Document
from typing import List


def get_file_extension_from_url(url: str) -> str:
    """Extracts the file extension from a URL."""

    try:
        path = urlparse(url).path
        return os.path.splitext(path)[1].lower()
    
    except Exception:
        return ""



def load_document_from_url(url: str) -> List[Document]:
    """ Loads a document from a URL, determines its type, and uses the appropriate LangChain document loader. """

    file_extension = get_file_extension_from_url(url)
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content = response.content

    except requests.RequestException as e:
        print(f"Error fetching URL '{url}': {e}")
        return []


    temp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
    temp_path = temp_file.name

    try:
        temp_file.write(content)
        temp_file.close()

        loader = None

        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_path)

        elif file_extension == ".docx":
            loader = Docx2txtLoader(temp_path)

        elif file_extension in [".eml", ".msg"]:
            loader = UnstructuredEmailLoader(temp_path)

        else:
            print(f"Warning: Unsupported file type '{file_extension}'. Treating as plain text.")
            loader = TextLoader(temp_path, encoding='utf-8', autodetect_encoding=True)

        # print(f"Loading document with {loader.__class__.__name__}...")
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = url

        return documents

    except Exception as e:
        print(f"Error loading document: {e}")
        return []
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)