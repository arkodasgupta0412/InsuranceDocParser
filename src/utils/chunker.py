from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_doc_content(doc_obj, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(doc_obj)

    return chunks