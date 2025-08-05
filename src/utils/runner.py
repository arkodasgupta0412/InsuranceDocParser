from utils import chunker, llm_infer, processor, embedder

def execute(query):
    doc_obj, questions = processor.process_query(query)
    chunks = chunker.chunk_doc_content(doc_obj, chunk_size=1200, chunk_overlap=180) 
    vectordb = embedder.get_vectorstore(chunks)
    response = llm_infer.generate_answers(questions, vectordb, topK=5, num_workers=4)

    return response