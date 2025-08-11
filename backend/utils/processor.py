from fastapi import HTTPException
import schema
import utils.loader as loader


def process_query(query: schema.Query):
    """ Processes a query by loading a document from a URL """

    try:
        documents = loader.load_document_from_url(query.documents)
        questions = query.questions

        #print(len(documents))
        #print(documents[21].page_content)
        #print(documents[21].metadata)

        if not documents:
            raise HTTPException(status_code=400, detail="Failed to load or process the document from the provided URL. The link may be invalid or the file type unsupported.")

        return documents, questions


    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        
        print(f"An unexpected error occurred in process_query: {e}")
        

        raise HTTPException(status_code=500, detail=f"An unexpected internal error occurred.")