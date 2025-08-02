from fastapi import HTTPException
import schema
import requests
import utils.extractor as extractor

def process_query(query: schema.Query):
    try:
        response = requests.get(query.documents)
        response.raise_for_status()
        questions = query.questions

        content_type = response.headers.get('content-type', '')
        url_lower = query.documents.lower()

        # Choose extraction method
        if 'pdf' in url_lower or 'application/pdf' in content_type:
            doc = extractor.load_pdf_from_bytes(response.content)

        elif 'docx' in url_lower or 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
            doc = extractor.load_docx_from_bytes(response.content)

        elif 'eml' in url_lower or 'message/rfc822' in content_type:
            doc = extractor.load_email_from_eml_bytes(response.content)

        elif 'msg' in url_lower or 'application/vnd.ms-outlook' in content_type:
            doc = extractor.load_email_from_msg_bytes(response.content)

        else:
            doc = [extractor.Document(page_content=response.content.decode("utf-8", errors="ignore"), metadata={"source": "text"})]

        return doc, questions
    

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
