# PolicyPal

**PolicyPal** is an AI-powered insurance document parser and query assistant.  
It uses a Retrieval-Augmented Generation (RAG) pipeline to read, index and understand an insurance policy document, enabling human-like responses to user queries about coverage, terms, and conditions.

## Features
- **Document Parsing** – Extracts structured and unstructured information from insurance policy PDFs.
- **RAG Pipeline** – Uses embeddings + vector search for accurate context retrieval.
- **Conversational Agent** – Answers queries in a human tone while staying grounded to the policy content.
- **Backend API** – Deployed and accessible as an endpoint (no frontend yet).

## Tech Stack
- **Backend**: Python, FastAPI, langchain
- **AI**: LLM (Gemini-2.5-Flash) + Vector Database (Chroma)
- **Deployment**: Render
- **Testing**: (Postman) - validate HTTP get/post requests

## Usage
1. Upload your insurance policy document to the backend endpoint. (using a link)
2. Ask natural language questions about your policy.
3. Receive accurate, human-like responses grounded in the document.

---

## Future Improvements
- Add frontend for user-friendly access
- Multi-document support
- Summarization and coverage comparison

---
