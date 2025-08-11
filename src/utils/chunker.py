from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_doc_content(doc_obj, chunk_size=500, chunk_overlap=100):
    """ Chunks document content using a generalized, structure-aware recursive splitter. """

    separators = [
        # Major headers (matches "Section A." or "1 PREAMBLE")
        r"\nSection [A-Z]\.",
        r"\n\d+ [A-Z ]+\n",

        # Multi-level numbered clauses and definitions
        # Matches "Def. 1.", "2.1", "3.1.1", "5.5.1", etc.
        r"\nDef\. \d+\.",
        r"\n\d+\.\d+\.\d+",
        r"\n\d+\.\d+",

        # List items
        # Matches "a)", "i.", "a.", etc.
        r"\n[a-z]\)",
        r"\n[ivx]+\.",
        r"\n[a-z]\.",

        # Standard fallbacks for any remaining text
        "\n\n",
        "\n",
        " ",
        "",
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        is_separator_regex=True,
    )

    #print(f"Starting chunking process on {len(doc_obj)} document sections...")
    chunks = splitter.split_documents(doc_obj)
    #print(f"Successfully created {len(chunks)} chunks.")
    # print(chunks[120])

    return chunks
