import langchain
import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
"""
    1.Loading is done using DirectoryLoader.
    2.Chunking the Doc into relevent sizes 
        we use textsplitters
"""


def pdfLoader(file_path: str) -> list:
    
    try:
        loader = DirectoryLoader(
            file_path,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=False,
        )
        pdf_doc = loader.load()
    except Exception as e:
        print(f"Either Directroy not found or there are no pdf's in dir: {type(e).__name__}: {e}")
        return []
    
    if not pdf_doc:
        print("No documents were loaded — check the directory path and PDF files.")
        return []
    

    """
    TEXT SPLITTING
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", "•", " ", ""],
    )

    chunks = text_splitter.split_documents(pdf_doc)

    for i, chunk in enumerate(chunks):
        chunk.metadata["id"] = str(uuid.uuid4())

    #for i, chunk in enumerate(chunks[:20]):
    #   # Access .page_content to avoid the TypeError
    #   chunk_id = chunk.metadata.get("id")
    #   print(f"chunk {i+1}, {chunk_id} : \n{chunk.page_content}...\n")

    return chunks


