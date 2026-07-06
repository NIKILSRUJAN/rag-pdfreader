import langchain
import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

print(os.getcwd())
loader = DirectoryLoader(
    "data",
    glob= "**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False                         
)

pdf_doc = loader.load()
print(pdf_doc[3])


