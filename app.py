from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os


loader = PyPDFLoader("/workspaces/Reading_docs_GEMINI/Reposición_ESIM_Manual.pdf")
data = loader.load()
print(data)